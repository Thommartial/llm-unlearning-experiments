"""Revision campaign runner: resumable (model, seed, variant) units.

A campaign config (`configs/revision/*.yaml`) is an experiment config plus a
`campaign:` block:

    campaign:
      models: [distilgpt2, gpt2]
      seeds: [0, 1, 2, 3, 4]
      requires: [unlearn.finetune_lr]      # calibration values that must be set
      variants:
        - {name: central, kind: central}
        - {name: iid_K5, kind: federated, federated: {num_clients: 5}}
        - {name: lora, kind: central, models: [tiny_llama], model: {lora: {}}}

A variant overrides any config section (model, data, unlearn, attack,
federated) and may restrict the models. `central` runs `central.run_central`
(all methods), `federated` runs `federated.run_fed` (retrain, client_ga).

A unit writes all its rows at once (atomic rewrite of the CSV) and is done
when the CSV holds a row for every expected method; a rerun skips done units,
so a campaign is resumed by running it again. Every row starts with the
campaign, variant, kind, git commit, a config hash and the device; the
metrics follow (including `gpu_name`, `torch_version`, `cuda_version`). Raw
per-record scores go to `<out_dir>/scores/<campaign>/` and the resolved
config of each unit to `<out_dir>/configs/<campaign>/`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import yaml

from .config import (
    AttackConfig,
    DataConfig,
    ExperimentConfig,
    FederatedConfig,
    ModelConfig,
    UnlearnConfig,
    load_config,
)

SECTIONS = {"model": ModelConfig, "data": DataConfig, "unlearn": UnlearnConfig, "attack": AttackConfig,
            "federated": FederatedConfig}
KINDS = ("central", "federated")


@dataclass
class Unit:
    model: str
    seed: int
    variant: str
    kind: str
    overrides: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int, str]:
        return self.model, self.seed, self.variant

    @property
    def slug(self) -> str:
        return f"{self.model.replace('/', '_')}_seed{self.seed}_{self.variant}"


def read_campaign(path) -> dict:
    raw = yaml.safe_load(Path(path).read_text())
    if "campaign" not in raw:
        raise ValueError(f"{path} has no campaign block.")
    return raw


def check_calibrated(raw: dict) -> None:
    """Refuse to run while a value marked `# set from pilot` is still null."""
    missing = []
    for dotted in raw["campaign"].get("requires", []):
        section, key = dotted.split(".")
        if raw.get(section, {}).get(key) is None:
            missing.append(dotted)
    if missing:
        raise SystemExit(f"Set {missing} from the pilot (marked '# set from pilot' in the config) before running.")


def units(raw: dict, models=None, seeds=None) -> list[Unit]:
    """Every (model, seed, variant) of the campaign, optionally restricted to
    `models` and `seeds`; ordered by seed, then variant, then model."""
    camp = raw["campaign"]
    out = []
    for seed in camp["seeds"]:
        if seeds is not None and seed not in seeds:
            continue
        for v in camp["variants"]:
            kind = v.get("kind", "central")
            if kind not in KINDS:
                raise ValueError(f"Unknown variant kind '{kind}'.")
            overrides = {k: val for k, val in v.items() if k in SECTIONS}
            for model in v.get("models", camp["models"]):
                if models is None or model in models:
                    out.append(Unit(model, int(seed), v["name"], kind, overrides))
    return out


def resolve(path, unit: Unit, device: str | None = None) -> ExperimentConfig:
    """The experiment config of one unit: the file, then the variant's
    overrides, then the unit's model and seed (and `device` if given)."""
    cfg = load_config(path)
    for section, values in unit.overrides.items():
        setattr(cfg, section, replace(getattr(cfg, section), **values))
    cfg.model = replace(cfg.model, name=unit.model)
    cfg.seed = unit.seed
    cfg.name = f"{cfg.name}_{unit.slug}"
    if device is not None:
        cfg.device = device
    return cfg


def config_hash(cfg: ExperimentConfig) -> str:
    """Hash of the resolved config without its seed, name and device: equal
    for every seed of a (model, variant) run with the same settings."""
    d = asdict(cfg)
    for k in ("seed", "name", "device", "output_dir"):
        d.pop(k)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def commit_info() -> dict:
    """Git commit of the working tree (`REPO_COMMIT` if git is unavailable)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                    text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = os.environ.get("REPO_COMMIT", "unknown"), None
    return {"commit": commit, "commit_dirty": dirty}


def expected_methods(kind: str) -> tuple[str, ...]:
    from . import central, federated

    return central.METHODS if kind == "central" else tuple(federated.VIEWS)


def read_rows(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows: list[dict], replace_keys=()) -> None:
    """Add `rows` to the CSV at `path`, first dropping rows whose (model,
    seed, variant) is in `replace_keys`; the header is the union of all
    columns. Written to a temporary file and renamed, so an interrupted
    write leaves the old file intact."""
    path = Path(path)
    drop = {(m, str(s), v) for m, s, v in replace_keys}
    old = [r for r in read_rows(path) if (r["model"], r["seed"], r["variant"]) not in drop]
    fields = list(dict.fromkeys(k for r in [*old, *rows] for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(old)
        w.writerows(rows)
    os.replace(tmp, path)


def done(rows: list[dict], unit: Unit, campaign: str) -> bool:
    have = {r["method"] for r in rows if r.get("campaign") == campaign
            and (r["model"], r["seed"], r["variant"]) == (unit.model, str(unit.seed), unit.variant)}
    return set(expected_methods(unit.kind)) <= have


def missing(paths, unit_list: list[Unit], campaign: str) -> list[Unit]:
    """Units without a complete set of rows in any of the CSVs at `paths`
    (how a campaign's success is judged: by its result files)."""
    rows = [r for p in paths for r in read_rows(p)]
    return [u for u in unit_list if not done(rows, u, campaign)]


def run_unit(path, unit: Unit, campaign: str, device: str, out_dir) -> list[dict]:
    """Run one unit; save its scores and resolved config; return its rows."""
    from . import central, federated, unlearn

    cfg = resolve(path, unit, device)
    dev = unlearn.resolve_device(cfg)
    if unit.kind == "central":
        result = central.run_central(cfg, dev)
        rows = central.central_rows(cfg, result)
    else:
        result = federated.run_fed(cfg, dev)
        rows = federated.fed_rows(cfg, result)
    out_dir = Path(out_dir)
    central.save_scores(result, out_dir / "scores" / campaign / f"{unit.slug}.npz")
    conf = out_dir / "configs" / campaign / f"{unit.slug}.json"
    conf.parent.mkdir(parents=True, exist_ok=True)
    conf.write_text(json.dumps(asdict(cfg), indent=2))
    head = {"campaign": campaign, "variant": unit.variant, "kind": unit.kind, **commit_info(),
            "config_hash": config_hash(cfg), "device": dev}
    return [{**head, **r} for r in rows]


def run_campaign(path, campaign: str, out, models=None, seeds=None, device: str | None = None,
                 out_dir="results/revision") -> list[Unit]:
    """Run every unit not yet done in `out`; returns the units that ran."""
    raw = read_campaign(path)
    check_calibrated(raw)
    ran = []
    for unit in units(raw, models, seeds):
        if done(read_rows(out), unit, campaign):
            print(f"[{campaign}] skip {unit.slug} (done)", flush=True)
            continue
        print(f"[{campaign}] run {unit.slug}", flush=True)
        rows = run_unit(path, unit, campaign, device, out_dir)
        write_rows(out, rows, replace_keys=[unit.key])
        ran.append(unit)
        print(f"[{campaign}] done {unit.slug}: {len(rows)} rows -> {out}", flush=True)
    return ran


def merge(paths, out) -> int:
    """Concatenate per-process CSVs into one (union of columns); returns the row count."""
    rows = [r for p in paths for r in read_rows(p)]
    Path(out).unlink(missing_ok=True)  # overwrite, never append to an earlier merge
    write_rows(out, rows)
    return len(rows)


# pilot value -> config files and dotted keys it fills (lines marked "# set from pilot")
CALIBRATION = {
    "finetune_lr": [("central.yaml", "unlearn.finetune_lr"), ("fed_iid.yaml", "unlearn.finetune_lr"),
                    ("fed_ext.yaml", "unlearn.finetune_lr")],
    "rounds": [("fed_iid.yaml", "federated.rounds"), ("fed_ext.yaml", "federated.rounds")],
    "rho_factor": [("fed_iid.yaml", "federated.rho_factor"), ("fed_ext.yaml", "federated.rho_factor")],
    "lora_finetune_lr": [("lora.yaml", "unlearn.finetune_lr")],
    "lora_learning_rate": [("lora.yaml", "unlearn.learning_rate")],
}
LORA_CALIBRATION = ("lora_finetune_lr", "lora_learning_rate")  # what a LoRA-only report (`only: lora`) fills


def _yaml_value(v) -> str:
    """A pilot value as YAML: ints as is, floats to 6 significant digits in a
    form YAML reads as a float (PyYAML needs the dot in 5.0e-05)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"not a number: {v!r}")
    if isinstance(v, int):
        return str(v)
    return yaml.safe_dump(float(f"{v:.6g}")).split("\n")[0]


def apply_calibration(report_path, configs_dir="configs/revision") -> str:
    """Fill the `# set from pilot` values of the campaign configs from the
    pilot's `pilot_calibration.json` (or, for a LoRA-only report with
    `"only": "lora"`, lora.yaml's two values); returns the unified diff.
    LoRA values measured on another model than lora.yaml's are refused.

    Only lines that carry the `# set from pilot` comment, under the right
    top-level section, are changed (value only; comments and layout stay).
    Applying again replaces the earlier values. Every file is checked first
    (each target line found, every `campaign.requires` set); on an error
    nothing is written.
    """
    import difflib
    import re

    report = json.loads(Path(report_path).read_text())
    selected = report["selected"]
    names = LORA_CALIBRATION if report.get("only") == "lora" else tuple(CALIBRATION)
    missing_values = sorted(set(names) - set(selected))
    if missing_values:
        raise ValueError(f"{report_path} lacks {missing_values}")
    measured = report.get("lora_model") or report.get("measurements", {}).get("lora_weight_load", {}).get("model")
    lora_model = yaml.safe_load((Path(configs_dir) / "lora.yaml").read_text())["model"]["name"]
    if measured is not None and measured != lora_model:
        raise ValueError(f"{report_path}: the LoRA rates were measured on {measured}, but lora.yaml runs "
                         f"{lora_model}; run the LoRA pilot (--campaign pilot --only lora) for it")
    edits: dict[str, dict[tuple[str, str], str]] = {}
    for name in names:
        for file, dotted in CALIBRATION[name]:
            edits.setdefault(file, {})[tuple(dotted.split("."))] = _yaml_value(selected[name])
    diff, writes = [], {}
    for file, keys in edits.items():
        path = Path(configs_dir) / file
        old = path.read_text().splitlines(keepends=True)
        new, section, done = [], None, set()
        for line in old:
            top = re.match(r"^(\w+):", line)
            if top:
                section = top.group(1)
            m = re.match(r"^(\s+)(\w+):(\s*)(\S+)(\s*)(# set from pilot.*\n?)$", line)
            if m and (section, m.group(2)) in keys:
                value = keys[(section, m.group(2))]
                gap = " " * max(1, len(m.group(5)) + len(m.group(4)) - len(value))  # keep the comment column
                line = f"{m.group(1)}{m.group(2)}:{m.group(3)}{value}{gap}{m.group(6)}"
                done.add((section, m.group(2)))
            new.append(line)
        if done != set(keys):
            raise ValueError(f"{path}: no '# set from pilot' line for {sorted(set(keys) - done)}")
        check_calibrated(yaml.safe_load("".join(new)))
        if new != old:
            writes[path] = "".join(new)
            diff += difflib.unified_diff(old, new, fromfile=str(path), tofile=str(path))
    for path, text in writes.items():  # all files checked first: nothing is written on an error
        path.write_text(text)
    return "".join(diff)
