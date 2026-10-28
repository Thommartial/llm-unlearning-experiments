"""Pilot: the calibration of ANALYSIS_PLAN §7 and the timing runs, one process.

Steps, in order (the first fails early if the LoRA model's weights do not load):
  0. Load the LoRA model's pre-trained weights and score a few records.
  1. Rule 1, full fine-tuning (GPT-2, seed 0): for each learning rate, fine-tune
     the base model on all members for `epochs` epochs; keep the rate with
     the lower retain loss among those whose world-facts loss is at most
     (1 + `world_facts_tolerance`) x the base model's.
  2. Rule 2: FedAvg (K clients, iid) with that rate, one round at a time;
     R = the smallest grid value whose retain loss is within
     `retain_tolerance` of the centrally fine-tuned model's.
  3. Rule 3: from θ_pre after R rounds, client GA with rho = factor x the
     norm of the last round's update, then the recovery rounds; the factor
     is the smallest whose post-recovery forget loss exceeds the forget loss
     of the federated retrain (R rounds over clients 1..K-1).
  4. Rule 1 for LoRA (`lora_model`, e.g. Pythia-1B) over its own grid.
  5. Sanity check (report only): GA at 1e-5 x (LoRA rate / full rate) on the
     LoRA fine-tuned model; flagged unless the forget loss rises by at
     least `ga_min_rise` nats.
  6. Timing: run_central and run_fed (iid) for GPT-2 seed 0 with the
     selected values, alone in this process; their rows (with every phase
     time) go to the pilot CSV.
`only="lora"` (`--campaign pilot --only lora`) runs steps 0, 4 and 5 with the
full rate `full_finetune_lr` (from the full pilot), then times one central
LoRA unit with the selected LoRA values; the report goes to
`pilot_lora_calibration.json` (with `"only": "lora"`) and the rows to
`pilot_lora.csv`, so the full pilot's files stay as they are.
When no grid value meets a rule, the last value is kept and a flag is
raised. Every phase is timed (`costs.phase`). The selected values, every
measurement and the flags go to `pilot_calibration.json` and are printed
with the config lines to paste.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import yaml

LOSS_RECORDS = 16  # step 0: records scored after loading the LoRA model's weights
GA_MIN_RISE = 0.05  # step 5: smallest forget-loss rise (nats) that passes, unless pilot.ga_min_rise is set
ONLY = ("lora",)


def _with(cfg, **sections):
    """`cfg` with fields of its sections replaced: _with(cfg, unlearn={"finetune_lr": 1e-5})."""
    return replace(cfg, **{s: replace(getattr(cfg, s), **v) for s, v in sections.items()})


class _Log:
    def __init__(self, device: str):
        self.device = device
        self.report = {"phases": {}, "measurements": {}, "selected": {}, "flags": []}

    @contextlib.contextmanager
    def phase(self, name: str):
        from . import costs

        with costs.phase(self.device) as c:
            yield
        self.report["phases"][name] = c
        peak = "" if c["peak_mem_mib"] is None else (f"  peak {c['peak_mem_mib']:.0f} MiB"
                                                     f" (reserved {c['peak_mem_reserved_mib']:.0f})")
        print(f"[pilot] {name}: {c['wall_s']:.1f} s{peak}", flush=True)

    def measure(self, name: str, **values):
        self.report["measurements"][name] = values
        print(f"[pilot] {name}: " + "  ".join(f"{k}={_fmt(v)}" for k, v in values.items()), flush=True)

    def flag(self, text: str):
        self.report["flags"].append(text)
        print(f"[pilot] FLAG: {text}", flush=True)


def _fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else v


def _loss(model, ds, cfg, device):
    from .evaluate import mean_loss

    return mean_loss(model, ds, cfg, device)


def _finetune_grid(log: _Log, cfg, lrs, epochs: int, tol: float, tag: str):
    """Rule 1 for one model; returns (rate, model at that rate, splits, snapshots)."""
    from . import central, data, tofu_metrics, unlearn
    from .seed import set_seed

    dev = log.device
    set_seed(cfg.seed)
    model, tok = unlearn.build_model_and_tokenizer(cfg)
    model.to(dev)
    splits = data.load_splits(cfg, tok)
    world = tofu_metrics.load_eval_sets(_with(cfg, data={"tofu_metrics": True}), tok, splits)["world_facts"].answer
    base_state = central._snapshot(model)
    base_wf = _loss(model, world, cfg, dev)
    base_retain = _loss(model, splits.retain, cfg, dev)
    results, states = {}, {}
    for lr in lrs:
        central._restore(model, base_state)
        set_seed(cfg.seed)
        c = _with(cfg, unlearn={"finetune_lr": lr, "finetune_epochs": epochs})
        steps = unlearn.StepStats()
        with log.phase(f"{tag}_finetune_lr{lr:g}"):
            unlearn.finetune_base(c, splits, dev, model=model, stats=steps)
        results[lr] = {"retain_loss": _loss(model, splits.retain, cfg, dev), "world_facts_loss": _loss(model, world, cfg, dev)}
        states[lr] = central._snapshot(model)
        log.measure(f"{tag}_lr{lr:g}", base_world_facts_loss=base_wf, base_retain_loss=base_retain, **results[lr],
                    steps=steps.summary())
    ok = [lr for lr in lrs if results[lr]["world_facts_loss"] <= (1 + tol) * base_wf]
    if ok:
        chosen = min(ok, key=lambda lr: results[lr]["retain_loss"])
    else:
        chosen = min(lrs, key=lambda lr: results[lr]["world_facts_loss"])
        log.flag(f"{tag}: no rate keeps the world-facts loss within {tol:.0%} of the base model; "
                 f"kept {chosen:g} (smallest world-facts loss)")
    central._restore(model, states[chosen])
    return chosen, model, splits, {"base": base_state, **states}, results[chosen]


def _free(model) -> None:
    import gc

    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _lora_weight_load(log: _Log, lcfg, p: dict) -> None:
    """Step 0: load the LoRA model's weights and score a few forget records; stop on failure."""
    from . import data, unlearn
    from .seed import set_seed

    with log.phase("lora_weight_load"):
        set_seed(lcfg.seed)
        try:
            model, tok = unlearn.build_model_and_tokenizer(lcfg)
        except RuntimeError as e:
            raise SystemExit(f"[pilot] stopping: {e}") from e
        model.to(log.device)
        few = data.load_splits(lcfg, tok).forget.head(LOSS_RECORDS)
        loss = _loss(model, few, lcfg, log.device)
    if not math.isfinite(loss):
        raise SystemExit(f"[pilot] stopping: {p['lora_model']} loaded but its loss on {LOSS_RECORDS} records is {loss}.")
    log.measure("lora_weight_load", model=p["lora_model"], forget_loss_base=loss)
    _free(model)


def _lora_rates(log: _Log, lcfg, p: dict, full_lr: float) -> tuple[float, float]:
    """Steps 4 and 5: rule 1 over `lora_lrs`, the unlearning rate
    `base_unlearn_lr` x LoRA rate / `full_lr`, and the GA sanity check."""
    from . import unlearn
    from .seed import set_seed

    dev = log.device
    lora_lr, model, lsplits, _, _ = _finetune_grid(log, lcfg, p["lora_lrs"], p["epochs"], p["world_facts_tolerance"],
                                                   "lora")
    unlearn_lr = p["base_unlearn_lr"] * lora_lr / full_lr
    log.report["selected"]["lora_finetune_lr"] = lora_lr
    log.report["selected"]["lora_learning_rate"] = unlearn_lr
    min_rise = p.get("ga_min_rise", GA_MIN_RISE)
    before = _loss(model, lsplits.forget, lcfg, dev)
    set_seed(lcfg.seed)
    steps = unlearn.StepStats()
    with log.phase("lora_ga_sanity"):
        unlearn.apply_unlearning(model, lsplits, _with(lcfg, unlearn={"method": "gradient_ascent",
                                                                      "learning_rate": unlearn_lr}), dev, steps)
    after = _loss(model, lsplits.forget, lcfg, dev)
    log.measure("lora_ga_sanity", learning_rate=unlearn_lr, full_finetune_lr=full_lr, forget_loss_before=before,
                forget_loss_after=after, rise=after - before, min_rise=min_rise, rises=after - before >= min_rise,
                steps=steps.summary())
    if not after - before >= min_rise:
        log.flag(f"LoRA GA at {unlearn_lr:g} raises the forget loss by {after - before:.4f} nats "
                 f"({before:.4f} -> {after:.4f}), less than {min_rise:g}")
    _free(model)
    return lora_lr, unlearn_lr


def _time_unit(log: _Log, path, unit, device, out_dir: Path, out_csv: Path) -> None:
    """Run one campaign unit alone in this process; its rows (every phase time) go to `out_csv`."""
    from . import campaign

    start = time.perf_counter()  # not a cost phase: run_unit times its own phases
    rows = campaign.run_unit(path, unit, "pilot", device, out_dir)
    log.report["phases"][f"timing_{unit.variant}"] = {"wall_s": time.perf_counter() - start}
    print(f"[pilot] timing_{unit.variant}: {time.perf_counter() - start:.1f} s", flush=True)
    campaign.write_rows(out_csv, rows, replace_keys=[unit.key])
    r = rows[0]
    times = {k[len("time_s_"):]: v for k, v in r.items() if k.startswith("time_s_") and v is not None}
    print(f"[pilot] timing_{unit.variant} phases (s, first row): "
          + "  ".join(f"{k}={v:.1f}" for k, v in times.items()), flush=True)
    for row in rows:
        if row["attack"] == "loss_threshold" and row["nonmember"] == "primary":
            print(f"[pilot]   {row['method']}: " + "  ".join(
                f"{k[len('time_s_'):]}={row[k]:.1f}" for k in row
                if k.startswith("time_s_") and row[k] is not None and k[len("time_s_"):] not in
                ("base_scoring", "finetune", "fedavg", "pre_eval")), flush=True)


def lora_timing_unit(p: dict, lora_lr: float, unlearn_lr: float):
    """The central LoRA unit the LoRA-only pilot times: the pilot config with
    the LoRA model, its LoRA settings and the selected rates (the same
    settings as lora.yaml's unit; a test checks this)."""
    from . import campaign

    return campaign.Unit(p["lora_model"], p["seed"], "lora", "central",
                         {"model": {"lora": p.get("lora", {})},
                          "unlearn": {"finetune_lr": lora_lr, "learning_rate": unlearn_lr}})


def run_pilot(path, device: str | None = None, out_dir="results/revision", out_csv=None,
              only: str | None = None) -> dict:
    """Run the pilot described in the module docstring (all of it, or only
    the LoRA part); returns its report."""
    from . import campaign, central, federated, unlearn
    from .config import load_config
    from .seed import set_seed

    if only is not None and only not in ONLY:
        raise ValueError(f"only must be one of {ONLY}, not {only!r}")
    raw = yaml.safe_load(Path(path).read_text())
    p = raw["pilot"]
    cfg = load_config(path)
    cfg.model = replace(cfg.model, name=p["model"])
    cfg.seed = p["seed"]
    if device is not None:
        cfg.device = device
    dev = unlearn.resolve_device(cfg)
    log = _Log(dev)
    lcfg = _with(cfg, model={"name": p["lora_model"], "lora": p.get("lora", {})})
    out_dir = Path(out_dir)

    # 0. the first load of the LoRA model's weights: fail early
    _lora_weight_load(log, lcfg, p)

    if only == "lora":
        log.report["only"] = "lora"
        log.report["lora_model"] = p["lora_model"]
        full_lr = p["full_finetune_lr"]
        lora_lr, unlearn_lr = _lora_rates(log, lcfg, p, full_lr)
        out_csv = Path(out_csv or out_dir / "pilot_lora.csv")
        _time_unit(log, path, lora_timing_unit(p, lora_lr, unlearn_lr), device, out_dir, out_csv)
        report_path = out_dir / "pilot_lora_calibration.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(log.report, indent=2, default=str))
        sel = log.report["selected"]
        print("[pilot] selected values (paste into configs/revision/lora.yaml, lines marked '# set from pilot', "
              "or use --apply-calibration):")
        print(f"  lora.yaml:  unlearn.finetune_lr: {sel['lora_finetune_lr']:g}   unlearn.learning_rate: "
              f"{sel['lora_learning_rate']:g}   (full rate {full_lr:g})")
        print(f"[pilot] flags: {log.report['flags'] or 'none'}")
        print(f"[pilot] report -> {report_path}; timing rows -> {out_csv}")
        return log.report

    # 1. rule 1, full fine-tuning
    epochs = p["epochs"]
    lr, model, splits, states, central_losses = _finetune_grid(
        log, cfg, p["finetune_lrs"], epochs, p["world_facts_tolerance"], "full")
    log.report["selected"]["finetune_lr"] = lr

    # 2. rule 2, rounds
    fcfg = _with(cfg, unlearn={"finetune_lr": lr})
    clients = federated.shard_clients(splits, fcfg)
    central._restore(model, states["base"])
    set_seed(cfg.seed)
    norms, pre_states, ok_rounds = [], {}, []
    for r in range(1, max(p["rounds"]) + 1):
        with log.phase(f"fedavg_round{r}"):
            norms += federated.fedavg_rounds(model, clients, fcfg, dev, 1)
        if r in p["rounds"]:
            retain = _loss(model, splits.retain, fcfg, dev)
            within = abs(retain - central_losses["retain_loss"]) <= p["retain_tolerance"] * central_losses["retain_loss"]
            log.measure(f"rounds{r}", retain_loss=retain, central_retain_loss=central_losses["retain_loss"],
                        update_norm=norms[-1], within=within)
            pre_states[r] = central._snapshot(model)
            if within:
                ok_rounds.append(r)
    rounds = min(ok_rounds) if ok_rounds else max(p["rounds"])
    if not ok_rounds:
        log.flag(f"rounds: no R in {p['rounds']} brings the retain loss within "
                 f"{p['retain_tolerance']:.0%} of the central model; kept {rounds}")
    log.report["selected"]["rounds"] = rounds

    # 3. rule 3, radius
    rcfg = _with(fcfg, federated={"rounds": rounds})
    central._restore(model, states["base"])
    set_seed(cfg.seed)
    with log.phase("fed_retrain"):
        federated.fedavg_rounds(model, clients[1:], rcfg, dev, rounds)
    retrain_forget = _loss(model, splits.forget, rcfg, dev)
    ok_factors = []
    for f in p["rho_factors"]:
        central._restore(model, pre_states[rounds])
        set_seed(cfg.seed)
        rho = f * norms[rounds - 1]
        with log.phase(f"client_ga_rho{f:g}"):
            federated.client_gradient_ascent(model, clients[0], rcfg, dev, pre_states[rounds], rho)
            federated.fedavg_rounds(model, clients[1:], rcfg, dev, rcfg.federated.recovery_rounds)
        forget = _loss(model, splits.forget, rcfg, dev)
        log.measure(f"rho_factor{f:g}", rho=rho, forget_loss_post_recovery=forget,
                    forget_loss_fed_retrain=retrain_forget, exceeds=forget > retrain_forget)
        if forget > retrain_forget:
            ok_factors.append(f)
    factor = min(ok_factors) if ok_factors else max(p["rho_factors"])
    if not ok_factors:
        log.flag(f"rho_factor: no factor in {p['rho_factors']} leaves the forget loss above the federated "
                 f"retrain's after recovery; kept {factor:g}")
    log.report["selected"]["rho_factor"] = factor
    _free(model)

    # 4. rule 1 for LoRA, 5. GA sanity check
    log.report["lora_model"] = p["lora_model"]
    _lora_rates(log, lcfg, p, lr)

    # 6. timing runs, alone in this process
    out_csv = Path(out_csv or out_dir / "pilot.csv")
    sel = log.report["selected"]
    overrides = {"unlearn": {"finetune_lr": lr}, "federated": {"rounds": rounds, "rho_factor": factor}}
    for unit in (campaign.Unit(p["model"], p["seed"], "central", "central", overrides),
                 campaign.Unit(p["model"], p["seed"], "iid_K5", "federated",
                               {**overrides, "federated": {**overrides["federated"], "num_clients": 5,
                                                           "sharding": "iid"}})):
        _time_unit(log, path, unit, device, out_dir, out_csv)

    report_path = out_dir / "pilot_calibration.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(log.report, indent=2, default=str))
    print("[pilot] selected values (paste into configs/revision/*.yaml, lines marked '# set from pilot'):")
    print(f"  central.yaml, fed_iid.yaml, fed_ext.yaml:  unlearn.finetune_lr: {sel['finetune_lr']:g}")
    print(f"  fed_iid.yaml, fed_ext.yaml:  federated.rounds: {sel['rounds']}   federated.rho_factor: "
          f"{sel['rho_factor']:g}")
    print(f"  lora.yaml:  unlearn.finetune_lr: {sel['lora_finetune_lr']:g}   unlearn.learning_rate: "
          f"{sel['lora_learning_rate']:g}")
    print(f"[pilot] flags: {log.report['flags'] or 'none'}")
    print(f"[pilot] report -> {report_path}; timing rows -> {out_csv}")
    return log.report
