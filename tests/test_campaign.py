"""Tests for the revision campaign runner, the configs and the pilot (CPU, tiny models)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from unlearning import campaign, data, tofu_metrics, unlearn
from unlearning.config import load_config

REVISION = Path("configs/revision")


def _script():
    spec = importlib.util.spec_from_file_location("run_revision", "scripts/run_revision.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name, n_units, models, requires", [
    ("central", 15, {"distilgpt2", "gpt2", "gpt2-medium"}, {"unlearn.finetune_lr"}),
    ("fed_iid", 15, {"distilgpt2", "gpt2", "gpt2-medium"},
     {"unlearn.finetune_lr", "federated.rounds", "federated.rho_factor"}),
    ("fed_ext", 9, {"gpt2"}, {"unlearn.finetune_lr", "federated.rounds", "federated.rho_factor"}),
    ("lora", 5, {"TinyLlama/TinyLlama_v1.1"}, {"unlearn.finetune_lr", "unlearn.learning_rate"}),
])
def test_campaign_configs(name, n_units, models, requires):
    path = REVISION / f"{name}.yaml"
    raw = campaign.read_campaign(path)
    us = campaign.units(raw)
    assert len(us) == n_units and {u.model for u in us} == models
    assert set(raw["campaign"]["requires"]) == requires
    with pytest.raises(SystemExit, match="set from pilot"):  # placeholders are still null
        campaign.check_calibrated(raw)
    text = path.read_text()
    assert all(f"{k.split('.')[1]}: null" in text for k in requires) and "# set from pilot" in text
    cfg = campaign.resolve(path, us[0])
    assert cfg.data.dataset == "tofu_official" and cfg.unlearn.finetune_epochs == 3
    assert cfg.model.precision == "fp16_amp" and cfg.unlearn.train_batch_size == 16


def test_fed_ext_variants_and_lora_settings():
    path = REVISION / "fed_ext.yaml"
    us = campaign.units(campaign.read_campaign(path))
    fed = {u.variant: campaign.resolve(path, u).federated for u in us if u.seed == 0}
    assert {v: (f.sharding, f.num_clients) for v, f in fed.items()} == {
        "by_author_K5": ("by_author", 5), "iid_K3": ("iid", 3), "iid_K10": ("iid", 10)}
    assert {u.seed for u in us} == {0, 1, 2}
    lora = campaign.resolve(REVISION / "lora.yaml", campaign.units(campaign.read_campaign(REVISION / "lora.yaml"))[0])
    assert unlearn.lora_settings(lora) == unlearn.LORA_DEFAULTS
    pilot = yaml.safe_load((REVISION / "pilot.yaml").read_text())["pilot"]
    assert pilot["finetune_lrs"] == [2e-5, 5e-5] and pilot["rounds"] == [3, 5, 8]
    assert pilot["rho_factors"] == [0.5, 1.0, 2.0] and pilot["lora_lrs"] == [1e-4, 2e-4]
    assert (pilot["model"], pilot["seed"]) == ("gpt2", 0)


def test_units_resolve_and_hash():
    path = REVISION / "smoke_cpu.yaml"
    raw = campaign.read_campaign(path)
    us = campaign.units(raw)
    assert [u.slug for u in us] == ["tiny_seed0_central", "tiny_seed0_iid_K3", "tiny_seed0_by_author_K3",
                                    "tiny_llama_seed0_lora"]
    cfg = campaign.resolve(path, us[2], device="cuda")
    assert (cfg.federated.sharding, cfg.model.name, cfg.seed, cfg.device) == ("by_author", "tiny", 0, "cuda")
    lora = campaign.resolve(path, us[3])
    assert lora.model.name == "tiny_llama" and lora.model.lora == {}
    other_seed = campaign.Unit("tiny", 7, "central", "central")
    assert campaign.config_hash(campaign.resolve(path, us[0])) == campaign.config_hash(campaign.resolve(path, other_seed))
    assert campaign.config_hash(campaign.resolve(path, us[0])) != campaign.config_hash(cfg)
    assert campaign.units(raw, models=["tiny_llama"]) == [us[3]] and campaign.units(raw, seeds=[1]) == []


def test_write_rows_and_merge(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    campaign.write_rows(a, [{"model": "m", "seed": 0, "variant": "v", "method": "x", "auc": 0.5}])
    campaign.write_rows(a, [{"model": "m", "seed": 1, "variant": "v", "method": "x", "extra": 1}])
    rows = campaign.read_rows(a)
    assert [r["seed"] for r in rows] == ["0", "1"] and rows[0]["extra"] == "" and rows[1]["auc"] == ""
    campaign.write_rows(a, [{"model": "m", "seed": 0, "variant": "v", "method": "y"}], replace_keys=[("m", 0, "v")])
    assert [(r["seed"], r["method"]) for r in campaign.read_rows(a)] == [("1", "x"), ("0", "y")]
    campaign.write_rows(b, [{"model": "n", "seed": 0, "variant": "v", "method": "z"}])
    out = tmp_path / "all.csv"
    assert campaign.merge([a, b], out) == 3
    assert campaign.merge([a, b], out) == 3 and len(campaign.read_rows(out)) == 3  # overwrites, no duplicates
    assert not list(tmp_path.glob("*.tmp"))


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rev")
    cfg = yaml.safe_load((REVISION / "smoke_cpu.yaml").read_text())
    cfg["campaign"]["variants"] = cfg["campaign"]["variants"][:2]  # central and iid_K3
    path = tmp / "smoke.yaml"
    path.write_text(yaml.safe_dump(cfg))
    out = tmp / "smoke.csv"
    ran = campaign.run_campaign(path, "smoke", out, out_dir=tmp)
    return path, out, tmp, ran


def test_run_campaign_rows_and_files(smoke):
    path, out, tmp, ran = smoke
    assert [u.slug for u in ran] == ["tiny_seed0_central", "tiny_seed0_iid_K3"]
    rows = campaign.read_rows(out)
    assert len(rows) == 4 * 3 * 2 + 3 * 3 * 2
    for r in rows:
        assert r["campaign"] == "smoke" and len(r["config_hash"]) == 12 and len(r["commit"]) == 40
        assert r["gpu_name"] == "cpu" and r["torch_version"] and r["device"] == "cpu"
    assert {r["variant"] for r in rows} == {"central", "iid_K3"}
    for u in ran:
        assert (tmp / "scores" / "smoke" / f"{u.slug}.npz").exists()
        assert json.loads((tmp / "configs" / "smoke" / f"{u.slug}.json").read_text())["seed"] == 0
    assert campaign.missing([out], campaign.units(campaign.read_campaign(path)), "smoke") == []


def test_run_campaign_resumes_and_reruns_incomplete_units(smoke, monkeypatch, tmp_path):
    path, out, _, _ = smoke
    monkeypatch.setattr(campaign, "run_unit", lambda *a, **k: pytest.fail("a done unit ran again"))
    assert campaign.run_campaign(path, "smoke", out, out_dir=tmp_path) == []
    # drop one method of the central unit: the unit is incomplete and runs again, replacing its rows
    partial = tmp_path / "partial.csv"
    keep = [r for r in campaign.read_rows(out) if not (r["variant"] == "central" and r["method"] == "npo")]
    campaign.write_rows(partial, keep)
    us = campaign.units(campaign.read_campaign(path))
    assert [u.variant for u in campaign.missing([partial], us, "smoke")] == ["central"]
    fake = [{**r, "method": m} for m in ("retrain", "gradient_ascent", "gradient_difference", "npo")
            for r in keep[:1]]
    monkeypatch.setattr(campaign, "run_unit", lambda *a, **k: fake)
    assert [u.variant for u in campaign.run_campaign(path, "smoke", partial, out_dir=tmp_path)] == ["central"]
    rows = campaign.read_rows(partial)
    assert sum(r["variant"] == "central" for r in rows) == 4 and campaign.missing([partial], us, "smoke") == []


def test_script_check_and_merge(smoke, tmp_path):
    path, out, _, _ = smoke
    script = _script()
    assert script.main(["--campaign", "smoke", "--config", str(path), "--check", str(out)]) == 0
    empty = tmp_path / "none.csv"
    assert script.main(["--campaign", "smoke", "--config", str(path), "--check", str(empty)]) == 1
    merged = tmp_path / "merged.csv"
    assert script.main(["--merge", str(out), "--out", str(merged)]) == 0
    assert len(campaign.read_rows(merged)) == len(campaign.read_rows(out))
    with pytest.raises(SystemExit, match="set from pilot"):
        script.main(["--campaign", "central", "--out", str(tmp_path / "c.csv")])


def test_pilot_on_smoke_config(tmp_path):
    from unlearning.pilot import run_pilot

    report = run_pilot(REVISION / "smoke_cpu.yaml", "cpu", tmp_path)
    sel = report["selected"]
    assert set(sel) == {"finetune_lr", "rounds", "rho_factor", "lora_finetune_lr", "lora_learning_rate"}
    assert sel["finetune_lr"] in (5e-4, 1e-3) and sel["rounds"] in (1, 2) and sel["rho_factor"] in (0.5, 1.0)
    assert sel["lora_learning_rate"] == pytest.approx(5e-4 * sel["lora_finetune_lr"] / sel["finetune_lr"])
    assert "lora_ga_sanity" in report["measurements"] and "lora_weight_load" in report["phases"]
    assert json.loads((tmp_path / "pilot_calibration.json").read_text())["selected"] == sel
    rows = campaign.read_rows(tmp_path / "pilot.csv")
    assert {r["variant"] for r in rows} == {"central", "iid_K5"} and all(r["campaign"] == "pilot" for r in rows)
    fed = [r for r in rows if r["variant"] == "iid_K5"]
    assert {r["num_clients"] for r in fed} == {"5"} and {float(r["rho_factor"]) for r in fed} == {sel["rho_factor"]}


def test_pilot_stops_early_when_lora_weights_do_not_load(monkeypatch, tmp_path):
    from unlearning import pilot

    build = unlearn.build_model_and_tokenizer

    def failing(cfg):
        if cfg.model.lora is not None:
            raise RuntimeError(f"Could not load the weights of {cfg.model.name}")
        pytest.fail("the pilot went on after a failed weight load")
        return build(cfg)
    monkeypatch.setattr(unlearn, "build_model_and_tokenizer", failing)
    with pytest.raises(SystemExit, match="Could not load the weights of tiny_llama"):
        pilot.run_pilot(REVISION / "smoke_cpu.yaml", "cpu", tmp_path)


def test_weight_load_error_is_clear(monkeypatch):
    import transformers

    cfg = load_config("configs/smoke_synthetic.yaml")
    cfg.model.name = "TinyLlama/TinyLlama_v1.1"
    cfg.model.lora = {}

    class FakeTokenizer:
        pad_token, eos_token = None, "</s>"
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer())

    def no_weights(*a, **k):
        raise OSError("pytorch_model.bin not found")
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", no_weights)
    with pytest.raises(RuntimeError, match="Could not load the weights of TinyLlama/TinyLlama_v1.1.*OSError"):
        unlearn.build_model_and_tokenizer(cfg)


def test_max_samples_caps_official_splits_and_eval_sets():
    cfg = load_config("configs/smoke_synthetic.yaml")
    cfg.data.dataset = "synthetic_official"
    full = data.load_splits(cfg)
    cfg.data.max_samples = 10
    s = data.load_splits(cfg)
    assert (len(s.forget), len(s.retain)) == (10, 10) and all(len(v) == 10 for v in s.nonmembers.values())
    assert s.holdout is s.nonmembers["primary"] and len(s.forget_is_outlier) == 10
    assert s.retain.row_ids == full.retain.row_ids[:10]
    sets = tofu_metrics.load_eval_sets(cfg, None, s)
    assert all(len(es.answer) <= 10 and all(len(p) == len(es.answer) for p in es.perturbed) for es in sets.values())


def test_notebooks_are_generated_and_parse():
    import ast

    spec = importlib.util.spec_from_file_location("make_revision_notebooks", "scripts/make_revision_notebooks.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    for name, nb_spec in gen.NOTEBOOKS.items():
        folder = Path("notebooks/revision") / name
        nb = json.loads((folder / f"{name}.ipynb").read_text())
        assert nb == gen.notebook(name, nb_spec), f"{name}: rerun scripts/make_revision_notebooks.py"
        meta = json.loads((folder / "kernel-metadata.json").read_text())
        assert meta == gen.metadata(name) and meta["id"].startswith("${KAGGLE_USERNAME}/")
        assert (meta["machine_shape"], meta["enable_internet"], meta["is_private"]) == ("NvidiaTeslaT4", "true", "true")
        code = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
        for cell in code:
            ast.parse(cell)
        assert 'REPO_COMMIT = "SET_BY_KAGGLE_RUN"' in code[0] and f'CAMPAIGN = "{nb_spec["campaign"]}"' in code[0]
        assert any('"REQUIRE_CUDA_TESTS": "1"' in c for c in code)
        seeds = [s for v in nb_spec["gpu_seeds"].values() if v for s in v]
        if nb_spec["campaign"] != "pilot":
            units = campaign.units(campaign.read_campaign(REVISION / f"{nb_spec['campaign']}.yaml"))
            assert sorted(seeds) == sorted({u.seed for u in units})  # every seed on exactly one GPU
