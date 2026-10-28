"""Memory probe: one step of each training phase of a campaign unit, before the campaign.

For one model of a campaign config (resolved as its first unit, so variant
overrides such as `grad_accum_steps` apply), on the longest records:
  finetune   one `train_lm` step at `train_batch_size` (with AdamW states);
  npo        one NPO step (`batch_size` forget + retain, frozen reference copy);
  fedavg     one FedAvg round of two one-batch clients (the second trains
             while the global state and the weighted sum are on the device,
             as configured);
  client_ga  one projected client-GA step (θ_pre kept on the device);
  eval       one scoring pass at `eval_batch_size`.
Under LoRA the federated phases are skipped. The phases run one after the other in one process, as in a campaign unit,
without emptying the cache in between, so the reserved peaks carry over.
Each phase is measured with `costs.phase`; the probe fails if the largest
reserved peak exceeds the limit.
"""

from __future__ import annotations

from dataclasses import replace

LIMIT_GIB = 14.0
PLACEHOLDER_LR = 1e-5  # for rates still null (uncalibrated config)


def _longest(dataset, n: int):
    from torch.utils.data import Subset

    order = sorted(range(len(dataset)), key=lambda i: -len(dataset[i]["input_ids"]))
    return Subset(dataset, order[:n])


def run_probe(path, model: str, device: str | None = None, limit_gib: float = LIMIT_GIB) -> dict:
    """Run the probe; returns {phase: costs dict} and raises SystemExit over the limit."""
    from . import attacks, campaign, central, costs, data, federated, unlearn
    from .seed import set_seed

    raw = campaign.read_campaign(path)
    unit = campaign.units(raw, models=[model])[0]
    cfg = campaign.resolve(path, unit, device)
    dev = unlearn.resolve_device(cfg)
    # before the pilot sets them: memory does not depend on the rates
    u = replace(cfg.unlearn, finetune_lr=cfg.unlearn.finetune_lr or PLACEHOLDER_LR,
                learning_rate=cfg.unlearn.learning_rate or PLACEHOLDER_LR)
    cfg = replace(cfg, unlearn=u)
    print(f"[memprobe] {model} on {dev}: train_batch_size {u.train_batch_size}, grad_accum_steps "
          f"{u.grad_accum_steps}, batch_size {u.batch_size}, eval_batch_size {u.eval_batch_size}, "
          f"precision {cfg.model.precision}", flush=True)
    set_seed(cfg.seed)
    net, tok = unlearn.build_model_and_tokenizer(cfg)
    net.to(dev)
    splits = data.load_splits(cfg, tok)
    train_bs = u.train_batch_size or u.batch_size
    members = _longest(splits.retain, train_bs)
    small = replace(splits, forget=_longest(splits.forget, u.batch_size), retain=_longest(splits.retain, u.batch_size))
    lr = u.finetune_lr or u.learning_rate
    out = {}

    with costs.phase(dev) as out["finetune"]:
        unlearn.train_lm(net, members, cfg, dev, 1, lr)
    with costs.phase(dev) as out["npo"]:
        unlearn.apply_unlearning(net, small, replace(cfg, unlearn=replace(u, method="npo", epochs=1)), dev)
    if unlearn.is_peft(net):
        print("[memprobe] LoRA model: no federated phases (the federated path trains full models)", flush=True)
    else:
        with costs.phase(dev) as out["fedavg"]:
            federated.fedavg_rounds(net, [members, members], cfg, dev, 1)
        center = central._snapshot(net)
        with costs.phase(dev) as out["client_ga"]:
            fcfg = replace(cfg, federated=replace(cfg.federated, client_unlearn_epochs=1))
            federated.client_gradient_ascent(net, small.forget, fcfg, dev, center, rho=1.0)
    with costs.phase(dev) as out["eval"]:
        attacks.score_all(net, _longest(splits.retain, u.eval_batch_size), cfg, dev)

    worst = 0.0
    for name, c in out.items():
        if c["peak_mem_mib"] is None:
            print(f"[memprobe] {name}: {c['wall_s']:.1f} s (no CUDA: no memory peaks)", flush=True)
            continue
        worst = max(worst, c["peak_mem_reserved_mib"])
        print(f"[memprobe] {name}: {c['wall_s']:.1f} s  peak allocated {c['peak_mem_mib']:.0f} MiB, "
              f"reserved {c['peak_mem_reserved_mib']:.0f} MiB", flush=True)
    if worst > limit_gib * 1024:
        raise SystemExit(f"[memprobe] {model}: peak reserved {worst:.0f} MiB exceeds {limit_gib:g} GiB; "
                         f"raise unlearn.grad_accum_steps for this model")
    print(f"[memprobe] {model}: largest reserved peak {worst:.0f} MiB (limit {limit_gib:g} GiB)", flush=True)
    return out
