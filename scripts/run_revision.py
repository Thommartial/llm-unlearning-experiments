"""Run a revision campaign (resumable), check its result files, or merge them.

    python scripts/run_revision.py --campaign central --seeds 0 2 4 --out results/revision/central_gpu0.csv
    python scripts/run_revision.py --campaign central --check results/revision/central_gpu*.csv
    python scripts/run_revision.py --merge results/revision/central_gpu*.csv --out results/revision/central.csv
    python scripts/run_revision.py --campaign pilot
    python scripts/run_revision.py --apply-calibration results/revision/pilot_calibration.json
    python scripts/run_revision.py --campaign central --memory-probe gpt2-medium

Campaigns: pilot, central, fed_iid, fed_ext, lora (Kaggle) and smoke, micro
(CPU). The config defaults to configs/revision/<campaign>.yaml
(smoke_cpu.yaml, micro_tofu_cpu.yaml). A run skips the (model, seed,
variant) units already complete in --out. --check judges success by the
result files: it exits 1 and lists the missing units if any unit of the
selected models and seeds lacks rows. --apply-calibration fills the values
marked `# set from pilot` in configs/revision/*.yaml from the pilot's report
and prints the diff and the pilot's flags (review both, then commit).
--memory-probe runs one step of each training phase of one model of the
campaign (`memprobe`) and exits 1 if the reserved GPU memory peak exceeds
--max-reserved-gib.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unlearning import campaign  # noqa: E402

CAMPAIGNS = ("pilot", "central", "fed_iid", "fed_ext", "lora", "smoke", "micro")
CONFIGS = {"smoke": "smoke_cpu.yaml", "micro": "micro_tofu_cpu.yaml"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", choices=CAMPAIGNS)
    ap.add_argument("--config", help="campaign config (default: configs/revision/<campaign>.yaml)")
    ap.add_argument("--models", nargs="+", help="restrict to these models")
    ap.add_argument("--seeds", nargs="+", type=int, help="restrict to these seeds")
    ap.add_argument("--out", help="CSV to write (default: results/revision/<campaign>.csv)")
    ap.add_argument("--out-dir", default="results/revision", help="scores, resolved configs and pilot report")
    ap.add_argument("--device", help="override the config's device (cpu, cuda)")
    ap.add_argument("--check", nargs="+", metavar="CSV", help="check these result files instead of running")
    ap.add_argument("--merge", nargs="+", metavar="CSV", help="merge these CSVs into --out")
    ap.add_argument("--list", action="store_true", help="list the units and exit")
    ap.add_argument("--apply-calibration", metavar="JSON", help="fill the '# set from pilot' config values")
    ap.add_argument("--configs-dir", default="configs/revision", help="configs for --apply-calibration")
    ap.add_argument("--memory-probe", metavar="MODEL", help="one step of each training phase of MODEL, then exit")
    ap.add_argument("--max-reserved-gib", type=float, default=14.0, help="memory probe limit (GiB reserved)")
    args = ap.parse_args(argv)

    if args.apply_calibration:
        import json

        diff = campaign.apply_calibration(args.apply_calibration, args.configs_dir)
        print(diff or "no change: the configs already hold these values")
        flags = json.loads(Path(args.apply_calibration).read_text()).get("flags", [])
        print("pilot flags:" + ("".join(f"\n  - {f}" for f in flags) if flags else " none"))
        return 0
    if args.merge:
        if not args.out:
            ap.error("--merge needs --out")
        n = campaign.merge(args.merge, args.out)
        print(f"merged {n} rows from {len(args.merge)} files -> {args.out}")
        return 0
    if not args.campaign:
        ap.error("--campaign is required")
    config = args.config or f"configs/revision/{CONFIGS.get(args.campaign, args.campaign + '.yaml')}"
    out = args.out or f"{args.out_dir}/{args.campaign}.csv"

    if args.memory_probe:
        from unlearning.memprobe import run_probe

        try:
            run_probe(config, args.memory_probe, args.device, args.max_reserved_gib)
        except SystemExit as e:
            print(e)
            return 1
        return 0
    if args.campaign == "pilot":
        from unlearning.pilot import run_pilot

        run_pilot(config, args.device, args.out_dir, out)
        return 0
    raw = campaign.read_campaign(config)
    unit_list = campaign.units(raw, args.models, args.seeds)
    if args.list:
        for u in unit_list:
            print(f"{u.slug}  ({u.kind})")
        return 0
    if args.check:
        gone = campaign.missing(args.check, unit_list, args.campaign)
        for u in gone:
            print(f"missing: {u.slug}")
        print(f"{len(unit_list) - len(gone)}/{len(unit_list)} units complete in {args.check}")
        return 1 if gone else 0
    campaign.run_campaign(config, args.campaign, out, args.models, args.seeds, args.device, args.out_dir)
    gone = campaign.missing([out], unit_list, args.campaign)
    print(f"{len(unit_list) - len(gone)}/{len(unit_list)} units complete in {out}")
    return 1 if gone else 0


if __name__ == "__main__":
    sys.exit(main())
