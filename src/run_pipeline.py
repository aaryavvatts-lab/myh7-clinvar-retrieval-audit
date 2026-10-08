"""v7 one-command pipeline:  bash run_all.sh   (or: python -m src.run_pipeline)
Every step runs in its own interpreter (so gene lists written by one step are seen by the next) and is timed.
Network stages reuse the archived raw files in data/raw, data/external, data/interim unless FORCE_DOWNLOAD=1.
All data inputs are inside v7/ (the truncated v5 table is the frozen snapshot in data/legacy_v5, SHA256SUMS).
The timed run ends with the paper figures (figures_paper_v7).
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

LEGACY = [  # v6 analysis chain (MYH7, nucleotide-level cohorts, 4 label/AF specifications) + the two paper-extras steps
    ("data_acquisition", None), ("build_labels", ["--af", "v4", "--rule", "aggregate"]), ("build_labels", ["--af", "v4", "--rule", "v5rcv"]),
    ("build_labels", ["--af", "v2", "--rule", "aggregate"]), ("build_labels", ["--af", "v2", "--rule", "v5rcv"]), ("build_features", None),
    ("diagnose_gap", None), ("coverage_v6", None), ("model_v6", None), ("perm_v6", None), ("vus_v6", None),
    ("spatial_block_cv", None), ("validate_on_v5", None), ("figures_v6", None), ("numbers", None),
    ("paper_extras", None), ("figures_paper", None)]
V7 = [("mg_clinvar", None), ("mg_acquire", None), ("mg_pool", None), ("mg_acquire", None), ("mg_labels", None),
      ("a5_retrieval_audit", None), ("a2_erepo_myh7", None), ("a1_myh7", None), ("a3_inference_myh7", None),
      ("a4_multigene", None), ("figures_v7", None), ("mk_data_availability", None), ("verdicts", None),
      ("figures_paper_v7", None)]


def run(name, args):
    t = time.time()
    if name == "data_acquisition" and Path("data/processed/myh7_all_variants_annotated.parquet").exists() and not os.environ.get("FORCE_DOWNLOAD"):
        print(f"== {name}: archived v6 acquisition output reused (FORCE_DOWNLOAD=1 refetches live APIs)", flush=True)
        return 0.0
    print(f"== {name} {args or ''}", flush=True)
    r = subprocess.run([sys.executable, "-m", f"src.{name}"] + (args or []), env={**os.environ, "PYTHONWARNINGS": "ignore"})
    if r.returncode != 0:
        raise SystemExit(f"step {name} failed with code {r.returncode}")
    dt = time.time() - t
    print(f"   done in {dt:.0f}s", flush=True)
    return dt


def main():
    timings = {}
    t0 = time.time()
    for name, args in LEGACY + V7:
        timings[f"{name} {' '.join(args) if args else ''}".strip()] = round(run(name, args), 1)
    timings["TOTAL_seconds"] = round(time.time() - t0, 1)
    Path("logs").mkdir(exist_ok=True)
    Path("logs/run_all_timings.json").write_text(json.dumps(timings, indent=1))
    run("numbers_v7", None)          # last, so that NUMBERS_v7.md also lists this run's timings
    print(json.dumps(timings, indent=1))


if __name__ == "__main__":
    main()
