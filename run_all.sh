#!/usr/bin/env bash
# One command that regenerates EVERY table and figure of v7 (legacy v6 chain, paper_extras, figures_paper, and all
# v7 analyses: tiers, estimand, inference, multi-gene study, retrieval audit, figures, verdicts, NUMBERS_v7.md).
# Usage:  bash run_all.sh          (from anywhere). Interpreter: $PYTHON if set, else ./.venv/bin/python, else
#         ../.venv/bin/python (the project-level environment that sits beside this folder), else python3.
#         requirements.lock.txt lists the exact package versions used. LaTeX engine: $TECTONIC, else tectonic on PATH.
# Needs network ONLY for the first run (ClinVar, gnomAD GraphQL, MyVariant.info, AlphaFold DB, UniProt, ClinGen erepo);
# afterwards the archived raw files in data/ are reused. FORCE_DOWNLOAD=1 refetches everything.
# Data inputs are all inside this folder (the truncated v5 table is the frozen snapshot data/legacy_v5, SHA256SUMS).
# The last step builds the paper figures (figures_paper_v7).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x ./.venv/bin/python ]; then PY=./.venv/bin/python
elif [ -x ../.venv/bin/python ]; then PY=../.venv/bin/python
else PY=python3; fi
echo "interpreter: $PY"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
{ echo "run started (UTC): $STAMP"; "$PY" --version 2>&1; "$PY" -m pip freeze 2>/dev/null; } > logs/run_${STAMP}_env.log
START=$(date +%s)
"$PY" -m src.run_pipeline 2>&1 | tee logs/run_${STAMP}.log
END=$(date +%s)
echo "finished (UTC): $(date -u +%Y%m%dT%H%M%SZ); wall clock $((END-START)) s" | tee -a logs/run_${STAMP}.log
