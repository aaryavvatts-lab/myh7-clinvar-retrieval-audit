# myh7-clinvar-retrieval-audit

Code, derived tables, logs and the written analysis plan for the GLBIO 2027 submission:

**Gene-symbol queries to MyVariant.info drop ClinVar missense variants without warning, and a 3D label-proximity score shows no gain over a neighbour count** (Aaryav Sharma).

## What is here

| Path | Contents |
|---|---|
| `src/` | All analysis code (retrieval audit, labels and benign sets, 3D scores, inference, figures, paper build) |
| `config/` | Pipeline configuration |
| `results/tables/` | Every derived table behind the paper's numbers |
| `results/NUMBERS_v7.md` | Each number in the paper, with its source file and filter |
| `results/VERDICTS.md` | One mechanical verdict per test in the written plan |
| `logs/` | Timeline, run timings and SHA-256 hashes of the written analysis plan (`prereg_sha256*.txt`); the plan itself is available from the author on request |
| `DATA_AVAILABILITY.md` | Every public source, with URL, release or date, and SHA-256 |

## Not included, and why

* `data/` (about 680 MB of raw downloads). All inputs are public; `DATA_AVAILABILITY.md` lists each source and its SHA-256. The ClinVar release used is pinned by its hash (`variant_summary.txt.gz`, server date 5 October 2026).
* Per-variant predictor scores (dbNSFP, AlphaMissense, REVEL). Their licences must be checked before they are redistributed.
* One frozen intermediate table from an earlier internal analysis (`data/legacy_v5/`). It is non-public and is used only for one supplementary comparison. The legacy steps of `run_all.sh` need it; the retrieval audit and the ten-gene analysis do not.

## Running

```bash
pip install -r requirements.lock.txt   # Python 3.9.6
bash run_all.sh                        # first run downloads the public inputs
```

The live services (MyVariant.info, gnomAD, AlphaFold DB, ClinGen) change over time, so a fresh download will not match the archived data byte for byte. The ClinVar bulk file is a new release every month. A run from a clean clone has not been tested.

## Status

The results are not for clinical use. See the paper for its declarations.
