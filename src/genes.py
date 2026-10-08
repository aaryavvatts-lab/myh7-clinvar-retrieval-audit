"""Gene panel (fixed in PREREG.md section 1) and shared constants for the v7 multi-gene study."""
from __future__ import annotations
from pathlib import Path

PANEL = [  # symbol, UniProt canonical
    ("MYH7", "P12883"), ("MYBPC3", "Q14896"), ("TNNT2", "P45379"), ("TNNI3", "P19429"),
    ("TPM1", "P09493"), ("ACTC1", "P68032"), ("MYL2", "P10916"), ("MYL3", "P08590"),
    ("LDLR", "P01130"), ("FBN1", "P35555"), ("SCN5A", "Q14524"), ("KCNQ1", "P51787"),
    ("KCNH2", "Q12809"), ("PTEN", "P60484"), ("MLH1", "P40692"), ("MSH2", "P43246"),
    ("MSH6", "P52701"), ("BRCA1", "P38398"), ("TP53", "P04637"),
]
UNIPROT = dict(PANEL)
GENES = [g for g, _ in PANEL]
SARCOMERE = ["MYH7", "MYBPC3", "TNNT2", "TNNI3", "TPM1", "ACTC1", "MYL2", "MYL3"]

# inclusion thresholds (PREREG I1-I4)
MIN_P, MIN_B = 100, 20
MIN_SEQ_MATCH = 0.95
MIN_PRED_AVAIL = 0.80

# benign tiers / stand-in thresholds (PREREG section 2)
FAF_BS1, FAF_BA1, MIN_AC_GROUP = 2e-4, 1e-3, 5
POPMAX_GROUPS = ["afr", "amr", "eas", "nfe", "sas"]

# score pseudo-count (PREREG section 3)
K_SHRINK = 1.0

RAW = Path("data/raw"); INTERIM = Path("data/interim"); EXT = Path("data/external")
PROC = Path("data/processed"); LOGS = Path("logs")
TAB = Path("results/tables"); FIG = Path("results/figures")
for _p in (RAW, INTERIM, EXT, PROC, LOGS, TAB, FIG, RAW / "gnomad", RAW / "myvariant", EXT / "alphafold"):
    _p.mkdir(parents=True, exist_ok=True)

AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
       "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
       "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}


# ---- Amendment A1: extra genes from the top-100 pool (written by src.mg_pool) --------------------------
def _load_extra():
    import csv
    f = INTERIM / "extra_genes.csv"
    out = []
    if f.exists():
        for r in csv.DictReader(open(f)):
            if r.get("uniprot"):
                out.append((r["gene"], r["uniprot"]))
    return out


EXTRA = _load_extra()
for _g, _u in EXTRA:
    UNIPROT.setdefault(_g, _u)
EXTRA_GENES = [g for g, _ in EXTRA]
