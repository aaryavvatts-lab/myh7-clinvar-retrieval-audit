"""Amendment A1 (PREREG): extend the candidate pool to the 100 genes with most ClinVar missense SNVs and apply the
SAME inclusion rule (I1 counts first, I2 AlphaFold availability second; I3/I4 are applied by src.mg_labels after the
gnomAD/annotation fetch). Output: data/interim/extra_genes.csv (genes passing I1 and I2) and
results/tables/v7_pool_screen.csv (every top-100 gene with the first failed criterion).
Run: python -m src.mg_pool
"""
from __future__ import annotations
import time
import pandas as pd
import requests
from src.genes import *   # noqa


def uniprot_for(gene):
    r = requests.get("https://rest.uniprot.org/uniprotkb/search", timeout=60,
                     params={"query": f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true",
                             "fields": "accession,length,gene_primary", "format": "tsv", "size": 5})
    lines = r.text.strip().split("\n")[1:]
    for ln in lines:
        acc, ln_, g = ln.split("\t")[:3]
        if g == gene:
            return acc, int(ln_)
    return None, None


def af_single(up):
    r = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{up}", timeout=60)
    if r.status_code != 200 or not r.json():
        return False, "no AlphaFold DB entry"
    f1 = [e for e in r.json() if e["entryId"] == f"AF-{up}-F1"]
    if not f1:
        return False, "no canonical F1"
    e = f1[0]
    ok = e["uniprotStart"] == 1 and e["uniprotEnd"] == len(e["uniprotSequence"])
    return ok, "" if ok else "F1 is not full-length"


def main():
    cv = pd.read_parquet(INTERIM / "clinvar_missense_snv.parquet")
    top = list(cv.gene.value_counts().head(100).index)
    rows, keep = [], []
    key = ["aaref", "aapos", "aaalt"]
    for g in top:
        c = cv[(cv.gene == g) & cv.y.isin(["P", "B", "VUS"])]
        a = c.groupby(key).agg(nP=("y", lambda s: (s == "P").sum()), nB=("y", lambda s: (s == "B").sum()),
                               nV=("y", lambda s: (s == "VUS").sum())).reset_index()
        nP = int(((a.nP > 0) & (a.nB == 0)).sum()); nB = int(((a.nB > 0) & (a.nP == 0) & (a.nV == 0)).sum())
        row = {"gene": g, "in_panel19": g in GENES, "clinvar_missense_nt": int((cv.gene == g).sum()), "n_P_T2": nP, "n_B_T2": nB,
               "I1": bool(nP >= MIN_P and nB >= MIN_B), "uniprot": None, "I2": None, "first_failed": ""}
        if g in GENES:
            row["first_failed"] = "pre-listed (see v7_inclusion.csv)"
        elif not row["I1"]:
            row["first_failed"] = "I1"
        else:
            up, L = uniprot_for(g)
            row["uniprot"] = up
            if up is None:
                row["I2"] = False; row["first_failed"] = "I2 no reviewed UniProt entry"
            else:
                ok, why = af_single(up)
                row["I2"] = ok
                if not ok:
                    row["first_failed"] = f"I2 {why}"
                else:
                    keep.append((g, up))
            time.sleep(0.3)
        rows.append(row)
    pd.DataFrame(rows).to_csv(TAB / "v7_pool_screen.csv", index=False)
    pd.DataFrame(keep, columns=["gene", "uniprot"]).to_csv(INTERIM / "extra_genes.csv", index=False)
    print(pd.DataFrame(rows).to_string()); print("pass I1+I2:", keep)


if __name__ == "__main__":
    main()
