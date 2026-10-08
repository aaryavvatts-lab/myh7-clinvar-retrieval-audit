"""v7 step: reduce the ClinVar bulk file to missense SNV rows for ALL genes (one pass), keyed by VariationID,
with hg19 and hg38 coordinates. Output: data/interim/clinvar_missense_snv.parquet

A row is kept when Type = single nucleotide variant and Name carries a missense protein change
p.Xxx123Yyy (3-letter, ref != alt, neither Ter). The gene of the protein change is the gene symbol inside
the transcript parentheses of Name, e.g. NM_000257.4(MYH7):c.5A>G (p.Xxx2Yyy) -> MYH7. The full GeneSymbol
list (semicolon separated, may include overlapping loci) is kept for the retrieval audit.
"""
from __future__ import annotations
import datetime as dt, hashlib, json, re
import numpy as np
import pandas as pd
import requests
from src.genes import *   # noqa

VS_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
VS_GZ = RAW / "variant_summary.txt.gz"
USE = ["Type", "Name", "GeneSymbol", "ClinicalSignificance", "LastEvaluated", "Assembly", "ReviewStatus",
       "NumberSubmitters", "VariationID", "Chromosome", "PositionVCF", "ReferenceAlleleVCF", "AlternateAlleleVCF"]
STAR = {"practice guideline": 4, "reviewed by expert panel": 3,
        "criteria provided, multiple submitters, no conflicts": 2,
        "criteria provided, single submitter": 1,
        "criteria provided, conflicting classifications": 1,
        "criteria provided, conflicting interpretations": 1,
        "no assertion criteria provided": 0, "no assertion provided": 0,
        "no classification provided": 0, "no classifications from unflagged records": 0}
PROT = re.compile(r"^(?:.*)\((?P<gene>[^()]+)\):.*\(p\.(?P<ref>[A-Z][a-z]{2})(?P<pos>\d+)(?P<alt>[A-Z][a-z]{2})\)")


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def download(force=False):
    meta = {}
    head = requests.head(VS_URL, timeout=60)
    meta["server_last_modified_now"] = head.headers.get("Last-Modified")
    meta["server_content_length_now"] = head.headers.get("Content-Length")
    if force or not VS_GZ.exists():
        with requests.get(VS_URL, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(VS_GZ, "wb") as f:
                for ch in r.iter_content(1 << 20):
                    f.write(ch)
    meta["file_mtime_utc"] = dt.datetime.utcfromtimestamp(VS_GZ.stat().st_mtime).isoformat() + "Z"
    meta["sha256"] = sha256(VS_GZ); meta["bytes"] = VS_GZ.stat().st_size
    return meta


def classify(raw):
    s = (raw or "").strip().lower() if isinstance(raw, str) else ""
    if not s or s in ("-", "not provided", "no classifications from unflagged records"):
        return "Not classified", "OTHER"
    if "conflicting" in s:
        return "Conflicting", "VUS"
    has_p = "pathogenic" in s and not s.startswith("benign")
    has_b = "benign" in s
    has_u = "uncertain significance" in s
    if has_p and has_b:
        return "Conflicting", "VUS"
    if has_u and not (has_p or has_b):
        return "Uncertain significance", "VUS"
    if has_p:
        return ("Likely pathogenic" if s.startswith("likely") else "Pathogenic"), "P"
    if has_b:
        return ("Likely benign" if s.startswith("likely") else "Benign"), "B"
    return raw[:60], "OTHER"


def main():
    import os
    meta = download(force=bool(os.environ.get("FORCE_DOWNLOAD")))
    parts = []
    n_rows = 0
    for ch in pd.read_csv(VS_GZ, sep="\t", usecols=USE, chunksize=400000, low_memory=False):
        n_rows += len(ch)
        ch = ch[(ch.Type == "single nucleotide variant") & ch.Name.str.contains(r"\(p\.", na=False)].copy()
        m = ch.Name.str.extract(PROT)
        ch = ch.join(m)
        ch = ch[ch.pos.notna() & ch.ref.isin(AA3) & ch.alt.isin(AA3) & (ch.ref != ch.alt)]
        parts.append(ch)
    d = pd.concat(parts, ignore_index=True)
    d["pos"] = d.pos.astype(int)
    g37 = d[d.Assembly == "GRCh37"].drop_duplicates("VariationID").set_index("VariationID")
    g38 = d[d.Assembly == "GRCh38"].drop_duplicates("VariationID").set_index("VariationID")
    ids = g37.index.union(g38.index)
    base = g37.reindex(ids)
    miss = base.Name.isna()
    base.loc[miss] = g38.reindex(ids[miss]).values
    out = pd.DataFrame({
        "VariationID": ids, "gene": base.gene.values, "gene_symbols_all": base.GeneSymbol.values,
        "name": base.Name.values, "aaref": base.ref.map(AA3).values, "aapos": base.pos.values,
        "aaalt": base.alt.map(AA3).values, "clinsig_raw": base.ClinicalSignificance.values,
        "review_status": base.ReviewStatus.values, "last_evaluated": base.LastEvaluated.values,
        "chrom": base.Chromosome.astype(str).values})
    out["stars"] = out.review_status.str.lower().map(STAR).fillna(0).astype(int)
    cs = out.clinsig_raw.apply(classify)
    out["clinsig"] = [c[0] for c in cs]; out["y"] = [c[1] for c in cs]
    s37 = g37.reindex(ids); s38 = g38.reindex(ids)
    out["pos19"] = s37.PositionVCF.values; out["ref19"] = s37.ReferenceAlleleVCF.values; out["alt19"] = s37.AlternateAlleleVCF.values
    out["pos38"] = s38.PositionVCF.values; out["ref38"] = s38.ReferenceAlleleVCF.values; out["alt38"] = s38.AlternateAlleleVCF.values
    out["has_hg19"] = out.pos19.notna(); out["has_hg38"] = out.pos38.notna()
    out["id19"] = np.where(out.has_hg19, "chr" + out.chrom + ":g." + out.pos19.fillna(0).astype(int).astype(str)
                           + out.ref19.fillna("") + ">" + out.alt19.fillna(""), None)
    out["id38"] = np.where(out.has_hg38, "chr" + out.chrom + ":g." + out.pos38.fillna(0).astype(int).astype(str)
                           + out.ref38.fillna("") + ">" + out.alt38.fillna(""), None)
    out.to_parquet(INTERIM / "clinvar_missense_snv.parquet", index=False)
    meta.update(rows_scanned=int(n_rows), missense_snv_variation_ids=int(len(out)),
                n_genes=int(out.gene.nunique()),
                max_variation_id=int(out.VariationID.max()))
    (LOGS / "provenance_clinvar_v7.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta, indent=1, default=str))


if __name__ == "__main__":
    main()
