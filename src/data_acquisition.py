"""v6 Phase 1: acquire the COMPLETE MYH7 ClinVar missense set and annotate it.

Why this replaces the v5 acquisition
------------------------------------
v5 used the MyVariant.info query `clinvar.gene.symbol:MYH7`. MyVariant stores ONE gene
symbol per ClinVar record, and for records in the MYH7 region that ClinVar attaches to
several overlapping gene records (LOC126861898, LOC126861897, MHRT, MIR208B, MYH6 ...)
it often keeps a different symbol, so those variants are never returned. See
src/diagnose_gap.py for the quantified diagnosis.

v6 sources
----------
* ClinVar: NCBI bulk file variant_summary.txt.gz (GRCh37 + GRCh38 rows). Records whose
  GeneSymbol list CONTAINS MYH7, single nucleotide variants, protein change parsed from
  Name (NM_000257.4 numbering). Cross-checked against an E-utilities esearch count.
* Predictors (dbNSFP 4.8a via MyVariant.info) and CADD: POST batch query by hg19 _id.
* Allele frequency: gnomAD v4 (GraphQL API, dataset gnomad_r4, GRCh38) is PRIMARY.
  gnomAD v2.1.1 (the version MyVariant.info serves, GRCh37) is kept as a sensitivity.
  Coordinates are mapped GRCh38 <-> GRCh37 with the constant offset of this gene
  region, validated against ClinVar's own paired GRCh37/GRCh38 coordinates.

Run from the v6 directory:  python -m src.data_acquisition
"""
from __future__ import annotations
import datetime as dt
import gzip
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

GENE = "MYH7"
RAW = Path("data/raw")
OUT = Path("data/processed")
LOGS = Path("logs")
for p in (RAW, OUT, LOGS):
    p.mkdir(parents=True, exist_ok=True)

VS_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
VS_GZ = RAW / "variant_summary.txt.gz"
VS_MYH7 = RAW / "myh7_variant_summary_raw.tsv"
MYH7_LEN = 1935

AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E",
       "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F",
       "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}

STAR = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting classifications": 1,
    "criteria provided, conflicting interpretations": 1,
    "no assertion criteria provided": 0,
    "no assertion provided": 0,
    "no classification provided": 0,
    "no classifications from unflagged records": 0,
}

# MyVariant fields: identical predictor panel to v5 (so results are comparable)
FIELDS = ",".join([
    "clinvar.rcv", "clinvar.variant_id", "clinvar.gene.symbol", "clinvar.hgvs.protein",
    "dbnsfp.genename", "dbnsfp.aa", "dbnsfp.hgvsp", "dbnsfp.hg19", "dbnsfp.hg38",
    "dbnsfp.sift.score", "dbnsfp.sift.pred", "dbnsfp.sift4g.score",
    "dbnsfp.polyphen2.hdiv.score", "dbnsfp.polyphen2.hdiv.pred",
    "dbnsfp.polyphen2.hvar.score",
    "dbnsfp.revel.score", "dbnsfp.alphamissense.score", "dbnsfp.alphamissense.pred",
    "dbnsfp.metalr.score", "dbnsfp.clinpred.score", "dbnsfp.primateai.score",
    "dbnsfp.vest4.score", "dbnsfp.provean.score", "dbnsfp.mpc.score",
    "dbnsfp.esm1b.score",
    "dbnsfp.phylop.100way_vertebrate.score", "dbnsfp.gerp++.rs",
    "cadd.phred",
    "gnomad_exome.af.af", "gnomad_genome.af.af",
])


# ---------------------------------------------------------------- helpers
def _get(d, path, default=None):
    cur = d
    for k in path.split("."):
        if isinstance(cur, list):
            cur = cur[0] if cur else {}
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _num(v):
    if isinstance(v, list):
        v = next((x for x in v if x is not None), None)
    return v


def sha256(path: Path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- ClinVar bulk file
def download_variant_summary(force=False) -> dict:
    """Download the bulk file if absent; return provenance (never silently reuses)."""
    meta = {}
    head = requests.head(VS_URL, timeout=60)
    meta["server_last_modified"] = head.headers.get("Last-Modified")
    meta["content_length_server"] = head.headers.get("Content-Length")
    if force or not VS_GZ.exists():
        print(f"[acquire] downloading {VS_URL}")
        with requests.get(VS_URL, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(VS_GZ, "wb") as f:
                for ch in r.iter_content(1 << 20):
                    f.write(ch)
    meta["retrieved_utc"] = dt.datetime.utcfromtimestamp(VS_GZ.stat().st_mtime).isoformat() + "Z"
    meta["sha256"] = sha256(VS_GZ)
    meta["bytes"] = VS_GZ.stat().st_size
    return meta


def extract_myh7_rows():
    """Stream the 450 MB file once and keep rows whose GeneSymbol list contains MYH7."""
    if VS_MYH7.exists():
        return
    print("[acquire] filtering variant_summary for MYH7 ...")
    with gzip.open(VS_GZ, "rt") as f, open(VS_MYH7, "w") as out:
        header = f.readline()
        out.write(header)
        gi = header.rstrip("\n").split("\t").index("GeneSymbol")
        for line in f:
            sym = line.split("\t", gi + 1)[gi]
            if GENE in sym.split(";"):
                out.write(line)


PROT = re.compile(r"\(p\.([A-Za-z]{3})(\d+)([A-Za-z]{3}|=|\*)\)")


def classify_clinsig(raw: str):
    """ClinVar aggregate germline classification -> (clinsig, y).

    Consistent with the v5 logic: conflicting -> VUS bucket; uncertain -> VUS;
    pathogenic/likely pathogenic (also 'low penetrance' qualifiers) -> P;
    benign/likely benign -> B; everything else (not provided, risk factor ...) -> OTHER.
    """
    s = (raw or "").strip().lower()
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


def parse_clinvar() -> pd.DataFrame:
    d = pd.read_csv(VS_MYH7, sep="\t", low_memory=False)
    d = d[d["GeneSymbol"].str.split(";").apply(lambda g: GENE in g)]
    d = d.rename(columns={"#AlleleID": "AlleleID"})
    g37 = d[d.Assembly == "GRCh37"].copy()
    g38 = d[d.Assembly == "GRCh38"].copy()
    g37 = g37[g37.Type == "single nucleotide variant"].copy()
    g38 = g38[g38.Type == "single nucleotide variant"].copy()
    assert g37.VariationID.is_unique, "duplicate VariationID in GRCh37 rows"
    m = g37["Name"].str.extract(PROT)
    g37["ref3"], g37["pos_p"], g37["alt3"] = m[0], m[1], m[2]
    g37["is_missense"] = (g37.pos_p.notna() & ~g37.alt3.isin(["=", "Ter", "*"])
                          & (g37.ref3 != g37.alt3) & g37.ref3.isin(AA3) & g37.alt3.isin(AA3))
    g37["pos_p"] = pd.to_numeric(g37.pos_p, errors="coerce")
    cs = g37["ClinicalSignificance"].apply(classify_clinsig)
    g37["clinsig"] = [c[0] for c in cs]
    g37["y"] = [c[1] for c in cs]
    g37["stars"] = g37["ReviewStatus"].str.lower().map(STAR).fillna(0).astype(int)
    g37["n_submitters"] = pd.to_numeric(g37["NumberSubmitters"], errors="coerce").fillna(0).astype(int)
    g37["hgvs_g"] = ("chr14:g." + g37.PositionVCF.astype(int).astype(str)
                     + g37.ReferenceAlleleVCF + ">" + g37.AlternateAlleleVCF)
    # hg38 coordinates from the paired GRCh38 rows (key = VariationID)
    g38 = g38.set_index("VariationID")
    g37["pos38"] = g37.VariationID.map(g38["PositionVCF"])
    out = pd.DataFrame({
        "hgvs_g": g37.hgvs_g, "clinvar_variant_id": g37.VariationID,
        "pos19": g37.PositionVCF.astype(int), "pos38": g37.pos38,
        "ref": g37.ReferenceAlleleVCF, "alt": g37.AlternateAlleleVCF,
        "clinsig": g37.clinsig, "clinsig_raw": g37.ClinicalSignificance, "y": g37.y,
        "stars": g37.stars, "review_status": g37.ReviewStatus,
        "n_submitters": g37.n_submitters, "last_evaluated": g37.LastEvaluated,
        "gene_symbols_clinvar": g37.GeneSymbol, "name": g37.Name,
        "is_missense_clinvar": g37.is_missense,
        "aapos_clinvar": g37.pos_p,
        "aaref_clinvar": g37.ref3.map(AA3), "aaalt_clinvar": g37.alt3.map(AA3),
    }).reset_index(drop=True)
    return out


def eutils_missense_count() -> dict:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    res = {}
    for key, term in {
        "missense_any_type": 'MYH7[gene] AND "missense variant"[molecular consequence]',
        "missense_snv": ('MYH7[gene] AND "single nucleotide variant"[Type of variation] '
                         'AND "missense variant"[molecular consequence]'),
        "all_myh7_records": "MYH7[gene]",
    }.items():
        time.sleep(0.5)
        r = requests.get(base, params={"db": "clinvar", "term": term, "retmode": "json",
                                       "retmax": 0}, timeout=60).json()
        res[key] = int(r["esearchresult"]["count"])
        res[key + "_term"] = term
    return res


# ---------------------------------------------------------------- gnomAD v4
def fetch_gnomad_v4() -> pd.DataFrame:
    q = '''query{ gene(gene_symbol:"MYH7", reference_genome:GRCh38){
        variants(dataset:gnomad_r4){ variant_id pos ref alt consequence hgvsp transcript_id
          exome{ac an af} genome{ac an af} } } }'''
    r = requests.post("https://gnomad.broadinstitute.org/api", json={"query": q}, timeout=300)
    r.raise_for_status()
    j = r.json()
    assert "errors" not in j, j.get("errors")
    rows = []
    for v in j["data"]["gene"]["variants"]:
        ex, ge = v.get("exome") or {}, v.get("genome") or {}
        rows.append({"variant_id38": v["variant_id"], "pos38": v["pos"], "ref": v["ref"],
                     "alt": v["alt"], "consequence": v["consequence"], "hgvsp": v["hgvsp"],
                     "ac_exome": ex.get("ac"), "an_exome": ex.get("an"), "af_exome": ex.get("af"),
                     "ac_genome": ge.get("ac"), "an_genome": ge.get("an"), "af_genome": ge.get("af")})
    g = pd.DataFrame(rows)
    ac = g[["ac_exome", "ac_genome"]].fillna(0).sum(axis=1)
    g["ac_total"] = ac
    # same convention as v5: AF = max(exome AF, genome AF), absent = not observed
    g["gnomad_af_v4"] = g[["af_exome", "af_genome"]].max(axis=1)
    g.loc[g.ac_total == 0, "gnomad_af_v4"] = np.nan
    return g


def lift_offset(clinvar: pd.DataFrame) -> int:
    """GRCh37 - GRCh38 offset in the MYH7 region; must be constant."""
    ok = clinvar.dropna(subset=["pos38"])
    diff = (ok.pos19 - ok.pos38.astype(int))
    assert diff.nunique() == 1, f"non-constant liftover offset: {diff.value_counts().head()}"
    return int(diff.iloc[0])


# ---------------------------------------------------------------- MyVariant annotation
def myvariant_post(ids, fields=FIELDS, chunk=1000):
    hits = {}
    for i in range(0, len(ids), chunk):
        part = list(ids[i:i + chunk])
        for attempt in range(4):
            try:
                r = requests.post("https://myvariant.info/v1/variant",
                                  data={"ids": ",".join(part), "fields": fields,
                                        "assembly": "hg19"}, timeout=300)
                r.raise_for_status()
                break
            except Exception as e:  # noqa
                print("   retry", attempt, e)
                time.sleep(3)
        else:
            raise RuntimeError("MyVariant POST failed")
        for h in r.json():
            hits[h["query"]] = h
        print(f"   annotated {min(i + chunk, len(ids))}/{len(ids)}")
    return hits


def aggregate_clinvar_v5(rcv):
    """EXACT v5 aggregation of MyVariant clinvar.rcv records (any-RCV pathogenic wins).

    Kept only for the label-rule sensitivity analysis: it reproduces how v5 labelled
    variants (May-2025 MyVariant ClinVar snapshot)."""
    if rcv is None:
        return None, None, 0, 0
    recs = rcv if isinstance(rcv, list) else [rcv]
    sigs, stars, nsub = [], 0, 0
    for r in recs:
        if not isinstance(r, dict):
            continue
        cs = (r.get("clinical_significance") or "").strip()
        if cs:
            sigs.append(cs.lower())
        stars = max(stars, STAR.get((r.get("review_status") or "").lower(), 0))
        try:
            nsub += int(r.get("number_submitters") or 0)
        except (TypeError, ValueError):
            pass
    if not sigs:
        return None, None, stars, nsub
    has_p = any("pathogenic" in s and "benign" not in s for s in sigs)
    has_b = any("benign" in s and "pathogenic" not in s for s in sigs)
    has_u = any("uncertain" in s for s in sigs)
    has_conf = any("conflicting" in s for s in sigs)
    if has_conf or (has_p and has_b):
        return "Conflicting", "VUS", stars, nsub
    if has_u and not (has_p or has_b):
        return "Uncertain significance", "VUS", stars, nsub
    if has_p and not has_b:
        lp = all("likely" in s for s in sigs if "pathogenic" in s)
        return ("Likely pathogenic" if lp else "Pathogenic"), "P", stars, nsub
    if has_b and not has_p:
        lb = all("likely" in s for s in sigs if "benign" in s)
        return ("Likely benign" if lb else "Benign"), "B", stars, nsub
    return "other", "OTHER", stars, nsub


def hit_to_row(h):
    cs5, y5, st5, ns5 = aggregate_clinvar_v5(_get(h, "clinvar.rcv") if not h.get("notfound") else None)
    gexo = _num(_get(h, "gnomad_exome.af.af"))
    ggen = _num(_get(h, "gnomad_genome.af.af"))
    af = max([x for x in [gexo, ggen] if x is not None], default=None)
    return {
        "mv_found": not h.get("notfound", False),
        "clinsig_v5rcv": cs5, "y_v5rcv": y5, "stars_v5rcv": st5, "nsub_v5rcv": ns5,
        "mv_clinvar_gene_symbol": str(_get(h, "clinvar.gene.symbol")),
        "mv_clinvar_variant_id": _get(h, "clinvar.variant_id"),
        "mv_genename": str(_get(h, "dbnsfp.genename")),
        "aaref": _get(h, "dbnsfp.aa.ref"), "aaalt": _get(h, "dbnsfp.aa.alt"),
        "aapos": _num(_get(h, "dbnsfp.aa.pos")),
        "sift": _get(h, "dbnsfp.sift.score"), "sift4g": _get(h, "dbnsfp.sift4g.score"),
        "polyphen2_hdiv": _get(h, "dbnsfp.polyphen2.hdiv.score"),
        "polyphen2_hvar": _get(h, "dbnsfp.polyphen2.hvar.score"),
        "revel": _get(h, "dbnsfp.revel.score"),
        "alphamissense": _get(h, "dbnsfp.alphamissense.score"),
        "am_pred": _get(h, "dbnsfp.alphamissense.pred"),
        "metalr": _get(h, "dbnsfp.metalr.score"), "clinpred": _get(h, "dbnsfp.clinpred.score"),
        "primateai": _get(h, "dbnsfp.primateai.score"), "vest4": _get(h, "dbnsfp.vest4.score"),
        "provean": _get(h, "dbnsfp.provean.score"), "mpc": _get(h, "dbnsfp.mpc.score"),
        "esm1b": _get(h, "dbnsfp.esm1b.score"),
        "cadd_phred": _get(h, "cadd.phred"),
        "phylop100": _get(h, "dbnsfp.phylop.100way_vertebrate.score"),
        "gerp": _get(h, "dbnsfp.gerp++.rs"),
        "gnomad_af_v2": af,
    }


NUMCOLS = ["aapos", "sift", "sift4g", "polyphen2_hdiv", "polyphen2_hvar", "revel",
           "alphamissense", "metalr", "clinpred", "primateai", "vest4", "provean", "mpc",
           "esm1b", "cadd_phred", "phylop100", "gerp", "gnomad_af_v2"]


def main():
    prov = {"run_started_utc": dt.datetime.utcnow().isoformat() + "Z"}
    import os
    prov["clinvar_variant_summary"] = download_variant_summary(force=bool(os.environ.get("FORCE_DOWNLOAD")))
    extract_myh7_rows()
    cv = parse_clinvar()
    prov["clinvar_n_snv_grch37"] = int(len(cv))
    prov["clinvar_n_missense_by_name"] = int(cv.is_missense_clinvar.sum())
    prov["eutils"] = eutils_missense_count()
    off = lift_offset(cv)
    prov["hg19_minus_hg38_offset"] = off
    print(f"[acquire] ClinVar SNVs: {len(cv)}; missense by Name: {cv.is_missense_clinvar.sum()}; "
          f"E-utils missense SNV: {prov['eutils']['missense_snv']}; offset {off}")

    # --- gnomAD v4 (GRCh38) ---
    gn = fetch_gnomad_v4()
    gn["pos19"] = gn.pos38 + off
    gn["hgvs_g"] = "chr14:g." + gn.pos19.astype(str) + gn.ref + ">" + gn.alt
    mm = gn[gn.consequence == "missense_variant"].copy()
    pm = mm.hgvsp.astype(str).str.extract(r"p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})")
    mm["aapos_gn"] = pd.to_numeric(pm[1], errors="coerce")
    mm["aaref_gn"], mm["aaalt_gn"] = pm[0].map(AA3), pm[2].map(AA3)
    prov["gnomad_v4"] = {"dataset": "gnomad_r4", "reference_genome": "GRCh38",
                         "n_variants_all": int(len(gn)), "n_missense": int(len(mm)),
                         "n_missense_observed_ac_gt0": int((mm.ac_total > 0).sum()),
                         "retrieved_utc": dt.datetime.utcnow().isoformat() + "Z"}
    gn.to_parquet(RAW / "gnomad_r4_myh7_all_variants.parquet", index=False)

    # --- annotate union of ids by MyVariant POST ---
    ids = sorted(set(cv.hgvs_g) | set(mm.hgvs_g))
    print(f"[acquire] annotating {len(ids)} variants via MyVariant.info POST ...")
    hits = myvariant_post(ids)
    ann = pd.DataFrame([{"hgvs_g": i, **hit_to_row(hits[i])} for i in ids])
    for c in NUMCOLS:
        ann[c] = pd.to_numeric(ann[c].apply(_num), errors="coerce")
    prov["myvariant"] = {"retrieved_utc": dt.datetime.utcnow().isoformat() + "Z",
                         "n_ids_posted": len(ids), "n_found": int(ann.mv_found.sum())}
    try:
        meta = requests.get("https://myvariant.info/v1/metadata", timeout=60).json()
        prov["myvariant"]["build_date"] = meta.get("build_date")
        prov["myvariant"]["source_versions"] = {k: meta["src"][k].get("version")
                                                for k in ("clinvar", "dbnsfp", "gnomad", "cadd")
                                                if k in meta["src"]}
    except Exception as e:  # noqa
        prov["myvariant"]["metadata_error"] = str(e)

    # --- assemble variant table: ClinVar rows + gnomAD-only rows ---
    cv["in_clinvar"] = True
    gcols = ["hgvs_g", "gnomad_af_v4", "ac_total", "hgvsp", "aapos_gn", "aaref_gn", "aaalt_gn"]
    mm2 = mm[gcols].rename(columns={"ac_total": "ac_v4", "hgvsp": "hgvsp_gnomad"})
    allv = cv.merge(mm2, on="hgvs_g", how="outer")
    allv["in_clinvar"] = allv["in_clinvar"].fillna(False).astype(bool)
    allv["in_gnomad_v4_missense"] = allv.gnomad_af_v4.notna() | allv.hgvsp_gnomad.notna()
    allv = allv.merge(ann, on="hgvs_g", how="left")
    # amino acid position/ref/alt: ClinVar Name first (NM_000257.4), then gnomAD, then dbNSFP
    allv["aapos_dbnsfp"] = allv["aapos"]
    allv["aapos"] = allv.aapos_clinvar.combine_first(allv.aapos_gn).combine_first(allv.aapos_dbnsfp)
    allv["aaref"] = allv.aaref_clinvar.combine_first(allv.aaref_gn).combine_first(allv.aaref)
    allv["aaalt"] = allv.aaalt_clinvar.combine_first(allv.aaalt_gn).combine_first(allv.aaalt)
    allv["hgvs_p"] = allv["name"].str.extract(r"\((p\.[^)]+)\)")[0].combine_first(allv.hgvsp_gnomad)
    allv["y"] = allv["y"].fillna("not_in_clinvar")
    allv["is_missense"] = allv.is_missense_clinvar.fillna(False).astype(bool) | allv.in_gnomad_v4_missense
    allv["gnomad_af_v4"] = allv.gnomad_af_v4
    allv["absent_from_gnomad_v4"] = allv.gnomad_af_v4.isna()
    allv.to_parquet(OUT / "myh7_all_variants_annotated.parquet", index=False)

    # diagnostic table: which ClinVar missense variants did the v5 query lose?
    prov["myvariant"]["clinvar_missense_not_found_by_symbol_query"] = None  # filled by diagnose_gap

    (LOGS / "provenance_acquisition.json").write_text(json.dumps(prov, indent=2, default=str))
    ms = allv[allv.is_missense]
    print(f"[acquire] missense records (ClinVar U gnomAD v4): {len(ms)}; "
          f"ClinVar missense: {int(allv.is_missense_clinvar.fillna(False).sum())}; "
          f"gnomAD-only missense: {int((~allv.in_clinvar & allv.is_missense).sum())}")
    print(f"[acquire] saved {OUT/'myh7_all_variants_annotated.parquet'}; "
          f"provenance -> {LOGS/'provenance_acquisition.json'}")


if __name__ == "__main__":
    main()
