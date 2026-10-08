"""v7 step: per-gene protein-level table with ALL benign tiers (PREREG sections 1 and 2), predictor
annotation (dbNSFP via MyVariant.info POST by genomic id) and the inclusion-rule accounting.

Unit of analysis: unique protein change (aaref, aapos, aaalt). Outputs:
  data/processed/<GENE>/protein_table.parquet
  results/tables/v7_inclusion.csv, v7_label_accounting.csv, v7_tier_counts.csv,
  v7_erepo_myh7.csv (ClinGen Evidence Repository MYH7 missense calls matched to the cohort)
Run: python -m src.mg_labels [GENE ...]
"""
from __future__ import annotations
import json, re, sys, time
import numpy as np
import pandas as pd
import requests
from scipy.stats import chi2
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from src.genes import *   # noqa

PRED_FIELDS = ",".join([
    "dbnsfp.sift.score", "dbnsfp.sift4g.score", "dbnsfp.polyphen2.hdiv.score", "dbnsfp.polyphen2.hvar.score",
    "dbnsfp.revel.score", "dbnsfp.alphamissense.score", "dbnsfp.metalr.score", "dbnsfp.clinpred.score",
    "dbnsfp.primateai.score", "dbnsfp.vest4.score", "dbnsfp.provean.score", "dbnsfp.mpc.score",
    "dbnsfp.esm1b.score", "dbnsfp.phylop.100way_vertebrate.score", "cadd.phred"])
PCOLS = {"sift": "dbnsfp.sift.score", "sift4g": "dbnsfp.sift4g.score", "polyphen2_hdiv": "dbnsfp.polyphen2.hdiv.score",
         "polyphen2_hvar": "dbnsfp.polyphen2.hvar.score", "revel": "dbnsfp.revel.score",
         "alphamissense": "dbnsfp.alphamissense.score", "metalr": "dbnsfp.metalr.score",
         "clinpred": "dbnsfp.clinpred.score", "primateai": "dbnsfp.primateai.score", "vest4": "dbnsfp.vest4.score",
         "provean": "dbnsfp.provean.score", "mpc": "dbnsfp.mpc.score", "esm1b": "dbnsfp.esm1b.score",
         "phylop100": "dbnsfp.phylop.100way_vertebrate.score", "cadd_phred": "cadd.phred"}
PNAMES = list(PCOLS)
HGVSP = re.compile(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")
EPROT = re.compile(r"\((?P<gene>[^()]+)\):[^()]*\(p\.(?P<ref>[A-Z][a-z]{2})(?P<pos>\d+)(?P<alt>[A-Z][a-z]{2})\)")


def _flat(h, path):
    cur = h
    for k in path.split("."):
        if isinstance(cur, list):
            cur = cur[0] if cur else None
        if not isinstance(cur, dict) or k not in cur:
            return np.nan
        cur = cur[k]
    if isinstance(cur, list):
        cur = next((x for x in cur if x is not None), np.nan)
    try:
        return float(cur)
    except (TypeError, ValueError):
        return np.nan


def myvariant_cached(gene, ids, assembly):
    f = RAW / "myvariant" / f"{gene}_{assembly}.parquet"
    have = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=["id"] + PNAMES)
    todo = [i for i in ids if i not in set(have["id"])]
    rows = []
    for i in range(0, len(todo), 1000):
        part = todo[i:i + 1000]
        for a in range(5):
            try:
                r = requests.post("https://myvariant.info/v1/variant", timeout=300,
                                  data={"ids": ",".join(part), "fields": PRED_FIELDS, "assembly": assembly})
                r.raise_for_status(); break
            except Exception as e:  # noqa
                print("   retry", a, str(e)[:80]); time.sleep(4 * (a + 1))
        else:
            raise RuntimeError("MyVariant POST failed")
        for h in r.json():
            row = {"id": h["query"]}
            for c, p in PCOLS.items():
                row[c] = np.nan if h.get("notfound") else _flat(h, p)
            row["mv_found"] = not h.get("notfound", False)
            rows.append(row)
    if rows:
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
        have.to_parquet(f, index=False)
    return have


def af_sequence(gene):
    p = EXT / "alphafold" / f"AF-{UNIPROT[gene]}-F1-model.pdb"
    s = PDBParser(QUIET=True).get_structure("m", str(p))
    res = [r for r in s[0]["A"]]
    return "".join(seq1(r.get_resname()) for r in res)


def faf95(ac, an):
    ac = np.asarray(ac, float); an = np.asarray(an, float)
    out = np.zeros_like(ac)
    m = (ac > 0) & (an > 0)
    out[m] = chi2.ppf(0.05, 2 * ac[m]) / (2 * an[m])
    return out


def load_erepo():
    d = pd.read_csv(RAW / "erepo_classifications_all.tsv", sep="\t", low_memory=False)
    m = d["HGVS Expressions"].astype(str).str.extract(EPROT)
    d = d.join(m)
    d = d[d.pos.notna() & d.ref.isin(AA3) & d.alt.isin(AA3) & (d.ref != d.alt)].copy()
    d["aapos"] = d.pos.astype(int); d["aaref"] = d.ref.map(AA3); d["aaalt"] = d.alt.map(AA3)
    d["gene"] = d["HGNC Gene Symbol"]
    d["erepo_class"] = d.Assertion.map({"Benign": "B", "Likely Benign": "B", "Pathogenic": "P", "Likely Pathogenic": "P",
                                        "Uncertain Significance": "VUS"})
    return d[["gene", "aaref", "aapos", "aaalt", "erepo_class", "Assertion", "Applied Evidence Codes (Met)", "Expert Panel"]]


def protein_table(gene, cv, erepo, gn, seq):
    c = cv[(cv.gene == gene) & cv.y.isin(["P", "B", "VUS", "OTHER"])].copy()
    key = ["aaref", "aapos", "aaalt"]
    c["is_P"] = c.y.eq("P"); c["is_B"] = c.y.eq("B"); c["is_V"] = c.y.eq("VUS")
    agg = c.groupby(key).agg(n_nt=("VariationID", "nunique"), nP=("is_P", "sum"), nB=("is_B", "sum"), nV=("is_V", "sum"),
                             stars_max=("stars", "max")).reset_index()
    sb = c[c.is_B].groupby(key).stars.max().rename("stars_B_max")
    sp = c[c.is_P].groupby(key).stars.max().rename("stars_P_max")
    agg = agg.merge(sb, on=key, how="left").merge(sp, on=key, how="left")
    agg["in_clinvar"] = True
    # gnomAD (pooled over nucleotide alleles of the same protein change)
    gm = gn[gn.consequence == "missense_variant"].copy()
    pm = gm.hgvsp.astype(str).str.extract(HGVSP)
    gm["aaref"] = pm[0].map(AA3); gm["aapos"] = pd.to_numeric(pm[1], errors="coerce"); gm["aaalt"] = pm[2].map(AA3)
    gm = gm[gm.aaref.notna() & gm.aaalt.notna() & gm.aapos.notna()].copy()
    gm["aapos"] = gm.aapos.astype(int)
    gm["ac_tot"] = gm[["ac_exome", "ac_genome"]].fillna(0).sum(axis=1)
    gm["af_glob"] = gm[["af_exome", "af_genome"]].max(axis=1)
    gm = gm[gm.ac_tot > 0]
    gagg = gm.groupby(key).agg(gn_n_nt=("variant_id38", "nunique"), gn_ac=("ac_tot", "sum"),
                               gn_af_legacy_max=("af_glob", "max")).reset_index()
    if gene == "MYH7":
        best = pd.DataFrame(index=gagg.index)
        gsum = {}
        for grp in POPMAX_GROUPS:
            ac = gm.groupby(key)[[f"exome_{grp}_ac", f"genome_{grp}_ac"]].sum().sum(axis=1)
            an = 0
            for ds in ("exome", "genome"):   # pooled AN; if the allele is absent from a dataset use that dataset's gene-wide median AN
                col = f"{ds}_{grp}_an"
                an = an + gm.groupby(key)[col].mean().fillna(gm[col].median())
            gsum[grp] = (ac, an)
        keys = gagg.set_index(key).index
        fafs = pd.DataFrame({g: pd.Series(faf95(gsum[g][0].reindex(keys).fillna(0).values,
                                                gsum[g][1].reindex(keys).fillna(0).values), index=keys) for g in POPMAX_GROUPS})
        acs = pd.DataFrame({g: gsum[g][0].reindex(keys).fillna(0) for g in POPMAX_GROUPS})
        gagg["faf95_popmax"] = fafs.max(axis=1).values
        gagg["faf95_popmax_group"] = fafs.idxmax(axis=1).values
        gagg["ac_in_popmax_group"] = [acs.loc[k, g] for k, g in zip(keys, gagg.faf95_popmax_group)]
        gagg["gnomad_api_faf95_popmax_max"] = gm.groupby(key)[["faf95_exome", "faf95_genome"]].max().max(axis=1).reindex(keys).values
    allp = agg.merge(gagg, on=key, how="outer")
    allp["in_clinvar"] = allp.in_clinvar.fillna(False).astype(bool)
    allp["in_gnomad"] = allp.gn_ac.notna()
    for col in ("nP", "nB", "nV", "n_nt"):
        allp[col] = allp[col].fillna(0).astype(int)
    # erepo
    er = erepo[erepo.gene == gene].copy()
    if len(er):
        eg = er.groupby(key).agg(erepo_P=("erepo_class", lambda s: (s == "P").any()), erepo_B=("erepo_class", lambda s: (s == "B").any()),
                                 erepo_V=("erepo_class", lambda s: (s == "VUS").any())).reset_index()
        allp = allp.merge(eg, on=key, how="left")
    for col in ("erepo_P", "erepo_B", "erepo_V"):
        allp[col] = allp.get(col, pd.Series(False, index=allp.index)).fillna(False).astype(bool)
    allp["gene"] = gene
    # sequence check (I3)
    L = len(seq)
    allp["seq_ok"] = [(1 <= p <= L and seq[int(p) - 1] == r) for r, p in zip(allp.aaref, allp.aapos)]
    return allp, c


def assign_labels(allp):
    a = allp
    cvP = (a.nP > 0) & (a.nB == 0)
    cvB = (a.nB > 0) & (a.nP == 0) & (a.nV == 0)
    cvV = ~cvP & ~cvB & a.in_clinvar & ((a.nV > 0) | ((a.nP > 0) & (a.nB > 0)) | ((a.nB > 0) & (a.nV > 0)))
    a["clinvar_protein_label"] = np.where(cvP, "P", np.where(cvB, "B", np.where(cvV, "VUS", None)))
    a["lab_T2"] = a.clinvar_protein_label
    # T1: B with >=2 stars or erepo B (and not contradicted by P)
    t1B = cvB & ((a.stars_B_max.fillna(0) >= 2) | a.erepo_B) & ~a.erepo_P
    a["lab_T1"] = np.where(cvP, "P", np.where(t1B, "B", np.where(cvV, "VUS", None)))
    a.loc[cvB & ~t1B, "lab_T1"] = None                 # lower-tier benign excluded in T1 (not counted as VUS)
    # T1-both: P also needs >=2 stars
    p2 = cvP & (a.stars_P_max.fillna(0) >= 2)
    a["lab_T1both"] = np.where(p2, "P", np.where(t1B, "B", np.where(cvV, "VUS", None)))
    a.loc[cvP & ~p2, "lab_T1both"] = None
    a.loc[cvB & ~t1B, "lab_T1both"] = None
    # T3 (MYH7 only): popmax FAF stand-ins
    a["lab_T3"] = a.lab_T2
    a["lab_T3_BA1"] = a.lab_T2
    a["lab_T3legacy_wide"] = a.lab_T2
    a["lab_T3legacy_tight"] = a.lab_T2
    any_cv_conflict = a.in_clinvar & ((a.nP > 0) | (a.nV > 0))
    any_cv_record = a.in_clinvar & ((a.nP + a.nB + a.nV) > 0)
    a["standin_T3"] = False
    if "faf95_popmax" in a:
        st = (a.faf95_popmax.fillna(0) >= FAF_BS1) & (a.ac_in_popmax_group.fillna(0) >= MIN_AC_GROUP) & ~any_cv_conflict & ~a.erepo_P & ~a.erepo_V & a.clinvar_protein_label.isna()
        a["standin_T3"] = st
        a.loc[st, "lab_T3"] = "B"
        ba1 = st & (a.faf95_popmax >= FAF_BA1)
        a.loc[ba1, "lab_T3_BA1"] = "B"
        a["vus_meeting_faf_rule"] = (a.clinvar_protein_label == "VUS") & (a.faf95_popmax.fillna(0) >= FAF_BS1) & (a.ac_in_popmax_group.fillna(0) >= MIN_AC_GROUP)
        a["P_meeting_faf_rule"] = (a.clinvar_protein_label == "P") & (a.faf95_popmax.fillna(0) >= FAF_BS1) & (a.ac_in_popmax_group.fillna(0) >= MIN_AC_GROUP)
    # legacy (v6) global-AF stand-ins; v6 wording: not ClinVar P, VUS or B at all
    af = a.gn_af_legacy_max.fillna(0)
    nolab = ~any_cv_record
    a["standin_legacy_wide"] = (af >= 1e-5) & nolab
    a["standin_legacy_tight"] = (af >= 5e-5) & nolab
    a.loc[a.standin_legacy_wide, "lab_T3legacy_wide"] = "B"
    a.loc[a.standin_legacy_tight, "lab_T3legacy_tight"] = "B"
    return a


TIERS = ["T1", "T2", "T3", "T3legacy_wide", "T3legacy_tight", "T1both", "T3_BA1"]


def main(genes=None):
    t0 = time.time()
    genes = genes or (GENES + EXTRA_GENES)
    cv = pd.read_parquet(INTERIM / "clinvar_missense_snv.parquet")
    erepo = load_erepo()
    prov = json.loads((LOGS / "provenance_acquire_v7.json").read_text())
    inc_rows, acct_rows, tier_rows = [], [], []
    for gene in genes:
        gd = PROC / gene; gd.mkdir(exist_ok=True)
        info = prov[gene]
        row = {"gene": gene, "uniprot": UNIPROT[gene], "af_available": bool(info.get("available")),
               "af_single_full_length": bool(info.get("full_length_single_fragment", False)),
               "af_model_version": info.get("model_version"), "af_len": info.get("seq_len")}
        gn = pd.read_parquet(RAW / "gnomad" / f"{gene}_gnomad_r4.parquet")
        c_all = cv[cv.gene == gene]
        row["clinvar_missense_nt"] = int(len(c_all))
        if not info.get("available"):
            # still count labels (no AF model): minimal accounting for the exclusion report
            seq = None
        else:
            seq = af_sequence(gene)
        if seq is None:
            c = c_all[c_all.y.isin(["P", "B", "VUS", "OTHER"])]
            key = ["aaref", "aapos", "aaalt"]
            g = c.groupby(key).agg(nP=("y", lambda s: (s == "P").sum()), nB=("y", lambda s: (s == "B").sum()),
                                   nV=("y", lambda s: (s == "VUS").sum())).reset_index()
            nPp = int(((g.nP > 0) & (g.nB == 0)).sum()); nBp = int(((g.nB > 0) & (g.nP == 0) & (g.nV == 0)).sum())
            row.update(n_P_T2=nPp, n_B_T2=nBp, seq_match_frac=np.nan, pred_avail_frac=np.nan,
                       I1=bool(nPp >= MIN_P and nBp >= MIN_B), I2=False, I3=False, I4=False, included=False,
                       failed="; ".join(["I2 no AlphaFold DB model (API returned empty)"] + ([] if nPp >= MIN_P and nBp >= MIN_B else ["I1"])))
            inc_rows.append(row); print(f"[labels] {gene}: no AF model; P={nPp} B={nBp}", flush=True)
            continue
        allp, craw = protein_table(gene, cv, erepo, gn, seq)
        allp = assign_labels(allp)
        # annotation for labelled (T2 P/B) protein changes and MYH7 T3 / legacy candidates
        need = allp[allp.lab_T2.isin(["P", "B"]) | allp.standin_T3 | allp.standin_legacy_wide | allp.lab_T3_BA1.eq("B")].copy()
        # nucleotide ids: ClinVar rows for these protein changes, gnomAD rows for gnomAD-only
        cvn = craw.merge(need[["aaref", "aapos", "aaalt"]], on=["aaref", "aapos", "aaalt"])
        ids38 = cvn.id38.dropna().unique().tolist()
        ids19 = cvn[cvn.id38.isna() & cvn.id19.notna()].id19.unique().tolist()
        gm = gn[gn.consequence == "missense_variant"].copy()
        pm = gm.hgvsp.astype(str).str.extract(HGVSP)
        gm["aaref"] = pm[0].map(AA3); gm["aapos"] = pd.to_numeric(pm[1], errors="coerce"); gm["aaalt"] = pm[2].map(AA3)
        gm = gm.dropna(subset=["aaref", "aapos", "aaalt"]); gm["aapos"] = gm.aapos.astype(int)
        gmn = gm.merge(need[["aaref", "aapos", "aaalt"]], on=["aaref", "aapos", "aaalt"])
        gmn["id38"] = ("chr" + gmn.variant_id38.str.split("-").str[0] + ":g." + gmn.pos38.astype(str) + gmn.ref + ">" + gmn.alt)
        ids38 = sorted(set(ids38) | set(gmn.id38))
        ann38 = myvariant_cached(gene, ids38, "hg38")
        ann19 = myvariant_cached(gene, ids19, "hg19") if ids19 else pd.DataFrame(columns=["id"] + PNAMES)
        # map each protein change to the mean predictor values over its nucleotide variants
        m38 = pd.concat([cvn[["aaref", "aapos", "aaalt", "id38"]].rename(columns={"id38": "id"}).dropna(),
                         gmn[["aaref", "aapos", "aaalt", "id38"]].rename(columns={"id38": "id"})]).drop_duplicates()
        m19 = cvn[cvn.id38.isna()][["aaref", "aapos", "aaalt", "id19"]].rename(columns={"id19": "id"}).dropna()
        ann = pd.concat([m38.merge(ann38, on="id"), m19.merge(ann19, on="id")]) if len(m19) else m38.merge(ann38, on="id")
        pa = ann.groupby(["aaref", "aapos", "aaalt"])[PNAMES].mean().reset_index()
        allp = allp.merge(pa, on=["aaref", "aapos", "aaalt"], how="left")
        allp.to_parquet(gd / "protein_table.parquet", index=False)
        # --- accounting
        lab = allp[allp.lab_T2.isin(["P", "B"])]
        row["seq_match_frac"] = float(lab.seq_ok.mean())
        row["pred_avail_frac"] = float(min(lab.alphamissense.notna().mean(), lab.revel.notna().mean()))
        nP = int((allp.lab_T2 == "P").sum()); nB = int((allp.lab_T2 == "B").sum())
        row.update(n_P_T2=nP, n_B_T2=nB, I1=bool(nP >= MIN_P and nB >= MIN_B), I2=True,
                   I3=bool(row["seq_match_frac"] >= MIN_SEQ_MATCH), I4=bool(row["pred_avail_frac"] >= MIN_PRED_AVAIL))
        fails = [n for n, ok in (("I1", row["I1"]), ("I3", row["I3"]), ("I4", row["I4"])) if not ok]
        row["included"] = not fails; row["failed"] = "; ".join(fails)
        inc_rows.append(row)
        # label accounting (duplicates, discordance)
        n_prot_cv = int(allp.in_clinvar.sum())
        acct_rows.append({"gene": gene, "clinvar_missense_nt_records": int(len(craw)), "n_clinvar_protein_changes": n_prot_cv,
                          "n_nt_collapsed_duplicates": int(len(craw) - n_prot_cv),
                          "n_protein_changes_with_multiple_nt": int((allp.n_nt > 1).sum()),
                          "n_protein_changes_P_and_B_discordant": int(((allp.nP > 0) & (allp.nB > 0)).sum()),
                          "n_protein_changes_B_with_VUS_record": int(((allp.nB > 0) & (allp.nV > 0) & (allp.nP == 0)).sum()),
                          "n_protein_changes_P_with_VUS_record": int(((allp.nP > 0) & (allp.nV > 0) & (allp.nB == 0)).sum()),
                          "seq_mismatch_labelled": int((~lab.seq_ok).sum()),
                          "nt_B_records": int((craw.y == "B").sum()), "nt_P_records": int((craw.y == "P").sum())})
        for t in TIERS:
            col = f"lab_{t}"
            if gene != "MYH7" and t.startswith("T3"):
                continue
            L = allp[allp[col].isin(["P", "B"]) & allp.seq_ok]
            tier_rows.append({"gene": gene, "tier": t, "n_P": int((L[col] == "P").sum()), "n_B": int((L[col] == "B").sum()),
                              "P_residues": int(L[L[col] == "P"].aapos.nunique()), "B_residues": int(L[L[col] == "B"].aapos.nunique()),
                              "n_VUS": int((allp[col] == "VUS").sum())})
        print(f"[labels] {gene}: P={nP} B={nB} included={row['included']} {row['failed']} ({time.time()-t0:.0f}s)", flush=True)
    def _merge(new, name, key=("gene",)):
        f = TAB / name
        new = pd.DataFrame(new)
        if f.exists():
            old = pd.read_csv(f)
            done = set(map(tuple, new[list(key)].values))
            old = old[[tuple(r) not in done for r in old[list(key)].values]]
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(f, index=False)
    _merge(inc_rows, "v7_inclusion.csv"); _merge(acct_rows, "v7_label_accounting.csv"); _merge(tier_rows, "v7_tier_counts.csv", ("gene", "tier"))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
