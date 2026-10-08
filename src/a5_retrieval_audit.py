"""v7 analysis 4(a) / roadmap 2.1: retrieval-failure prevalence audit across genes.

For each gene: ClinVar missense SNVs from the complete variant_summary release (denominator) versus the
ClinVar VariationIDs returned by the MyVariant.info query  clinvar.gene.symbol:<GENE>  (all pages, scroll).
Each missed variant is classified by a follow-up POST (hg19 id):
  symbol_field        MyVariant holds the ClinVar record but under a different gene symbol (or none)
  snapshot_missing    MyVariant has no ClinVar section / no record for the variant (snapshot staleness);
                      flagged newer_than_snapshot if VariationID > max VariationID retrieved for the gene
  no_hg19             ClinVar has no GRCh37 coordinate (cannot be tested by hg19 id)
  other               anything else (should be ~0)
Audit sets: A = the 19 panel genes; A2 = the 100 genes with most ClinVar missense SNVs (exploratory).
Outputs: results/tables/v7_retrieval_audit_gene.csv, v7_retrieval_audit_bins.csv, v7_retrieval_audit_summary.json
Run: python -m src.a5_retrieval_audit
"""
from __future__ import annotations
import datetime as dt, json, time
import numpy as np
import pandas as pd
import requests
from scipy.stats import binomtest
from src.genes import *   # noqa

AUD = RAW / "audit"; AUD.mkdir(exist_ok=True)
FIELDS = "clinvar.variant_id,clinvar.gene.symbol"


def _req(method, url, **kw):
    for a in range(6):
        try:
            r = requests.request(method, url, timeout=300, **kw)
            if r.status_code == 200:
                return r
            if r.status_code in (400, 404):
                return r
        except Exception as e:  # noqa
            pass
        time.sleep(4 * (a + 1))
    raise RuntimeError(f"failed {url}")


def retrieve_gene(gene):
    """All records returned by the symbol query: VariationID and stored symbol(s)."""
    f = AUD / f"{gene}_retrieved.parquet"
    if f.exists():
        return pd.read_parquet(f)
    rows = []
    r = _req("GET", "https://myvariant.info/v1/query",
             params={"q": f"clinvar.gene.symbol:{gene}", "fields": FIELDS, "size": 1000, "fetch_all": "true"}).json()
    total = r.get("total")
    while True:
        for h in r.get("hits", []):
            cv = h.get("clinvar") or {}
            sym = (cv.get("gene") or {}).get("symbol")
            rows.append({"VariationID": cv.get("variant_id") if not isinstance(cv.get("variant_id"), list) else cv["variant_id"][0],
                         "stored_symbol": str(sym), "_id": h["_id"]})
        sid = r.get("_scroll_id")
        if not r.get("hits") or not sid:
            break
        r = _req("GET", "https://myvariant.info/v1/query", params={"scroll_id": sid}).json()
        if "hits" not in r:
            break
    d = pd.DataFrame(rows, columns=["VariationID", "stored_symbol", "_id"])
    d.attrs["total"] = total
    d.to_parquet(f, index=False)
    (AUD / f"{gene}_total.json").write_text(json.dumps({"total_reported": total, "n_fetched": len(d)}))
    return d


def post_missed(gene, ids):
    f = AUD / f"{gene}_missed_post.parquet"
    if f.exists():
        return pd.read_parquet(f)
    out = []
    ids = list(ids)
    for i in range(0, len(ids), 1000):
        part = ids[i:i + 1000]
        r = _req("POST", "https://myvariant.info/v1/variant",
                 data={"ids": ",".join(part), "fields": FIELDS, "assembly": "hg19"})
        for h in r.json():
            if h.get("notfound"):
                out.append({"id19": h["query"], "found": False, "has_clinvar": False, "mv_variation_id": None, "mv_symbol": None})
            else:
                cv = h.get("clinvar") or {}
                vid = cv.get("variant_id"); vid = vid[0] if isinstance(vid, list) and vid else vid
                sym = (cv.get("gene") or {}).get("symbol")
                out.append({"id19": h["query"], "found": True, "has_clinvar": bool(cv), "mv_variation_id": vid,
                            "mv_symbol": None if sym is None else str(sym)})
    d = pd.DataFrame(out, columns=["id19", "found", "has_clinvar", "mv_variation_id", "mv_symbol"])
    d.to_parquet(f, index=False)
    return d


def audit_gene(cv, gene, nbins=10):
    g = cv[cv.gene == gene].copy()
    ret = retrieve_gene(gene)
    got = set(pd.to_numeric(ret.VariationID, errors="coerce").dropna().astype(int))
    g["retrieved"] = g.VariationID.isin(got)
    ceiling = max(got) if got else 0
    miss = g[~g.retrieved & g.has_hg19]
    pr = post_missed(gene, miss.id19.unique()) if len(miss) else pd.DataFrame(columns=["id19", "found", "has_clinvar", "mv_variation_id", "mv_symbol"])
    pr = pr.drop_duplicates("id19").set_index("id19")
    cat = pd.Series("other", index=g.index, dtype=object)
    cat[g.retrieved] = "retrieved"
    cat[~g.retrieved & ~g.has_hg19] = "no_hg19"
    for idx, row in miss.iterrows():
        p = pr.loc[row.id19] if row.id19 in pr.index else None
        if p is None or not p["found"]:
            cat[idx] = "snapshot_missing"
        elif not p["has_clinvar"]:
            cat[idx] = "snapshot_missing"
        elif p["mv_symbol"] != gene:
            cat[idx] = "symbol_field"
        else:
            cat[idx] = "other"
    g["category"] = cat
    g["newer_than_snapshot"] = (g.category == "snapshot_missing") & (g.VariationID > ceiling)
    n = len(g); nm = int((~g.retrieved).sum())
    mx = int(g.aapos.max())
    edges = np.linspace(0, mx, nbins + 1)
    g["bin"] = np.clip(np.digitize(g.aapos, edges[1:-1]), 0, nbins - 1)
    bt = g.groupby("bin").agg(n=("retrieved", "size"), n_lost=("retrieved", lambda s: int((~s).sum())))
    bt["loss_pct"] = 100 * bt.n_lost / bt.n
    bt["gene"] = gene; bt["res_from"] = [int(edges[i]) + 1 for i in bt.index]; bt["res_to"] = [int(edges[i + 1]) for i in bt.index]
    # longest run of consecutive bins with >=90% loss and >=5 variants
    run = best = 0; best_range = None; start = None
    for b in range(nbins):
        if b in bt.index and bt.loc[b, "n"] >= 5 and bt.loc[b, "loss_pct"] >= 90:
            run += 1; start = start if run > 1 else b
            if run > best:
                best = run; best_range = (int(edges[start]) + 1, int(edges[b + 1]))
        else:
            run = 0
    rowsum = {"gene": gene, "n_clinvar_missense": n, "n_returned": int(g.retrieved.sum()), "n_missed": nm,
              "loss_pct": 100 * nm / n if n else np.nan,
              "n_symbol_field": int((g.category == "symbol_field").sum()),
              "n_snapshot_missing": int((g.category == "snapshot_missing").sum()),
              "n_snapshot_missing_newer": int(g.newer_than_snapshot.sum()),
              "n_no_hg19": int((g.category == "no_hg19").sum()), "n_other": int((g.category == "other").sum()),
              "n_records_returned_by_query_all_types": int(len(ret)),
              "retrieved_ceiling_variation_id": int(ceiling),
              "n_P_missed": int(((g.y == "P") & ~g.retrieved).sum()), "n_B_missed": int(((g.y == "B") & ~g.retrieved).sum()),
              "n_P": int((g.y == "P").sum()), "n_B": int((g.y == "B").sum()),
              "bins_ge90pct_loss_run": best, "longest_run_residues": None if best_range is None else f"{best_range[0]}-{best_range[1]}",
              "max_bin_loss_pct": float(bt.loss_pct.max()), "protein_extent_max_aapos": mx,
              "top_stored_symbols_of_missed": json.dumps(
                  pr.reindex(g.loc[g.category == "symbol_field", "id19"]).mv_symbol.value_counts().head(4).to_dict())}
    return rowsum, bt.reset_index(drop=True)


def main():
    t0 = time.time()
    cv = pd.read_parquet(INTERIM / "clinvar_missense_snv.parquet")
    counts = cv.gene.value_counts()
    top100 = list(counts.head(100).index)
    sets = {"A": GENES, "A2": [g for g in top100]}
    allg = list(dict.fromkeys(GENES + top100))
    rows, bins = [], []
    for i, gene in enumerate(allg):
        t = time.time()
        r, b = audit_gene(cv, gene)
        r["in_panel_A"] = gene in GENES; r["in_top100_A2"] = gene in top100
        rows.append(r); bins.append(b)
        print(f"[audit {i+1}/{len(allg)}] {gene}: n={r['n_clinvar_missense']} lost={r['loss_pct']:.1f}% "
              f"(symbol {r['n_symbol_field']}, snapshot {r['n_snapshot_missing']}, nohg19 {r['n_no_hg19']}, other {r['n_other']}) {time.time()-t:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(TAB / "v7_retrieval_audit_gene.csv", index=False)
    res = pd.DataFrame(rows)
    pd.concat(bins).to_csv(TAB / "v7_retrieval_audit_bins.csv", index=False)
    summ = {"myvariant_build": requests.get("https://myvariant.info/v1/metadata", timeout=60).json().get("build_date"),
            "audit_run_utc": dt.datetime.utcnow().isoformat() + "Z"}
    for name, gl in (("A_panel19", GENES), ("A2_top100", top100)):
        d = res[res.gene.isin(gl)]
        d = d[d.n_clinvar_missense >= 20]
        k5 = int((d.loss_pct > 5).sum()); k20 = int((d.loss_pct > 20).sum()); n = len(d)
        ci5 = binomtest(k5, n).proportion_ci(method="exact")
        summ[name] = {"n_genes": n, "n_loss_gt5pct": k5, "n_loss_gt20pct": k20,
                      "prop_gt5pct": k5 / n, "prop_gt5pct_exact_ci": [ci5.low, ci5.high],
                      "median_loss_pct": float(d.loss_pct.median()), "max_loss_pct": float(d.loss_pct.max()),
                      "total_missense": int(d.n_clinvar_missense.sum()), "total_missed": int(d.n_missed.sum()),
                      "pooled_loss_pct": 100 * float(d.n_missed.sum() / d.n_clinvar_missense.sum()),
                      "causes_of_missed": {c: int(d[c].sum()) for c in ("n_symbol_field", "n_snapshot_missing", "n_snapshot_missing_newer", "n_no_hg19", "n_other")},
                      "genes_with_90pct_loss_run": d[d.bins_ge90pct_loss_run > 0].gene.tolist()}
    (TAB / "v7_retrieval_audit_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1)); print(f"audit done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
