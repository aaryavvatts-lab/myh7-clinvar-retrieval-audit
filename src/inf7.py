"""v7 inference library (PREREG section 4): nested CV with repeated residue-grouped splits, simulation power under the
observed tie structure, small-cluster inference for the coverage association (exact, Haldane, wild cluster score
bootstrap, CR2 + Satterthwaite), random-effects meta-analysis.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.eng import *   # noqa

PANEL = PRED_ALL + ["coordination", "plddt"]
MODELS = {"M0_panel": [], "M1_count": ["n_nb", "cov3"], "M2_S1": ["S1"], "M3_count_S1": ["n_nb", "cov3", "S1"]}
CONTRASTS = [("M1_count", "M0_panel", "count over panel"), ("M2_S1", "M0_panel", "S1 over panel"),
             ("M3_count_S1", "M1_count", "S1 beyond count (primary nested)"), ("M3_count_S1", "M2_S1", "count beyond S1")]
DELTA = 0.02


def _lr():
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0))])


def _design(ds, geom, train_idx, model, panel_cols):
    tr = ds.iloc[train_idx]
    pm, bm = geom.masks(tr.loc[tr.y == 1, "aapos"], tr.loc[tr.y == 0, "aapos"])
    sc = derive(geom.dens(ds.aapos.values, pm, bm), pm.sum(), bm.sum())
    X = ds[panel_cols].copy()
    extra = MODELS[model]
    if "n_nb" in extra:
        X["n_nb"] = sc["n_nb"]; X["cov3"] = (sc["n_nb"] >= 3).astype(float)
    if "S1" in extra:
        X["S1"] = sc["S1"]
    return X


def nested_cv(ds, geom, repeats=20, seed0=SEED, group="residue", models=tuple(MODELS)):
    """Repeated residue- (or 100-residue-block-) grouped 5-fold CV; structure recomputed from training labels per fold.
    Returns dict model -> array (repeats x n) of OOF probabilities."""
    y = ds.y.values
    grp = ds.aapos.values if group == "residue" else (ds.aapos.values // 100)
    panel_cols = [c for c in PANEL if c in ds.columns]
    oof = {m: np.zeros((repeats, len(ds))) for m in models}
    for r in range(repeats):
        skf = StratifiedGroupKFold(5, shuffle=True, random_state=seed0 + r)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for tr, te in skf.split(ds, y, grp):
                for m in models:
                    X = _design(ds, geom, tr, m, panel_cols)
                    oof[m][r, te] = _lr().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    return oof


def cv_summary(ds, oof, n_boot=N_BOOT, seed=SEED, tag=""):
    y = ds.y.values; g = ds.aapos.values
    rows = []
    for m, arr in oof.items():
        a = np.array([auroc(y, arr[r]) for r in range(arr.shape[0])])
        rows.append({"tag": tag, "model": m, "auroc_mean": r3(a.mean()), "auroc_sd_across_repeats": r3(a.std(ddof=1), 4), "repeats": arr.shape[0]})
    drows = []
    for A, B, q in CONTRASTS:
        if A not in oof or B not in oof:
            continue
        pt = np.array([auroc(y, oof[A][r]) - auroc(y, oof[B][r]) for r in range(oof[A].shape[0])])
        reps = []
        for idx in boot_draws(g, y, n_boot, seed):
            yy = y[idx]
            reps.append(np.mean([auroc(yy, oof[A][r][idx]) - auroc(yy, oof[B][r][idx]) for r in range(oof[A].shape[0])]))
        reps = np.array(reps)
        drows.append({"tag": tag, "A": A, "B": B, "question": q, "delta_mean": r3(pt.mean()), "delta_sd_across_repeats": r3(pt.std(ddof=1), 4),
                      "delta_min": r3(pt.min()), "delta_max": r3(pt.max()), "n_repeats_positive": int((pt > 0).sum()),
                      "ci_low": r3(np.percentile(reps, 2.5)), "ci_high": r3(np.percentile(reps, 97.5)),
                      "boot_sd": r3(reps.std(), 4), "mdd_80": r3(2.8 * reps.std()),
                      "p_two_sided_boot": diff_p(reps), "p_tost_0.02": tost_p(reps, DELTA),
                      "equivalence_at_0.02": bool(tost_p(reps, DELTA) < 0.05), "n_boot": len(reps)})
    return rows, drows


# ------------------------------------------------------------------ power under the observed tie structure
def _auc_from_counts(cP, cB):
    """AUROC (ties 0.5) from category count vectors (last axis = ordered distinct score values)."""
    cumB = np.cumsum(cB, axis=-1) - cB
    num = (cP * (cumB + 0.5 * cB)).sum(axis=-1)
    return num / (cP.sum(axis=-1) * cB.sum(axis=-1))


def _mixture(p_scores, pool_all, q, rng, size):
    """Benign draws: prob 1-q from the pathogenic empirical distribution, prob q from the lower tail pool."""
    from_low = rng.random(size) < q
    draw = rng.choice(p_scores, size)
    if from_low.any():
        draw[from_low] = rng.choice(pool_all, int(from_low.sum()))
    return draw


def lower_pool(P, B):
    allv = np.concatenate([P, B])
    vals, cnt = np.unique(P, return_counts=True)
    mode = vals[np.argmax(cnt)]
    pool = allv[(allv <= np.quantile(allv, 0.2)) & (allv != mode)]
    if len(pool) < 3:
        pool = allv[allv < mode]
    if len(pool) < 3:
        pool = np.array([allv.min() - 1e-3] * 3)
    return pool


def power_curve(P, nB, pool, targets=(0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95), nsim=2000, M=999, seed=SEED, nP=None):
    """Power of the one-sided permutation (Mann-Whitney) test at 5% when pathogenic scores follow the observed
    pathogenic empirical distribution and benign scores follow the mixture; ties preserved. Returns rows."""
    rng = np.random.default_rng(seed)
    nP = nP or len(P)
    if len(np.unique(np.concatenate([P, pool]))) > 40:
        # continuous score: discretise to 30 pooled-quantile bins (keeps order; makes the multivariate hypergeometric null cheap)
        edges = np.unique(np.quantile(np.concatenate([P, pool]), np.linspace(0, 1, 31))[1:-1])
        P = np.digitize(P, edges).astype(float); pool = np.digitize(pool, edges).astype(float)
    vals = np.unique(np.concatenate([P, pool]))
    Pc = np.array([(P == v).sum() for v in vals], float); Pp = Pc / Pc.sum()
    # true AUROC as a function of q by large-sample Monte Carlo
    qs = np.linspace(0, 1, 41)
    big = 200000
    xs = rng.choice(P, big)
    tq = []
    for q in qs:
        b = _mixture(P, pool, q, rng, big)
        o = np.argsort(b); bs = b[o]
        lt = np.searchsorted(bs, xs, "left"); le = np.searchsorted(bs, xs, "right")
        tq.append(((lt + 0.5 * (le - lt)) / big).mean())
    tq = np.array(tq)
    rows = []
    for tgt in targets:
        if tgt < tq.min() or tgt > tq.max():
            rows.append({"true_auroc": tgt, "q": None, "power": None, "note": "target outside reachable AUROC range of the mixture"}); continue
        q = float(np.interp(tgt, tq, qs))
        hit = 0
        for _ in range(nsim):
            xP = rng.choice(P, nP)
            xB = _mixture(P, pool, q, rng, nB)
            allx = np.concatenate([xP, xB])
            idx = np.searchsorted(vals, allx)
            tot = np.bincount(idx, minlength=len(vals)).astype(float)
            cB = np.bincount(idx[nP:], minlength=len(vals)).astype(float); cP = tot - cB
            obs = _auc_from_counts(cP, cB)
            null_cB = rng.multivariate_hypergeometric(tot.astype(int), nB, size=M).astype(float)
            nu = _auc_from_counts(tot - null_cB, null_cB)
            p = (np.sum(nu >= obs - 1e-12) + 1) / (M + 1)
            hit += p < 0.05
        rows.append({"true_auroc": tgt, "q": round(q, 3), "power": hit / nsim, "note": ""})
    return rows


def min_detectable(rows):
    pts = [(r["true_auroc"], r["power"]) for r in rows if r["power"] is not None]
    for (a0, p0), (a1, p1) in zip(pts[:-1], pts[1:]):
        if p0 < 0.8 <= p1:
            return a0 + (0.8 - p0) * (a1 - a0) / (p1 - p0 + 1e-12)
    return pts[0][0] if pts and pts[0][1] >= 0.8 else np.nan


# ------------------------------------------------------------------ coverage association: small-cluster inference
def cr2_ols(X, y, groups, c_idx=1):
    """CR2 cluster-robust SE with Satterthwaite df (Bell-McCaffrey / Pustejovsky-Tipton) for coefficient c_idx."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, p = X.shape
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    H = X @ bread @ X.T
    e = y - X @ beta
    uniq, inv = np.unique(groups, return_inverse=True)
    G = len(uniq)
    Kmat = np.zeros((n, G))
    for g in range(G):
        idx = np.where(inv == g)[0]
        Hg = H[np.ix_(idx, idx)]
        w, V = np.linalg.eigh(np.eye(len(idx)) - Hg)
        w = np.where(w > 1e-10, w, np.nan)
        A = (V * (1 / np.sqrt(w))) @ V.T if not np.isnan(w).any() else np.eye(len(idx))
        m = (X[idx] @ bread[:, c_idx])            # vector length n_g
        Kmat[idx, g] = A.T @ m if True else m
    B = Kmat @ Kmat.T
    var = float(e @ B @ e)
    I_H = np.eye(n) - H
    Mx = I_H @ B @ I_H
    trM = np.trace(Mx); trM2 = np.sum(Mx * Mx.T)
    df = float(trM ** 2 / trM2)
    se = np.sqrt(var)
    tcrit = stats.t.ppf(0.975, df)
    return {"coef": float(beta[c_idx]), "se_cr2": float(se), "df_satterthwaite": df,
            "ci_low": float(beta[c_idx] - tcrit * se), "ci_high": float(beta[c_idx] + tcrit * se),
            "p": float(2 * stats.t.sf(abs(beta[c_idx] / se), df))}


def wild_cluster_score_p(x, y, groups, nboot=9999, seed=SEED):
    """Wild cluster score bootstrap (Rademacher, null imposed) for the logistic coefficient on x (intercept-only restricted model)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    xt = x - x.mean(); e = y - y.mean()
    uniq, inv = np.unique(groups, return_inverse=True)
    u = np.bincount(inv, weights=xt * e)
    tobs = u.sum() / np.sqrt((u ** 2).sum())
    rng = np.random.default_rng(seed)
    W = rng.choice([-1.0, 1.0], size=(nboot, len(u)))
    tb = (W @ u) / np.sqrt((u ** 2).sum())
    return float((np.sum(np.abs(tb) >= abs(tobs) - 1e-12) + 1) / (nboot + 1)), float(tobs), len(u)


def coverage_inference(ds, k):
    f = (ds.n_nb >= k).astype(float).values; y = ds.y.values; g = ds.aapos.values
    a = int(((y == 1) & (f == 1)).sum()); b = int(((y == 1) & (f == 0)).sum())
    c = int(((y == 0) & (f == 1)).sum()); d = int(((y == 0) & (f == 0)).sum())
    out = {"min_neighbours": k, "P_covered": a, "P_uncovered": b, "B_covered": c, "B_uncovered": d}
    orr = stats.contingency.odds_ratio([[a, b], [c, d]])
    ci = orr.confidence_interval(0.95)
    out.update(or_conditional_mle=float(orr.statistic), or_exact_ci_low=float(ci.low), or_exact_ci_high=float(ci.high),
               fisher_p=float(stats.fisher_exact([[a, b], [c, d]])[1]))
    ha, hb, hc, hd = a + .5, b + .5, c + .5, d + .5
    orh = ha * hd / (hb * hc); se = np.sqrt(1 / ha + 1 / hb + 1 / hc + 1 / hd)
    out.update(or_haldane=float(orh), or_haldane_ci_low=float(np.exp(np.log(orh) - 1.96 * se)), or_haldane_ci_high=float(np.exp(np.log(orh) + 1.96 * se)))
    pw, tobs, ncl = wild_cluster_score_p(f, y, g)
    out.update(wild_cluster_p=pw, wild_cluster_t=tobs, n_clusters=ncl, n_benign_clusters=int(ds[ds.y == 0].aapos.nunique()))
    if 0 < f.sum() < len(f):
        try:
            cr = cr2_ols(np.column_stack([np.ones(len(f)), f]), y, g)
            out.update({f"lpm_{k_}": v for k_, v in cr.items()})
        except Exception as e:  # noqa
            out["lpm_error"] = str(e)[:80]
    return out


# ------------------------------------------------------------------ meta-analysis
def random_effects(est, se):
    """DerSimonian-Laird tau2 with Hartung-Knapp CI and prediction interval; Q, I2."""
    est = np.asarray(est, float); se = np.asarray(se, float)
    k = len(est); w = 1 / se ** 2
    fe = (w * est).sum() / w.sum()
    Q = float((w * (est - fe) ** 2).sum()); dfQ = k - 1
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - dfQ) / C) if k > 1 else 0.0
    ws = 1 / (se ** 2 + tau2)
    mu = float((ws * est).sum() / ws.sum())
    se_dl = float(np.sqrt(1 / ws.sum()))
    if k > 1:
        hk = float(np.sqrt((ws * (est - mu) ** 2).sum() / ((k - 1) * ws.sum())))
        tcrit = stats.t.ppf(0.975, k - 1)
        lo, hi = mu - tcrit * hk, mu + tcrit * hk
        pi_h = np.sqrt(tau2 + hk ** 2); pl, ph = mu - stats.t.ppf(0.975, max(k - 2, 1)) * pi_h, mu + stats.t.ppf(0.975, max(k - 2, 1)) * pi_h
        p = float(2 * stats.t.sf(abs(mu / hk), k - 1))
    else:
        hk = se_dl; lo = hi = pl = ph = np.nan; p = np.nan
    return {"k": k, "pooled": mu, "ci_low_hk": float(lo), "ci_high_hk": float(hi), "p_hk": p, "se_dl": se_dl,
            "tau2": float(tau2), "tau": float(np.sqrt(tau2)), "Q": Q, "Q_df": dfQ, "Q_p": float(stats.chi2.sf(Q, dfQ)) if k > 1 else np.nan,
            "I2_pct": float(max(0.0, (Q - dfQ) / Q) * 100) if Q > 0 else 0.0, "pred_low": float(pl), "pred_high": float(ph),
            "fixed_effect": float(fe)}
