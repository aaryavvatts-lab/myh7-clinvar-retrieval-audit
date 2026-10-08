"""Phase 2 — Axis A (sequence predictors + conservation) and Axis C (population).

Axis A via MyVariant.info (wraps dbNSFP): SIFT, PolyPhen-2 HDIV/HVAR, CADD-phred,
REVEL, AlphaMissense (score+class), phyloP/phastCons/GERP conservation.
Axis C via gnomAD v4 GraphQL: global + per-ancestry AF, AC, popmax, homozygotes,
gene-level constraint (missense z, LOEUF).

Contract:
    annotate_predictors(variants) -> DataFrame keyed by variant_id (Axis A columns)
    annotate_gnomad(variants, gene) -> DataFrame keyed by variant_id (Axis C columns)
Batch requests, cache to data/interim/, be polite to public APIs (rate-limit + retry).
"""
from __future__ import annotations
import pandas as pd


def annotate_predictors(variants: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("Phase 2: MyVariant.info batch annotation.")


def annotate_gnomad(variants: pd.DataFrame, gene: str) -> pd.DataFrame:
    raise NotImplementedError("Phase 2: gnomAD GraphQL frequencies + constraint.")
