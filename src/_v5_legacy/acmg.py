"""Phase 5 — translate computational evidence into ACMG/AMP codes + Bayesian points.

Turns the raw scores into the language clinicians use, producing a provisional
classification per variant. NOT a clinical diagnosis (disclaimer travels with output).

Code mapping (implement on go-ahead, following ClinGen SVI calibration for in-silico):
    PM1  - residue in a mutational hotspot / high 3D-PPS
    PM2  - absent or ultra-rare in gnomAD
    PP3  - multiple in-silico lines damaging (incl. AlphaMissense) at calibrated strength
    BP4  - multiple in-silico lines benign
    BS1  - allele frequency greater than expected for the disorder
    BA1  - allele frequency too common to be pathogenic (stand-alone benign)

Contract:
    assign_codes(row) -> set[str]
    bayesian_points(codes) -> (points, provisional_class)   # Tavtigian et al. 2018
"""
from __future__ import annotations

DISCLAIMER = ("Provisional, in-silico, hypothesis-generating classification. "
              "NOT a clinical diagnosis; requires expert/experimental confirmation.")


def assign_codes(row) -> set:
    raise NotImplementedError("Phase 5: map evidence -> ACMG codes.")


def bayesian_points(codes: set) -> tuple:
    raise NotImplementedError("Phase 5: Tavtigian 2018 Bayesian points -> class.")
