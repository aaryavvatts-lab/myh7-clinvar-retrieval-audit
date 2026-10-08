"""MYH7 Tri-Axis VUS reclassification pipeline.

Modules:
    data_acquisition  - pull ClinVar variants (labeled + VUS)
    annotate          - Axis A (predictors/conservation) + Axis C (gnomAD)
    structure         - Axis B: AlphaFold structure + 3D Pathogenic Proximity Score
    model             - fuse axes, validate on labeled set, score VUS (RCS)
    acmg              - translate evidence -> ACMG/AMP codes (Bayesian points)
    run_pipeline      - orchestrator

Nothing here runs analysis until invoked. See PLAN.md.
"""
__version__ = "0.0.1"
