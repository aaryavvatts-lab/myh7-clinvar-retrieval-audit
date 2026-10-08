2026-10-07T03:29:40Z v7 created: rsync of v6 (excluding paper/, docs/, v6 run logs); AF pdb + friedman copied; legacy v5 snapshot frozen in data/legacy_v5 (SHA256SUMS)
2026-10-07T03:29:40Z data inventory before PREREG (no cross-gene results looked at): ClinGen erepo full TSV downloaded (6342 rows; MYH7=171 rows, all Cardiomyopathy VCEP); AlphaFold DB API availability for 19 candidate UniProts checked (FBN1 P35555 has NO entry); Kelly 2018 BS1/BA1 text verified via EuropePMC PMC5876064
2026-10-07T03:30:45Z PREREG.md written and hashed (first deliverable); execution begins
2026-10-07T03:38:09Z Amendment A1 appended to PREREG (6/19 genes pass I1-I4; pool extended to top-100 ClinVar-missense genes with same rule; before any performance result)
2026-10-07T03:41:53Z acquire (AF+gnomAD) done for 19 panel + 4 extra (TSC2 SCN1A CACNA1A ABCA4); labels/tiers done; 10 genes pass I1-I4 (MYH7 LDLR MLH1 MSH2 BRCA1 TP53 TSC2 SCN1A CACNA1A ABCA4); retrieval audit running in bg
2026-10-07T03:54:59Z a3 done; a1 re-running with union contrasts; PREREG deviations D1-D5 appended (D2 sets MDDx threshold before verdicts)
2026-10-07T04:06:57Z starting full end-to-end bash run_all.sh (timing run)
2026-10-07T04:34:46Z full run_all.sh finished OK: 1657 s wall clock (logs/run_all_timings.json)
2026-10-07 (Round 2 erratum) PREREG.md (hashed, not edited) and the line above say 6342 ClinGen erepo rows; the archived file data/raw/erepo_classifications_all.tsv has 6343 data rows. MYH7=171 rows is unchanged. Times in this file are UTC; local calendar date of the work was 6 October (CDT = UTC-5).
