# Gold Standard Data Notes

## syllabified_stems_for_review_corrections.csv

This file contains 71 syllabification corrections made by the annotator during Phase 1
validation. As of the authoritative dataset (tash_nouns.csv, updated 2024-12-27),
**64 of 71 corrections are incorporated**.

### 7 unreconciled corrections

The following 7 records show a mismatch between the annotator's corrected syllabification
and the current value in tash_nouns.csv. These discrepancies are almost certainly due to
the singular theme itself being corrected during the December 27, 2024 data cleaning
(Phase 4), which produced a different syllabification from the same rule-based algorithm.
The annotator worked on an earlier version of the themes.

| recordID | analysisSingularTheme | Annotator correction | Current (tash_nouns.csv) |
|----------|-----------------------|----------------------|--------------------------|
| 55 | aʒʒgal | aʒʒ.gal | aʒ.ʒ.gal |
| 412 | wwullu | w.wu.lu | w.wul.lu |
| 1992 | ukksa | ukk.sa | uk.k.sa |
| 2343 | !sskkwar | s.sk.kar | ss.kk.war |
| 2351 | !akkʷzin | akk.zin | ak.kz.in |
| 2520 | lmiziria | l.mi.zir.ja | l.mi.zir.ia |
| 2699 | ʕli!iʒʒan | ʕ.liʒ.ʒan | ʕli.iʒ.ʒan |

**Action needed**: Verify these 7 forms manually. For each, check whether the current
`analysisSingularTheme` in tash_nouns.csv differs from what the annotator saw (which
would explain the discrepancy), and whether the current syllabification is correct
given the updated theme.

Records 2343 and 2520 are confirmed theme changes from the Dec 27 cleaning log
(`!sskkar → !sskkwar` and `lmizirja → lmiziria`), which accounts for those two.
The remaining 5 should be inspected.

## LH validation files

- `tash_nouns_check_LH_withCorrections.xlsx` — 72-form annotator validation of
  Light/Heavy patterns. Contains "Confirm LH Pattern" and "Notes" columns.
  Source of the 97.3% accuracy figure reported for the LH extraction pipeline.

- `tash_nouns_check_LH2_corrected.xlsx` — 75-form second validation set with
  "Confirm LH" column. Annotator corrections for initial geminate handling
  (key finding: initial geminates are Heavy, not Light).

## Geminate corrections

- `noninitial_geminates_11corrections.xlsx` — 11 annotator corrections to
  non-initial geminate syllabification. All 11 are incorporated into tash_nouns.csv
  (one apparent mismatch for record 2623 is a whitespace encoding artifact, not a
  real disagreement).
