#!/usr/bin/env python3
"""
Validate data update after cleaning and regeneration.

This script compares the old dataset (before cleaning) with the new dataset
(after cleaning and regeneration) to verify:
1. Reproducibility: unchanged inputs → identical outputs
2. Changes: document all forms where inputs changed
3. Quality: ensure derived features are internally consistent
"""

import pandas as pd
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_data_update.py <backup_file>")
        print("Example: python validate_data_update.py data/tash_nouns_backup_pre_cleaning_20251219.csv")
        sys.exit(1)

    backup_file = sys.argv[1]

    # Load datasets
    print("=" * 70)
    print("DATA UPDATE VALIDATION REPORT")
    print("=" * 70)
    print(f"\nLoading old dataset from: {backup_file}")
    df_old = pd.read_csv(backup_file)

    print(f"Loading new dataset from: data/tash_nouns.csv")
    df_new = pd.read_csv('data/tash_nouns.csv')

    print(f"\nOld dataset: {len(df_old)} rows, {len(df_old.columns)} columns")
    print(f"New dataset: {len(df_new)} rows, {len(df_new.columns)} columns")

    # Verify same number of rows
    if len(df_old) != len(df_new):
        print("\n⚠️  WARNING: Row count changed!")
        print(f"   Old: {len(df_old)} rows")
        print(f"   New: {len(df_new)} rows")

    # Ensure both have recordID for tracking
    if 'recordID' not in df_old.columns or 'recordID' not in df_new.columns:
        print("\n❌ ERROR: recordID column missing. Cannot validate.")
        sys.exit(1)

    # Align by recordID
    df_old = df_old.set_index('recordID').sort_index()
    df_new = df_new.set_index('recordID').sort_index()

    # Check 1: Reproducibility for unchanged inputs
    print("\n" + "=" * 70)
    print("CHECK 1: REPRODUCIBILITY (unchanged inputs → identical outputs)")
    print("=" * 70)

    if 'analysisSingularTheme' in df_old.columns and 'analysisSingularTheme' in df_new.columns:
        # Find forms where input didn't change
        unchanged_mask = (df_old['analysisSingularTheme'].fillna('') ==
                         df_new['analysisSingularTheme'].fillna(''))

        unchanged_count = unchanged_mask.sum()
        print(f"\nForms with UNCHANGED input: {unchanged_count}")

        # For unchanged inputs, check if syllabification is identical
        if 'p_stem_sing_syllabified' in df_old.columns and 'p_stem_sing_syllabified' in df_new.columns:
            unchanged_subset = unchanged_mask & df_old['p_stem_sing_syllabified'].notna()
            same_output = (df_old.loc[unchanged_subset, 'p_stem_sing_syllabified'] ==
                          df_new.loc[unchanged_subset, 'p_stem_sing_syllabified'])

            reproducible_count = same_output.sum()
            total_unchanged = unchanged_subset.sum()

            print(f"Syllabifications identical: {reproducible_count} / {total_unchanged}")
            print(f"Reproducibility rate: {100*reproducible_count/total_unchanged:.2f}%")

            if reproducible_count < total_unchanged:
                print(f"\n⚠️  {total_unchanged - reproducible_count} forms have different outputs despite unchanged inputs!")

                # Show examples of non-reproducible forms
                different = unchanged_subset & ~same_output
                print("\nExamples of non-reproducible forms:")
                for idx in df_old[different].head(5).index:
                    print(f"  Record {idx}:")
                    print(f"    Input: {df_old.loc[idx, 'analysisSingularTheme']}")
                    print(f"    Old: {df_old.loc[idx, 'p_stem_sing_syllabified']}")
                    print(f"    New: {df_new.loc[idx, 'p_stem_sing_syllabified']}")
            else:
                print("✅ PASS: Perfect reproducibility")

    # Check 2: Document all changes
    print("\n" + "=" * 70)
    print("CHECK 2: CHANGES IN INPUT DATA")
    print("=" * 70)

    # Track changes in analysisSingularTheme
    if 'analysisSingularTheme' in df_old.columns and 'analysisSingularTheme' in df_new.columns:
        changed_inputs = (df_old['analysisSingularTheme'].fillna('') !=
                         df_new['analysisSingularTheme'].fillna(''))

        changed_count = changed_inputs.sum()
        print(f"\nForms with CHANGED analysisSingularTheme: {changed_count}")

        if changed_count > 0:
            # Create change report
            changes_list = []

            for idx in df_old[changed_inputs].index:
                changes_list.append({
                    'recordID': idx,
                    'old_input': df_old.loc[idx, 'analysisSingularTheme'],
                    'new_input': df_new.loc[idx, 'analysisSingularTheme'],
                    'old_syllabified': df_old.loc[idx, 'p_stem_sing_syllabified'] if 'p_stem_sing_syllabified' in df_old.columns else '',
                    'new_syllabified': df_new.loc[idx, 'p_stem_sing_syllabified'] if 'p_stem_sing_syllabified' in df_new.columns else '',
                    'new_LH': df_new.loc[idx, 'p_stem_sing_LH'] if 'p_stem_sing_LH' in df_new.columns else '',
                    'new_foot': df_new.loc[idx, 'p_stem_sing_foot'] if 'p_stem_sing_foot' in df_new.columns else '',
                })

            changes_df = pd.DataFrame(changes_list)

            # Save change report
            report_path = 'reports/data_cleaning_changes_report.csv'
            changes_df.to_csv(report_path, index=False)
            print(f"✅ Change report saved to: {report_path}")

            # Show sample
            print("\nSample of changed forms (first 5):")
            for _, row in changes_df.head(5).iterrows():
                print(f"\n  Record {row['recordID']}:")
                print(f"    Old input: {row['old_input']}")
                print(f"    New input: {row['new_input']}")
                print(f"    New syllabified: {row['new_syllabified']}")

    # Check 3: Internal consistency of derived features
    print("\n" + "=" * 70)
    print("CHECK 3: INTERNAL CONSISTENCY")
    print("=" * 70)

    # Count non-empty values in derived columns
    derived_cols = ['p_stem_sing_syllabified', 'p_stem_sing_LH', 'p_stem_sing_foot']

    print("\nDerived feature coverage:")
    for col in derived_cols:
        if col in df_new.columns:
            non_empty = (df_new[col].notna() & (df_new[col] != '')).sum()
            print(f"  {col}: {non_empty} forms")

    # Check that all three derived columns have same coverage
    if all(col in df_new.columns for col in derived_cols):
        syll_filled = df_new['p_stem_sing_syllabified'].notna() & (df_new['p_stem_sing_syllabified'] != '')
        lh_filled = df_new['p_stem_sing_LH'].notna() & (df_new['p_stem_sing_LH'] != '')
        foot_filled = df_new['p_stem_sing_foot'].notna() & (df_new['p_stem_sing_foot'] != '')

        consistent = (syll_filled == lh_filled) & (lh_filled == foot_filled)

        if consistent.all():
            print("\n✅ PASS: All derived columns have consistent coverage")
        else:
            inconsistent_count = (~consistent).sum()
            print(f"\n⚠️  WARNING: {inconsistent_count} forms have inconsistent derived column coverage")

    # Check 4: Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Count how many forms have syllabification
    if 'p_stem_sing_syllabified' in df_new.columns:
        syllabified_count = (df_new['p_stem_sing_syllabified'].notna() &
                            (df_new['p_stem_sing_syllabified'] != '')).sum()
        print(f"\nTotal syllabified forms: {syllabified_count} / {len(df_new)}")

    # Count changes in analysisInternalChanges
    if 'analysisInternalChanges' in df_old.columns and 'analysisInternalChanges' in df_new.columns:
        internal_changes_changed = (df_old['analysisInternalChanges'].fillna('') !=
                                    df_new['analysisInternalChanges'].fillna(''))
        print(f"Forms with changed analysisInternalChanges: {internal_changes_changed.sum()}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
