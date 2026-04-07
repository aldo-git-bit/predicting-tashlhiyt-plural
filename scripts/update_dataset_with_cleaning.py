#!/usr/bin/env python3
"""
Master script to update tash_nouns.csv with cleaned data and regenerate all derived features.

This script automates the complete workflow:
1. Backup current dataset
2. Load cleaned data (provided by user)
3. Remove old derived columns
4. Regenerate all derived features in correct order
5. Validate the results

Usage:
    python update_dataset_with_cleaning.py <path_to_cleaned_data.csv>

Example:
    python update_dataset_with_cleaning.py ../cleaned_data/tash_nouns_cleaned.csv
"""

import pandas as pd
import sys
import subprocess
from datetime import datetime
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}")
    print(f"Running: {cmd}\n")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ ERROR: Command failed")
        print(f"STDERR: {result.stderr}")
        return False
    else:
        print(result.stdout)
        print(f"✅ SUCCESS: {description}")
        return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python update_dataset_with_cleaning.py <path_to_cleaned_data.csv>")
        print("\nExample:")
        print("  python update_dataset_with_cleaning.py ../cleaned_data/tash_nouns_cleaned.csv")
        sys.exit(1)

    cleaned_data_path = sys.argv[1]

    if not Path(cleaned_data_path).exists():
        print(f"❌ ERROR: File not found: {cleaned_data_path}")
        sys.exit(1)

    print("=" * 70)
    print("DATASET UPDATE WITH CLEANED DATA")
    print("=" * 70)
    print(f"\nCleaned data source: {cleaned_data_path}")
    print(f"Target: data/tash_nouns.csv")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Confirm with user
    response = input("This will replace data/tash_nouns.csv. Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted by user.")
        sys.exit(0)

    # Step 1: Create backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f'data/tash_nouns_backup_pre_cleaning_{timestamp}.csv'

    print(f"\n{'='*70}")
    print("STEP 1: Creating backup")
    print(f"{'='*70}")

    df_current = pd.read_csv('data/tash_nouns.csv')
    df_current.to_csv(backup_path, index=False)

    print(f"✅ Backup created: {backup_path}")
    print(f"   Rows: {len(df_current)}")
    print(f"   Columns: {len(df_current.columns)}")

    # Step 2: Load cleaned data
    print(f"\n{'='*70}")
    print("STEP 2: Loading cleaned data")
    print(f"{'='*70}")

    df_cleaned = pd.read_csv(cleaned_data_path)
    print(f"Cleaned data loaded:")
    print(f"   Rows: {len(df_cleaned)}")
    print(f"   Columns: {len(df_cleaned.columns)}")

    # Step 3: Remove derived columns from cleaned data
    print(f"\n{'='*70}")
    print("STEP 3: Removing old derived columns")
    print(f"{'='*70}")

    derived_cols = ['p_stem_sing_syllabified', 'p_stem_sing_LH', 'p_stem_sing_foot']
    removed = []

    for col in derived_cols:
        if col in df_cleaned.columns:
            df_cleaned = df_cleaned.drop(columns=[col])
            removed.append(col)
            print(f"  Removed: {col}")

    if removed:
        print(f"✅ Removed {len(removed)} derived columns")
    else:
        print("  No derived columns found (already clean)")

    # Save cleaned data as new tash_nouns.csv
    df_cleaned.to_csv('data/tash_nouns.csv', index=False)
    print(f"\n✅ Updated data/tash_nouns.csv")
    print(f"   Rows: {len(df_cleaned)}")
    print(f"   Columns: {len(df_cleaned.columns)}")

    # Step 4: Regenerate syllabifications
    if not run_command(
        "cd scripts && python regenerate_syllabified_column.py",
        "Regenerate syllabifications"
    ):
        print("\n❌ FAILED at syllabification step. Aborting.")
        sys.exit(1)

    # Step 5: Regenerate LH patterns
    if not run_command(
        "cd scripts && python add_lh_column.py",
        "Regenerate LH patterns"
    ):
        print("\n❌ FAILED at LH pattern step. Aborting.")
        sys.exit(1)

    # Step 6: Regenerate foot structures
    if not run_command(
        "cd scripts && python add_foot_column.py",
        "Regenerate foot structures"
    ):
        print("\n❌ FAILED at foot structure step. Aborting.")
        sys.exit(1)

    # Step 7: Validate on gold standard
    print(f"\n{'='*70}")
    print("STEP 7: Validating against gold standard")
    print(f"{'='*70}")

    result = subprocess.run(
        "cd scripts && python evaluate_rule_based.py",
        shell=True,
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        print("⚠️  WARNING: Gold standard evaluation had issues")
    else:
        print("✅ Gold standard evaluation complete")

    # Step 8: Run validation report
    print(f"\n{'='*70}")
    print("STEP 8: Generating validation report")
    print(f"{'='*70}")

    result = subprocess.run(
        f"cd scripts && python validate_data_update.py ../{backup_path}",
        shell=True,
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        print("⚠️  WARNING: Validation had issues")

    # Final summary
    print(f"\n{'='*70}")
    print("UPDATE COMPLETE")
    print(f"{'='*70}")

    df_final = pd.read_csv('data/tash_nouns.csv')

    print(f"\nFinal dataset:")
    print(f"  Rows: {len(df_final)}")
    print(f"  Columns: {len(df_final.columns)}")

    # Check derived columns exist
    for col in derived_cols:
        if col in df_final.columns:
            non_empty = (df_final[col].notna() & (df_final[col] != '')).sum()
            print(f"  {col}: {non_empty} forms")

    print(f"\nBackup saved at: {backup_path}")
    print(f"Change report: reports/data_cleaning_changes_report.csv")

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Review the validation report above")
    print("2. Check reports/data_cleaning_changes_report.csv for all changes")
    print("3. Verify gold standard accuracy is acceptable")
    print("4. If issues found, restore from backup and investigate")
    print(f"   Backup location: {backup_path}")


if __name__ == '__main__':
    main()
