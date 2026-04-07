#!/usr/bin/env python3
"""
Add L/H pattern column to tash_nouns.csv

Reads the p_stem_sing_syllabified column and extracts Light/Heavy patterns
to create a new column p_stem_sing_LH.
"""

import pandas as pd
from rule_based_syllabifier import RuleBasedSyllabifier
from datetime import datetime

def main():
    # Initialize syllabifier
    print("Initializing syllabifier...")
    syllabifier = RuleBasedSyllabifier()

    # Load dataset
    csv_path = 'data/tash_nouns.csv'
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Total rows: {len(df)}")

    # Create backup with timestamp
    backup_path = f"data/tash_nouns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"Creating backup at {backup_path}...")
    df.to_csv(backup_path, index=False)

    # Check if p_stem_sing_LH column already exists
    if 'p_stem_sing_LH' in df.columns:
        print("Column p_stem_sing_LH already exists. It will be overwritten.")
        # Remove the existing column
        df = df.drop(columns=['p_stem_sing_LH'])

    # Extract L/H patterns
    print("\nExtracting L/H patterns...")
    lh_patterns = []
    processed = 0
    empty = 0
    errors = 0

    for idx, row in df.iterrows():
        # Get the original input (analysisSingularTheme)
        input_str = row.get('analysisSingularTheme', '')

        # Check if we have an input
        if pd.isna(input_str) or input_str == '':
            lh_patterns.append('')
            empty += 1
        else:
            try:
                # Extract L/H pattern from the input
                lh_pattern = syllabifier.get_lh_pattern(input_str)
                lh_patterns.append(lh_pattern)
                processed += 1

                # Progress indicator
                if processed % 100 == 0:
                    print(f"  Processed {processed} forms...")

            except Exception as e:
                print(f"  Error processing row {idx} (input: {input_str}): {e}")
                lh_patterns.append('')
                errors += 1

    # Add new column
    df['p_stem_sing_LH'] = lh_patterns

    # Save updated dataset
    print(f"\nSaving updated dataset to {csv_path}...")
    df.to_csv(csv_path, index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("L/H Pattern Extraction Complete!")
    print("=" * 70)
    print(f"Total rows: {len(df)}")
    print(f"L/H patterns extracted: {processed}")
    print(f"Empty inputs (skipped): {empty}")
    print(f"Errors: {errors}")
    print(f"\nNew column 'p_stem_sing_LH' added to {csv_path}")
    print(f"Backup saved at {backup_path}")

    # Show some examples
    print("\nSample L/H patterns:")
    print("-" * 70)
    sample_df = df[df['p_stem_sing_LH'] != ''].head(10)
    for _, row in sample_df.iterrows():
        input_str = row.get('analysisSingularTheme', '')
        syllabified = row.get('p_stem_sing_syllabified', '')
        lh_pattern = row.get('p_stem_sing_LH', '')
        print(f"  {input_str:15s} → {syllabified:15s} → {lh_pattern}")

if __name__ == '__main__':
    main()
