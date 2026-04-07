#!/usr/bin/env python3
"""
Add foot structure column to tash_nouns.csv

Reads the p_stem_sing_LH column and maps to foot structures
using right-to-left moraic trochee principle.

Foot Structure Rules:
- H (heavy syllable) → F (one foot)
- LL (two light syllables) → F (one foot)
- L (unparsed light) → l (lowercase, unparsed)

Processing is right-to-left.
"""

import pandas as pd
from datetime import datetime


def lh_to_foot(lh_pattern):
    """
    Convert LH pattern to foot structure using right-to-left moraic trochee.

    Examples:
        LH → lF
        HH → FF
        LLL → lF
        LLH → FF
        HHL → FFl
        LLLL → FF
    """
    if pd.isna(lh_pattern) or lh_pattern == '':
        return ''

    # Process right-to-left
    result = []
    i = len(lh_pattern) - 1

    while i >= 0:
        if lh_pattern[i] == 'H':
            # Heavy syllable = one foot
            result.append('F')
            i -= 1
        elif lh_pattern[i] == 'L':
            # Check if there's another L to the left
            if i > 0 and lh_pattern[i-1] == 'L':
                # LL = one foot
                result.append('F')
                i -= 2
            else:
                # Single L = unparsed light
                result.append('l')
                i -= 1
        else:
            # Unknown character, skip
            i -= 1

    # Reverse because we built right-to-left
    return ''.join(reversed(result))


def main():
    # Load dataset
    csv_path = 'data/tash_nouns.csv'
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Total rows: {len(df)}")

    # Create backup with timestamp
    backup_path = f"data/tash_nouns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"Creating backup at {backup_path}...")
    df.to_csv(backup_path, index=False)

    # Check if p_stem_sing_foot column already exists
    if 'p_stem_sing_foot' in df.columns:
        print("Column p_stem_sing_foot already exists. It will be overwritten.")
        df = df.drop(columns=['p_stem_sing_foot'])

    # Check that LH column exists
    if 'p_stem_sing_LH' not in df.columns:
        print("ERROR: p_stem_sing_LH column not found. Run add_lh_column.py first.")
        return

    # Extract foot patterns
    print("\nExtracting foot patterns from LH patterns...")
    foot_patterns = []
    processed = 0
    empty = 0

    for idx, row in df.iterrows():
        lh_pattern = row.get('p_stem_sing_LH', '')

        if pd.isna(lh_pattern) or lh_pattern == '':
            foot_patterns.append('')
            empty += 1
        else:
            foot_pattern = lh_to_foot(lh_pattern)
            foot_patterns.append(foot_pattern)
            processed += 1

            # Progress indicator
            if processed % 100 == 0:
                print(f"  Processed {processed} forms...")

    # Add new column
    df['p_stem_sing_foot'] = foot_patterns

    # Save updated dataset
    print(f"\nSaving updated dataset to {csv_path}...")
    df.to_csv(csv_path, index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("Foot Structure Extraction Complete!")
    print("=" * 70)
    print(f"Total rows: {len(df)}")
    print(f"Foot patterns extracted: {processed}")
    print(f"Empty inputs (skipped): {empty}")
    print(f"\nNew column 'p_stem_sing_foot' added to {csv_path}")
    print(f"Backup saved at {backup_path}")

    # Show some examples
    print("\nSample foot patterns:")
    print("-" * 70)
    sample_df = df[df['p_stem_sing_foot'] != ''].head(15)
    for _, row in sample_df.iterrows():
        input_str = row.get('analysisSingularTheme', '')
        syllabified = row.get('p_stem_sing_syllabified', '')
        lh_pattern = row.get('p_stem_sing_LH', '')
        foot_pattern = row.get('p_stem_sing_foot', '')
        print(f"  {input_str:15s} → {syllabified:15s} → {lh_pattern:6s} → {foot_pattern}")

    # Show pattern distribution
    print("\nFoot pattern distribution (top 10):")
    print("-" * 70)
    foot_counts = df['p_stem_sing_foot'].value_counts().head(10)
    for pattern, count in foot_counts.items():
        if pattern != '':
            print(f"  {pattern:6s}: {count:4d} forms")


if __name__ == '__main__':
    main()
