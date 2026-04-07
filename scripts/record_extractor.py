#!/usr/bin/env python3
"""
Record Extractor for Tashlhiyt Plural Analysis
Extracts and samples records based on analysisInternalChanges values.
"""

import pandas as pd
import sys
from pathlib import Path
from difflib import get_close_matches


def get_valid_internal_changes(df):
    """Extract all unique individual values from the multivalue analysisInternalChanges field."""
    all_values = []
    for val in df['analysisInternalChanges'].dropna():
        individual_values = str(val).split('\n')
        all_values.extend(individual_values)
    return sorted(set(all_values))


def find_matching_rows(df, search_string):
    """Find all rows where search_string appears in analysisInternalChanges field."""
    mask = df['analysisInternalChanges'].apply(
        lambda x: search_string in str(x) if pd.notna(x) else False
    )
    return df[mask]


def format_record(row):
    """Format a single record according to the template."""
    def get_val(field):
        """Get field value or empty string if NaN."""
        val = row[field]
        return '' if pd.isna(val) else str(val)

    # Build the record with proper formatting
    lines = [
        f"{get_val('recordID')}\t{get_val('lexiconGlossEnglish')}\t{get_val('lexiconGlossFrench')}",
        "",
        get_val('analysisSingularTheme'),
        get_val('analysisPluralTheme'),
        "",
        f"{get_val('word1MSF')}\t{get_val('word2MSB')}",
        f"{get_val('word3MPF')}\t{get_val('word4MPB')}",
        f"{get_val('word5FSF')}\t{get_val('word6FSB')}",
        f"{get_val('word7FPF')}\t{get_val('word8FPB')}",
        "",
        get_val('analysisInternalChanges'),
        get_val('analysisNotes')
    ]

    return '\n'.join(lines)


def main():
    # Set up paths
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / 'data' / 'nouns_internal_changes.xlsx'
    output_dir = base_dir / 'reports'

    # Create reports directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_excel(data_file)

    # Get valid internal changes values
    valid_values = get_valid_internal_changes(df)

    print("\nValid analysisInternalChanges values:")
    for i, val in enumerate(valid_values, 1):
        print(f"  {i}. {val}")

    # Prompt for search string
    print("\n" + "="*60)
    search_string = input("Enter search string: ").strip()

    # Validate input
    if search_string not in valid_values:
        # Try to find close matches
        close_matches = get_close_matches(search_string, valid_values, n=3, cutoff=0.6)
        if close_matches:
            print(f"\n'{search_string}' is not a valid value.")
            print(f"Did you mean: {', '.join(close_matches)}?")
        else:
            print(f"\n'{search_string}' is not a valid value and no close matches found.")
            print(f"Valid values are: {', '.join(valid_values)}")
        return

    # Find matching rows
    print(f"\nSearching for rows containing '{search_string}'...")
    matching_rows = find_matching_rows(df, search_string)

    if len(matching_rows) == 0:
        print(f"No rows found containing '{search_string}'.")
        return

    print(f"Found {len(matching_rows)} matching rows.")

    # Sample up to 20 rows
    sample_size = min(20, len(matching_rows))
    sampled_rows = matching_rows.sample(n=sample_size, random_state=None)

    if sample_size < 20:
        print(f"Note: Only {sample_size} rows available (less than 20).")
    else:
        print(f"Randomly sampling {sample_size} rows.")

    # Generate output
    output_filename = f"internal_changes_{search_string.replace(' ', '_').replace('ʷ', 'w')}.txt"
    output_path = output_dir / output_filename

    print(f"\nGenerating report: {output_filename}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Internal Changes Report: {search_string}\n")
        f.write(f"Total matching rows: {len(matching_rows)}\n")
        f.write(f"Sampled rows: {sample_size}\n")
        f.write(f"{'='*60}\n\n")

        # Write template structure
        f.write("Record Organization\n\n")
        f.write("recordID\tlexiconGlossEnglish\tlexiconGlossFrench\n\n")
        f.write("analysisSingularTheme\n")
        f.write("analysisPluralTheme\n\n")
        f.write("word1MSF\tword2MSB\n")
        f.write("word3MPF\tword4MPB\n")
        f.write("word5FSF\tword6FSB\n")
        f.write("word7FPF\tword8FPB\n\n")
        f.write("analysisInternalChanges\n")
        f.write("analysisNotes\n\n")
        f.write(f"{'='*60}\n\n")

        for idx, (_, row) in enumerate(sampled_rows.iterrows(), 1):
            f.write(format_record(row))
            if idx < sample_size:
                f.write(f"\n\n{'='*60}\n\n")

    print(f"✓ Report saved to: {output_path}")
    print(f"\nSummary: {sample_size} records extracted from {len(matching_rows)} total matches.")


if __name__ == "__main__":
    main()
