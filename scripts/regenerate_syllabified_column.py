"""
Regenerate syllabified singular column in tash_nouns.csv
- Creates backup with date suffix
- Removes duplicate p_stem_syllabified column
- Regenerates p_stem_sing_syllabified with updated algorithm
"""
import sys
# sys.path modification removed - using relative imports
# Scripts are in the same directory

import pandas as pd
import shutil
from datetime import datetime
from pathlib import Path
from rule_based_syllabifier import RuleBasedSyllabifier

def main():
    # File paths
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / 'data' / 'tash_nouns.csv'

    # Create backup with date suffix
    date_suffix = datetime.now().strftime('%Y%m%d')
    backup_file = base_dir / 'data' / f'tash_nouns_{date_suffix}.csv'

    print("="*80)
    print("REGENERATING SYLLABIFIED COLUMN")
    print("="*80)
    print()

    print(f"Creating backup with date suffix: {date_suffix}")
    shutil.copy2(input_file, backup_file)
    print(f"✓ Backup created: {backup_file}")
    print()

    # Load the data
    print("Loading data...")
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df)} rows")
    print()

    # Check which columns exist
    has_p_stem_syllabified = 'p_stem_syllabified' in df.columns
    has_p_stem_sing_syllabified = 'p_stem_sing_syllabified' in df.columns

    print("Current columns:")
    if has_p_stem_syllabified:
        print("  • p_stem_syllabified (will be removed)")
    if has_p_stem_sing_syllabified:
        print("  • p_stem_sing_syllabified (will be regenerated)")
    print()

    # Remove duplicate column if it exists
    if has_p_stem_syllabified:
        print("Removing duplicate column 'p_stem_syllabified'...")
        df = df.drop(columns=['p_stem_syllabified'])
        print("✓ Column removed")
        print()

    # Initialize syllabifier (with updated exceptions)
    syllabifier = RuleBasedSyllabifier()

    # Regenerate p_stem_sing_syllabified column
    print("Regenerating p_stem_sing_syllabified column with updated algorithm...")
    syllabified_values = []
    count_syllabified = 0
    count_empty = 0

    for idx, row in df.iterrows():
        singular_theme = row.get('analysisSingularTheme', '')

        # Only syllabify if analysisSingularTheme is not empty
        if pd.notna(singular_theme) and singular_theme != '':
            syllabified = syllabifier.syllabify(singular_theme)
            syllabified_values.append(syllabified)
            count_syllabified += 1
        else:
            # Leave empty for Only Plural nouns
            syllabified_values.append(None)
            count_empty += 1

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} rows...")

    # Add/update the column
    df['p_stem_sing_syllabified'] = syllabified_values

    print()
    print("Summary:")
    print(f"  ✓ Syllabified: {count_syllabified} rows")
    print(f"  ✓ Left empty (no singular theme): {count_empty} rows")
    print()

    # Save the updated dataframe
    print("Saving updated dataframe...")
    df.to_csv(input_file, index=False)
    print(f"✓ Saved: {input_file}")
    print()

    # Test the 4 new exceptions
    print("="*80)
    print("TESTING NEW EXCEPTIONS")
    print("="*80)

    exception_stems = ['!mddsr', 'mssfld', 'krr', 'mttʃu']
    expected_results = ['mdd.sr', 'mss.fld', 'krr', 'mtt.ʃu']

    for stem, expected in zip(exception_stems, expected_results):
        # Find this stem in the dataframe
        matching_rows = df[df['analysisSingularTheme'] == stem]
        if len(matching_rows) > 0:
            actual = matching_rows.iloc[0]['p_stem_sing_syllabified']
            match = "✓" if actual == expected else "✗"
            print(f"{match} {stem:15} → {actual:15} (expected: {expected})")
        else:
            print(f"  {stem:15} → (not found in dataset)")

    print()

    # Show some sample results
    print("="*80)
    print("SAMPLE RESULTS (first 10 with singular themes)")
    print("="*80)
    sample = df[df['p_stem_sing_syllabified'].notna()].head(10)
    for _, row in sample.iterrows():
        input_form = row['analysisSingularTheme']
        output_form = row['p_stem_sing_syllabified']
        gloss = row.get('lexiconGlossEnglish', '')
        print(f"{input_form:20} → {output_form:20} ({gloss})")

    print()
    print("="*80)
    print("DONE!")
    print("="*80)
    print(f"Backup: {backup_file}")
    print(f"Updated: {input_file}")

if __name__ == "__main__":
    main()
