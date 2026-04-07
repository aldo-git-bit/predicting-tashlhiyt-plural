"""
Add p_stem_sing_syllabified column to tash_nouns.csv
Takes analysisSingularTheme as input and applies syllabification
"""
import sys
from pathlib import Path
# sys.path modification removed - using relative imports
# Scripts are in the same directory

import pandas as pd
import shutil
from rule_based_syllabifier import RuleBasedSyllabifier

# File paths
input_file = str(Path(__file__).parent.parent / 'data' / r'tash_nouns.csv')
backup_file = str(Path(__file__).parent.parent / 'data' / r'tash_nouns.csv.backup')

# Create backup
print("Creating backup...")
shutil.copy2(input_file, backup_file)
print(f"Backup created: {backup_file}")
print()

# Load the data
print("Loading data...")
df = pd.read_csv(input_file)
print(f"Total rows: {len(df)}")
print()

# Initialize syllabifier
syllabifier = RuleBasedSyllabifier()

# Check if column already exists
if 'p_stem_sing_syllabified' in df.columns:
    print("WARNING: Column 'p_stem_sing_syllabified' already exists. It will be overwritten.")
    print()

# Create the new column
print("Creating p_stem_sing_syllabified column...")
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
        print(f"Processed {idx + 1}/{len(df)} rows...")

# Add the column to the dataframe
df['p_stem_sing_syllabified'] = syllabified_values

print()
print("Summary:")
print(f"  Syllabified: {count_syllabified} rows")
print(f"  Left empty (no singular theme): {count_empty} rows")
print()

# Save the enriched dataframe
print("Saving enriched dataframe...")
df.to_csv(input_file, index=False)
print(f"Saved: {input_file}")
print()

# Show some examples
print("="*80)
print("Sample results (first 10 with singular themes):")
print("="*80)
sample = df[df['p_stem_sing_syllabified'].notna()].head(10)
print(sample[['analysisSingularTheme', 'p_stem_sing_syllabified', 'lexiconGlossEnglish']].to_string(index=False))
print()

# Show the Only Plural cases
print("="*80)
print("Sample Only Plural cases (first 5 without singular themes):")
print("="*80)
only_plural = df[df['p_stem_sing_syllabified'].isna()].head(5)
if len(only_plural) > 0:
    print(only_plural[['analysisSingularTheme', 'analysisPluralTheme', 'analysisPluralPattern', 'lexiconGlossEnglish']].to_string(index=False))
else:
    print("No rows without singular themes found.")

print()
print("Done!")
