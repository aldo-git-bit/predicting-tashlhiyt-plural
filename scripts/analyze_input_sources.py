"""
Analyze where the Input column in syllabified_stems_for_review_2.xlsx came from
"""
import pandas as pd
from pathlib import Path

# Load the original source data
source_df = pd.read_csv(str(Path(__file__).parent.parent / 'data' / r'tash_nouns.csv'))

# Load the output xlsx
output_df = pd.read_excel(str(Path(__file__).parent.parent / 'data' / r'syllabified_stems_for_review_2.xlsx'))

print(f"Source data rows: {len(source_df)}")
print(f"Output data rows: {len(output_df)}")
print()

# Track which forms came from where
from_plural = []
from_singular = []
from_unknown = []

# Process each output row and find its source
output_idx = 0
for idx, source_row in source_df.iterrows():
    # Replicate the logic from create_review_spreadsheet_2.py
    plural_stem = source_row.get('analysisPluralTheme', '')
    singular_stem = source_row.get('analysisSingularTheme', '')

    # Determine which stem would have been used
    if pd.notna(plural_stem) and plural_stem != '':
        used_stem = plural_stem
        source_col = 'analysisPluralTheme'
    elif pd.notna(singular_stem) and singular_stem != '':
        used_stem = singular_stem
        source_col = 'analysisSingularTheme'
    else:
        continue  # This row would have been skipped

    # Check if this matches the output
    if output_idx < len(output_df):
        output_stem = output_df.iloc[output_idx]['Input']

        if output_stem == used_stem:
            record = {
                'output_row': output_idx + 2,  # +2 for 1-indexed and header
                'input': used_stem,
                'source_column': source_col,
                'gloss': source_row.get('lexiconGlossEnglish', ''),
                'plural_theme': plural_stem if pd.notna(plural_stem) else '',
                'singular_theme': singular_stem if pd.notna(singular_stem) else ''
            }

            if source_col == 'analysisPluralTheme':
                from_plural.append(record)
            else:
                from_singular.append(record)

            output_idx += 1

print(f"Forms from analysisPluralTheme: {len(from_plural)}")
print(f"Forms from analysisSingularTheme: {len(from_singular)}")
print(f"Total: {len(from_plural) + len(from_singular)}")
print()

# Show summary statistics
print("=" * 80)
print(f"SUMMARY: {len(from_plural)} forms ({len(from_plural)/len(output_df)*100:.1f}%) came from analysisPluralTheme")
print("=" * 80)
print()

# Check for special characters in plural-sourced forms
plural_with_special = [r for r in from_plural if any(c in r['input'] for c in '()/')]
print(f"Forms from analysisPluralTheme containing '(', ')', or '/': {len(plural_with_special)}")
print()

# Find the specific case mentioned by the user
print("=" * 80)
print("SPECIFIC CASE: !tbsa/i)l")
print("=" * 80)
matching = [r for r in from_plural if '!tbsa' in r['input'] or 'tbsa' in r['input']]
for r in matching:
    print(f"Row {r['output_row']}: {r['input']}")
    print(f"  Gloss: {r['gloss']}")
    print(f"  analysisPluralTheme: {r['plural_theme']}")
    print(f"  analysisSingularTheme: {r['singular_theme']}")
    print()

# Show first 20 forms from analysisPluralTheme
print("=" * 80)
print("FIRST 20 FORMS FROM analysisPluralTheme:")
print("=" * 80)
for i, r in enumerate(from_plural[:20], 1):
    print(f"{i}. Row {r['output_row']}: {r['input']}")
    print(f"   Gloss: {r['gloss']}")
    print(f"   Singular would be: {r['singular_theme']}")
    print()

# Save detailed results to CSV
results_df = pd.DataFrame(from_plural)
results_df.to_csv(str(Path(__file__).parent.parent / 'data' / r'forms_from_plural_theme.csv'), index=False)
print(f"Saved detailed list to: data/forms_from_plural_theme.csv")

# Count forms with special characters
special_char_forms = results_df[results_df['input'].str.contains(r'[\(\)/]', regex=True, na=False)]
print(f"\nForms with special characters ( ) /: {len(special_char_forms)}")
if len(special_char_forms) > 0:
    print("\nAll forms with special characters:")
    print(special_char_forms[['output_row', 'input', 'gloss']].to_string(index=False))
