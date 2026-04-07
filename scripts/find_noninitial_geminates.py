import pandas as pd
import re

# Read the data
df = pd.read_csv('data/tash_nouns.csv')

# Filter to rows that have p_stem_sing_syllabified
df_with_syll = df[df['p_stem_sing_syllabified'].notna()].copy()

# Function to check if syllabification has a NON-INITIAL geminate split across syllable boundary
def has_split_geminate(syll):
    if pd.isna(syll) or not isinstance(syll, str):
        return False

    # Find all geminate patterns X.X in the syllabification
    geminate_matches = list(re.finditer(r'(.)\.\1', syll))

    if not geminate_matches:
        return False

    # Check each geminate to see if it's initial (to exclude) or non-initial (to include)
    for match in geminate_matches:
        start_pos = match.start()

        # Remove optional ! prefix for position calculation
        check_syll = syll.lstrip('!')
        adjusted_start = start_pos
        if syll.startswith('!'):
            adjusted_start = start_pos - 1

        # Initial geminates start within the first 3 characters (0, 1, or 2)
        # Examples:
        # - ʃ.ʃur.fa -> X.X starts at position 0 (initial)
        # - mʃ.ʃur.da -> X.X starts at position 1 (initial)
        # - nf.ful.sa -> X.X starts at position 1 (initial)
        # - lm.ʕz.z -> X.X starts at position 5 (NOT initial)
        # - m.mus.s.na -> s.s starts at position 5 (NOT initial)

        if adjusted_start < 3:  # Skip initial geminates
            continue

        # Check if this is a ZX.XV pattern (to exclude)
        # Pattern: X.X followed immediately by a vowel (a, i, or u)
        # Examples to exclude: l.ʕs.sa, nz.zum.ma, nu.hn.nu
        # Examples to keep: m.mut.tl.a (has t.tl, not t.ta)
        end_pos = match.end()
        if end_pos < len(syll) and syll[end_pos] in 'aiu':
            # This is ZX.XV pattern, skip it
            continue

        # This is a valid non-initial geminate that's not ZX.XV
        return True

    return False

# Find all rows with split geminates
split_geminate_rows = df_with_syll[df_with_syll['p_stem_sing_syllabified'].apply(has_split_geminate)]

# Select the columns we need
output_df = split_geminate_rows[['recordID', 'lexiconGlossFrench', 'lexiconGlossEnglish', 'p_stem_sing_syllabified']].copy()

# Sort by recordID
output_df = output_df.sort_values('recordID').reset_index(drop=True)

print(f"Found {len(output_df)} cases with split geminates:")
print(output_df.to_string())

# Save to Excel
output_df.to_excel('data/noninitial_geminates.xlsx', index=False)
print(f"\nSaved to data/noninitial_geminates.xlsx")
