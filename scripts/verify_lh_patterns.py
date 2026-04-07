#!/usr/bin/env python3
"""
Verify L/H pattern extraction with detailed syllable structure display
"""

from rule_based_syllabifier import RuleBasedSyllabifier

# Initialize syllabifier
syllabifier = RuleBasedSyllabifier()

# Test cases from the sample output
test_cases = [
    'lmʕzz',     # lm.ʕz.z → HLL
    'lʕssa',     # l.ʕs.sa → LLL
    'ajdr',      # aj.dr → HL
    'gʃrir',     # gʃ.rir → HH
    'lħsab',     # lħ.sab → HH
    'ktatbi',    # k.tat.bi → LHL
]

print("Verifying L/H Pattern Extraction")
print("=" * 80)

for input_str in test_cases:
    syllabified = syllabifier.syllabify(input_str)
    lh_pattern = syllabifier.get_lh_pattern(input_str)
    structures = syllabifier.get_syllable_structures(input_str)

    print(f"\nInput: {input_str}")
    print(f"Syllabified: {syllabified}")
    print(f"L/H Pattern: {lh_pattern}")
    print(f"Syllable breakdown:")

    for syll in structures:
        onset_str = ','.join([s.text for s in syll.onset]) if syll.onset else '∅'
        peak_str = syll.peak.text if syll.peak else '∅'
        coda_str = ','.join([s.text for s in syll.coda]) if syll.coda else '∅'
        lh = 'H' if syll.coda and len(syll.coda) > 0 else 'L'

        syll_str = syll.to_string()
        print(f"  [{syll_str:6s}] onset={onset_str:4s} peak={peak_str:2s} coda={coda_str:4s} → {lh}")

print("\n" + "=" * 80)
