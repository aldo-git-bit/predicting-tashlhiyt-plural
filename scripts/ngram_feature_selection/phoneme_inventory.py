"""
Tashlhiyt Phoneme Inventory

Defines the complete phoneme inventory for Tashlhiyt Berber with proper handling
of labialized consonants as single phonemes.

Total: 30 segments
- 22 simple consonants
- 5 labialized consonants (treated as single phonemes)
- 3 vowels
"""

# Simple consonants (22)
SIMPLE_CONSONANTS = [
    'k', 't', 'q',           # Voiceless stops
    'd', 'b', 'g',           # Voiced stops
    's', 'f', 'ʃ', 'ħ', 'h', 'χ',  # Voiceless fricatives
    'z', 'ʒ', 'ʁ', 'ʕ',     # Voiced fricatives
    'r', 'l',                # Liquids
    'm', 'n',                # Nasals
    'j', 'w'                 # Glides
]

# Labialized consonants (5) - treated as SINGLE phonemes
LABIALIZED_CONSONANTS = [
    'kʷ', 'gʷ', 'qʷ', 'χʷ', 'ʁʷ'
]

# Vowels (3)
VOWELS = ['a', 'i', 'u']

# Complete inventory (30 total)
ALL_PHONEMES = LABIALIZED_CONSONANTS + SIMPLE_CONSONANTS + VOWELS


def tokenize_phonemes(word):
    """
    Tokenize a phonemic string into individual phonemes.

    Handles labialized consonants as single units (kʷ, gʷ, qʷ, χʷ, ʁʷ).

    Args:
        word (str): Phonemic string (e.g., "kʷrat")

    Returns:
        list: List of phonemes (e.g., ["kʷ", "r", "a", "t"])

    Examples:
        >>> tokenize_phonemes("kʷrat")
        ['kʷ', 'r', 'a', 't']

        >>> tokenize_phonemes("afus")
        ['a', 'f', 'u', 's']

        >>> tokenize_phonemes("aqʷrab")
        ['a', 'qʷ', 'r', 'a', 'b']
    """
    if not isinstance(word, str) or word == '':
        return []

    phonemes = []
    i = 0

    while i < len(word):
        # Check for labialized consonants (2-character sequences)
        if i + 1 < len(word):
            two_char = word[i:i+2]
            if two_char in LABIALIZED_CONSONANTS:
                phonemes.append(two_char)
                i += 2
                continue

        # Single character phoneme
        phonemes.append(word[i])
        i += 1

    return phonemes


def validate_phoneme_inventory():
    """
    Validate that the phoneme inventory is correctly defined.

    Returns:
        tuple: (is_valid, message)
    """
    # Check total count
    if len(ALL_PHONEMES) != 30:
        return False, f"Expected 30 phonemes, found {len(ALL_PHONEMES)}"

    # Check for duplicates
    if len(ALL_PHONEMES) != len(set(ALL_PHONEMES)):
        return False, "Duplicate phonemes found in inventory"

    # Check component counts
    if len(SIMPLE_CONSONANTS) != 22:
        return False, f"Expected 22 simple consonants, found {len(SIMPLE_CONSONANTS)}"

    if len(LABIALIZED_CONSONANTS) != 5:
        return False, f"Expected 5 labialized consonants, found {len(LABIALIZED_CONSONANTS)}"

    if len(VOWELS) != 3:
        return False, f"Expected 3 vowels, found {len(VOWELS)}"

    return True, "Phoneme inventory validated successfully"


if __name__ == '__main__':
    # Validation
    is_valid, message = validate_phoneme_inventory()
    print(f"\n{'='*70}")
    print("PHONEME INVENTORY VALIDATION")
    print(f"{'='*70}\n")

    if is_valid:
        print(f"✅ {message}")
        print(f"\nTotal phonemes: {len(ALL_PHONEMES)}")
        print(f"  Simple consonants: {len(SIMPLE_CONSONANTS)}")
        print(f"  Labialized consonants: {len(LABIALIZED_CONSONANTS)}")
        print(f"  Vowels: {len(VOWELS)}")

        print(f"\nLabialized consonants: {', '.join(LABIALIZED_CONSONANTS)}")
        print(f"\nSimple consonants:")
        for i in range(0, len(SIMPLE_CONSONANTS), 10):
            print(f"  {', '.join(SIMPLE_CONSONANTS[i:i+10])}")
        print(f"\nVowels: {', '.join(VOWELS)}")
    else:
        print(f"❌ {message}")

    # Test tokenization
    print(f"\n{'='*70}")
    print("TOKENIZATION TESTS")
    print(f"{'='*70}\n")

    test_cases = [
        ("kʷrat", ['kʷ', 'r', 'a', 't']),
        ("afus", ['a', 'f', 'u', 's']),
        ("aqʷrab", ['a', 'qʷ', 'r', 'a', 'b']),
        ("χʷriʁʷ", ['χʷ', 'r', 'i', 'ʁʷ']),
        ("gʷma", ['gʷ', 'm', 'a'])
    ]

    all_passed = True
    for word, expected in test_cases:
        result = tokenize_phonemes(word)
        passed = result == expected
        status = "✅" if passed else "❌"
        print(f"{status} {word:10} → {result}")
        if not passed:
            print(f"   Expected: {expected}")
            all_passed = False

    if all_passed:
        print(f"\n✅ All tokenization tests passed")
    else:
        print(f"\n❌ Some tokenization tests failed")
