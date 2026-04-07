"""
N-gram Extraction with Positional Encoding

Extracts n-grams (1-3 segments) from word edges (initial and final positions)
with positional encoding markers (^ for initial, $ for final).

Key features:
- Handles labialized consonants as single phonemes
- Extracts only from word edges (no middle n-grams)
- Positional encoding: ^ prefix for initial, $ suffix for final
- N-gram sizes: 1, 2, 3
"""

from phoneme_inventory import tokenize_phonemes


def extract_initial_ngrams(phonemes, max_n=3):
    """
    Extract initial n-grams from a phoneme list.

    Args:
        phonemes (list): List of phonemes (e.g., ['k', 'r', 'a', 't'])
        max_n (int): Maximum n-gram size (default: 3)

    Returns:
        list: Initial n-grams with ^ marker (e.g., ['^k', '^kr', '^kra'])

    Examples:
        >>> extract_initial_ngrams(['k', 'r', 'a', 't'])
        ['^k', '^kr', '^kra']

        >>> extract_initial_ngrams(['a', 'f', 'u', 's'])
        ['^a', '^af', '^afu']
    """
    if not phonemes:
        return []

    ngrams = []
    for n in range(1, min(max_n + 1, len(phonemes) + 1)):
        ngram = ''.join(phonemes[:n])
        ngrams.append(f'^{ngram}')

    return ngrams


def extract_final_ngrams(phonemes, max_n=3):
    """
    Extract final n-grams from a phoneme list.

    Args:
        phonemes (list): List of phonemes (e.g., ['k', 'r', 'a', 't'])
        max_n (int): Maximum n-gram size (default: 3)

    Returns:
        list: Final n-grams with $ marker (e.g., ['t$', 'at$', 'rat$'])

    Examples:
        >>> extract_final_ngrams(['k', 'r', 'a', 't'])
        ['t$', 'at$', 'rat$']

        >>> extract_final_ngrams(['a', 'f', 'u', 's'])
        ['s$', 'us$', 'fus$']
    """
    if not phonemes:
        return []

    ngrams = []
    for n in range(1, min(max_n + 1, len(phonemes) + 1)):
        ngram = ''.join(phonemes[-n:])
        ngrams.append(f'{ngram}$')

    return ngrams


def extract_all_ngrams(word, max_n=3):
    """
    Extract all n-grams from a word (initial and final only).

    Args:
        word (str): Phonemic string (e.g., "krat")
        max_n (int): Maximum n-gram size (default: 3)

    Returns:
        list: All n-grams with positional encoding

    Examples:
        >>> extract_all_ngrams("krat")
        ['^k', '^kr', '^kra', 't$', 'at$', 'rat$']

        >>> extract_all_ngrams("afus")
        ['^a', '^af', '^afu', 's$', 'us$', 'fus$']

        >>> extract_all_ngrams("kʷrat")
        ['^kʷ', '^kʷr', '^kʷra', 't$', 'at$', 'rat$']
    """
    phonemes = tokenize_phonemes(word)

    if not phonemes:
        return []

    initial = extract_initial_ngrams(phonemes, max_n)
    final = extract_final_ngrams(phonemes, max_n)

    return initial + final


def extract_ngrams_from_dataset(df, column='analysisSingularTheme', max_n=3):
    """
    Extract all n-grams from a dataset column.

    Args:
        df (DataFrame): Dataset
        column (str): Column name with phonemic strings
        max_n (int): Maximum n-gram size (default: 3)

    Returns:
        tuple: (ngrams_per_word, all_unique_ngrams)
            - ngrams_per_word: list of lists (one per row)
            - all_unique_ngrams: set of all unique n-grams found

    Examples:
        >>> df = pd.DataFrame({'analysisSingularTheme': ['afus', 'krat']})
        >>> ngrams_list, unique = extract_ngrams_from_dataset(df)
        >>> len(ngrams_list)
        2
        >>> '^a' in unique
        True
    """
    ngrams_per_word = []
    all_unique_ngrams = set()

    for idx, row in df.iterrows():
        word = row[column]

        # Handle missing/null values
        if not isinstance(word, str) or word == '':
            ngrams_per_word.append([])
            continue

        ngrams = extract_all_ngrams(word, max_n)
        ngrams_per_word.append(ngrams)
        all_unique_ngrams.update(ngrams)

    return ngrams_per_word, all_unique_ngrams


if __name__ == '__main__':
    print(f"\n{'='*70}")
    print("N-GRAM EXTRACTION TESTS")
    print(f"{'='*70}\n")

    # Test cases
    test_cases = [
        ("afus", ['^a', '^af', '^afu', 's$', 'us$', 'fus$']),
        ("krat", ['^k', '^kr', '^kra', 't$', 'at$', 'rat$']),
        ("kʷrat", ['^kʷ', '^kʷr', '^kʷra', 't$', 'at$', 'rat$']),
        ("i", ['^i', 'i$']),
        ("ab", ['^a', '^ab', 'b$', 'ab$']),
    ]

    all_passed = True
    for word, expected in test_cases:
        result = extract_all_ngrams(word)
        passed = result == expected
        status = "✅" if passed else "❌"

        print(f"{status} {word:10}")
        print(f"   Result:   {result}")
        if not passed:
            print(f"   Expected: {expected}")
            all_passed = False
        print()

    if all_passed:
        print(f"✅ All n-gram extraction tests passed\n")
    else:
        print(f"❌ Some n-gram extraction tests failed\n")

    # Test with real data sample
    print(f"{'='*70}")
    print("REAL DATA SAMPLE TEST")
    print(f"{'='*70}\n")

    import pandas as pd
    import sys
    sys.path.append('..')

    try:
        df = pd.read_csv('../../data/tash_nouns.csv')
        sample = df.head(5)

        print(f"Extracting n-grams from first 5 records:\n")

        for idx, row in sample.iterrows():
            word = row['analysisSingularTheme']
            if pd.isna(word) or word == '':
                print(f"Record {row['recordID']}: (empty)")
                continue

            ngrams = extract_all_ngrams(word)
            print(f"Record {row['recordID']}: {word}")
            print(f"  N-grams: {', '.join(ngrams)}")
            print()

        print(f"✅ Real data sample test complete")

    except Exception as e:
        print(f"⚠️  Could not test with real data: {e}")
