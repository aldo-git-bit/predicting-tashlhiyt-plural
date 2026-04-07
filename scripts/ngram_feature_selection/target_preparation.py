"""
Target Variable Preparation

Prepares multi-label target variables for n-gram feature selection.

Target Variables (8 total):

Macro-level (n=1,185):
- y_macro_suffix: Has external suffix (External or Mixed patterns)
- y_macro_mutated: Has internal mutation (Internal or Mixed patterns)

Micro-level (n=562):
- y_micro_ablaut: Has Ablaut mutation
- y_micro_templatic: Has Templatic mutation
- y_micro_medial_a: Has Medial A mutation
- y_micro_final_a: Has Final A mutation
- y_micro_final_vw: Has Final Vw mutation
- y_micro_insert_c: Has Insert C mutation

Note: Multi-label approach - a noun with "Ablaut\nMedial A" gets y=1 for both targets.
"""

import pandas as pd
import numpy as np


# Mutation type names (standardized from analysisInternalChanges)
MUTATION_TYPES = [
    'Ablaut',
    'Templatic',
    'Medial A',
    'Final A',
    'Final Vw',
    'Insert C'
]


def has_mutation(internal_changes_str, mutation_type):
    """
    Check if a mutation type is present in the internal changes field.

    Args:
        internal_changes_str (str): Value from analysisInternalChanges
        mutation_type (str): Mutation type to check for

    Returns:
        bool: True if mutation is present, False otherwise

    Examples:
        >>> has_mutation("Ablaut", "Ablaut")
        True

        >>> has_mutation("Ablaut\\nMedial A", "Ablaut")
        True

        >>> has_mutation("Ablaut\\nMedial A", "Medial A")
        True

        >>> has_mutation("Templatic", "Ablaut")
        False

        >>> has_mutation(np.nan, "Ablaut")
        False
    """
    if pd.isna(internal_changes_str) or internal_changes_str == '':
        return False

    # Split on newline (multi-value field separator)
    mutations = [m.strip() for m in str(internal_changes_str).split('\n')]

    return mutation_type in mutations


def prepare_macro_targets(df):
    """
    Prepare macro-level target variables.

    Args:
        df (DataFrame): Dataset with analysisPluralPattern column

    Returns:
        DataFrame: Two binary columns (y_macro_suffix, y_macro_mutated)

    Inclusion criteria:
        - Exclude: No Plural, Only Plural, id Plural
        - Include: External, Internal, Mixed (n=1,185 total)
    """
    # Filter to usable nouns
    usable_patterns = ['External', 'Internal', 'Mixed']
    df_macro = df[df['analysisPluralPattern'].isin(usable_patterns)].copy()

    # y_macro_suffix: External or Mixed (has suffix)
    df_macro['y_macro_suffix'] = df_macro['analysisPluralPattern'].isin(
        ['External', 'Mixed']
    ).astype(int)

    # y_macro_mutated: Internal or Mixed (has mutation)
    df_macro['y_macro_mutated'] = df_macro['analysisPluralPattern'].isin(
        ['Internal', 'Mixed']
    ).astype(int)

    return df_macro[['y_macro_suffix', 'y_macro_mutated']]


def prepare_micro_targets(df):
    """
    Prepare micro-level target variables.

    Args:
        df (DataFrame): Dataset with analysisPluralPattern and analysisInternalChanges

    Returns:
        DataFrame: Six binary columns (one per mutation type)

    Inclusion criteria:
        - Only Internal or Mixed patterns (n=562 total)
        - Multi-label: noun with "Ablaut\\nMedial A" gets y=1 for both
    """
    # Filter to mutated nouns only
    mutated_patterns = ['Internal', 'Mixed']
    df_micro = df[df['analysisPluralPattern'].isin(mutated_patterns)].copy()

    # Create binary column for each mutation type
    for mutation in MUTATION_TYPES:
        col_name = f"y_micro_{mutation.lower().replace(' ', '_')}"
        df_micro[col_name] = df_micro['analysisInternalChanges'].apply(
            lambda x: has_mutation(x, mutation)
        ).astype(int)

    target_cols = [f"y_micro_{m.lower().replace(' ', '_')}" for m in MUTATION_TYPES]

    return df_micro[target_cols]


def prepare_all_targets(df):
    """
    Prepare all 8 target variables with proper filtering.

    Args:
        df (DataFrame): Full dataset

    Returns:
        tuple: (df_macro_targets, df_micro_targets, metadata)
            - df_macro_targets: DataFrame with 2 macro targets (n=1,185)
            - df_micro_targets: DataFrame with 6 micro targets (n=562)
            - metadata: dict with sample sizes and distributions

    Example:
        >>> df = pd.read_csv('data/tash_nouns.csv')
        >>> macro, micro, meta = prepare_all_targets(df)
        >>> print(f"Macro samples: {len(macro)}, Micro samples: {len(micro)}")
    """
    # Prepare macro targets
    usable_patterns = ['External', 'Internal', 'Mixed']
    df_macro_filtered = df[df['analysisPluralPattern'].isin(usable_patterns)].copy()
    macro_targets = prepare_macro_targets(df)

    # Prepare micro targets
    mutated_patterns = ['Internal', 'Mixed']
    df_micro_filtered = df[df['analysisPluralPattern'].isin(mutated_patterns)].copy()
    micro_targets = prepare_micro_targets(df)

    # Add indices for alignment
    macro_targets.index = df_macro_filtered.index
    micro_targets.index = df_micro_filtered.index

    # Metadata
    metadata = {
        'n_total': len(df),
        'n_macro': len(macro_targets),
        'n_micro': len(micro_targets),
        'macro_targets': {
            'suffix': macro_targets['y_macro_suffix'].sum(),
            'mutated': macro_targets['y_macro_mutated'].sum()
        },
        'micro_targets': {}
    }

    for mutation in MUTATION_TYPES:
        col_name = f"y_micro_{mutation.lower().replace(' ', '_')}"
        if col_name in micro_targets.columns:
            metadata['micro_targets'][mutation] = micro_targets[col_name].sum()

    return macro_targets, micro_targets, metadata


if __name__ == '__main__':
    print(f"\n{'='*70}")
    print("TARGET VARIABLE PREPARATION TESTS")
    print(f"{'='*70}\n")

    # Test 1: Mutation detection
    print("Test 1: Mutation detection")
    test_cases = [
        ("Ablaut", "Ablaut", True),
        ("Ablaut\nMedial A", "Ablaut", True),
        ("Ablaut\nMedial A", "Medial A", True),
        ("Templatic", "Ablaut", False),
        (np.nan, "Ablaut", False),
        ("", "Ablaut", False)
    ]

    all_passed = True
    for internal_changes, mutation, expected in test_cases:
        result = has_mutation(internal_changes, mutation)
        passed = result == expected
        status = "✅" if passed else "❌"
        print(f"{status} has_mutation({repr(internal_changes)}, {repr(mutation)}) = {result}")
        if not passed:
            all_passed = False

    if all_passed:
        print(f"\n✅ All mutation detection tests passed\n")
    else:
        print(f"\n❌ Some mutation detection tests failed\n")

    # Test 2: Real data
    print("Test 2: Target preparation with real data")
    try:
        df = pd.read_csv('../../data/tash_nouns.csv')

        macro_targets, micro_targets, metadata = prepare_all_targets(df)

        print(f"✅ Targets prepared successfully\n")

        print("Sample sizes:")
        print(f"  Total nouns: {metadata['n_total']}")
        print(f"  Macro targets (External/Internal/Mixed): {metadata['n_macro']}")
        print(f"  Micro targets (Internal/Mixed only): {metadata['n_micro']}")
        print()

        print("Macro target distributions:")
        print(f"  y_macro_suffix (has suffix): {metadata['macro_targets']['suffix']}")
        print(f"  y_macro_mutated (has mutation): {metadata['macro_targets']['mutated']}")
        print()

        print("Micro target distributions:")
        for mutation, count in metadata['micro_targets'].items():
            print(f"  y_micro_{mutation.lower().replace(' ', '_')}: {count}")
        print()

        # Check for multi-label cases
        print("Multi-label analysis (micro targets):")
        micro_targets['n_mutations'] = micro_targets.sum(axis=1)
        multi_label_counts = micro_targets['n_mutations'].value_counts().sort_index()
        for n_mutations, count in multi_label_counts.items():
            print(f"  {int(n_mutations)} mutations: {count} nouns")
        print()

        # Show example multi-label cases
        print("Example multi-label cases:")
        multi_label_examples = df.loc[
            micro_targets[micro_targets['n_mutations'] >= 2].index
        ].head(5)

        for idx, row in multi_label_examples.iterrows():
            mutations = row['analysisInternalChanges']
            print(f"  Record {row['recordID']}: {mutations}")

        print(f"\n✅ All tests passed")

    except Exception as e:
        print(f"⚠️  Error testing with real data: {e}")
        import traceback
        traceback.print_exc()
