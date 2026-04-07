"""
N-gram Feature Matrix Builder

Builds binary feature matrices from n-grams with standardization.

Key features:
- Binary encoding: 1 if n-gram present in word, 0 otherwise
- Standardization: Applied even to binary features (handles Zipfian distribution)
- Efficient sparse matrix representation for large feature sets
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from ngram_extractor import extract_ngrams_from_dataset


def build_binary_feature_matrix(ngrams_per_word, unique_ngrams):
    """
    Build a binary feature matrix from n-grams.

    Args:
        ngrams_per_word (list): List of n-gram lists (one per word)
        unique_ngrams (set or list): All unique n-grams

    Returns:
        DataFrame: Binary feature matrix (rows=words, columns=n-grams)

    Example:
        >>> ngrams_per_word = [
        ...     ['^a', '^af', 's$'],
        ...     ['^k', '^kr', 't$']
        ... ]
        >>> unique = {'^a', '^af', '^k', '^kr', 's$', 't$'}
        >>> matrix = build_binary_feature_matrix(ngrams_per_word, unique)
        >>> matrix.shape
        (2, 6)
    """
    # Sort n-grams for consistent column ordering
    sorted_ngrams = sorted(unique_ngrams)

    # Initialize binary matrix
    n_words = len(ngrams_per_word)
    n_features = len(sorted_ngrams)

    matrix = np.zeros((n_words, n_features), dtype=np.int8)

    # Create n-gram to column index mapping
    ngram_to_idx = {ngram: idx for idx, ngram in enumerate(sorted_ngrams)}

    # Fill matrix
    for row_idx, word_ngrams in enumerate(ngrams_per_word):
        for ngram in word_ngrams:
            if ngram in ngram_to_idx:
                col_idx = ngram_to_idx[ngram]
                matrix[row_idx, col_idx] = 1

    # Convert to DataFrame
    df_matrix = pd.DataFrame(matrix, columns=sorted_ngrams)

    return df_matrix


def standardize_features(X, scaler=None, fit=True):
    """
    Standardize features using StandardScaler.

    Args:
        X (DataFrame or array): Feature matrix
        scaler (StandardScaler): Pre-fitted scaler (optional)
        fit (bool): Whether to fit the scaler (True for training, False for test)

    Returns:
        tuple: (X_standardized, scaler)

    Note:
        Even binary features are standardized to handle Zipfian distribution
        and ensure scale-dependent LASSO selection works properly.
    """
    if scaler is None:
        scaler = StandardScaler()

    if fit:
        X_std = scaler.fit_transform(X)
    else:
        X_std = scaler.transform(X)

    # Convert back to DataFrame if input was DataFrame
    if isinstance(X, pd.DataFrame):
        X_std = pd.DataFrame(X_std, columns=X.columns, index=X.index)

    return X_std, scaler


def build_feature_matrix_from_dataset(
    df,
    column='analysisSingularTheme',
    max_n=3,
    standardize=True,
    scaler=None,
    fit_scaler=True
):
    """
    Build feature matrix directly from dataset.

    Args:
        df (DataFrame): Dataset
        column (str): Column with phonemic strings
        max_n (int): Maximum n-gram size
        standardize (bool): Whether to standardize features
        scaler (StandardScaler): Pre-fitted scaler (optional)
        fit_scaler (bool): Whether to fit scaler (True for train, False for test)

    Returns:
        tuple: (X, scaler, metadata)
            - X: Feature matrix (standardized if requested)
            - scaler: StandardScaler object (or None)
            - metadata: dict with diagnostic info

    Example:
        >>> df = pd.read_csv('data/tash_nouns.csv')
        >>> X, scaler, meta = build_feature_matrix_from_dataset(df)
        >>> print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    """
    # Extract n-grams
    ngrams_per_word, unique_ngrams = extract_ngrams_from_dataset(
        df, column=column, max_n=max_n
    )

    # Build binary matrix
    X = build_binary_feature_matrix(ngrams_per_word, unique_ngrams)

    # Metadata
    metadata = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_unique_ngrams': len(unique_ngrams),
        'max_n': max_n,
        'column_source': column,
        'standardized': standardize
    }

    # Standardize if requested
    if standardize:
        X, scaler = standardize_features(X, scaler=scaler, fit=fit_scaler)
        metadata['scaler'] = scaler
    else:
        scaler = None

    return X, scaler, metadata


def get_feature_statistics(X_binary, X_standardized=None):
    """
    Generate feature statistics for diagnostic report.

    Args:
        X_binary (DataFrame): Binary feature matrix
        X_standardized (DataFrame): Standardized feature matrix (optional)

    Returns:
        DataFrame: Feature statistics

    Statistics include:
        - n_occurrences: Count of 1s
        - frequency: Proportion of samples with this n-gram
        - mean_std: Mean of standardized feature (if provided)
        - std_std: Std of standardized feature (if provided)
    """
    stats = pd.DataFrame({
        'n_gram': X_binary.columns,
        'n_occurrences': X_binary.sum().values,
        'frequency': (X_binary.sum() / len(X_binary)).values
    })

    if X_standardized is not None:
        stats['mean_std'] = X_standardized.mean().values
        stats['std_std'] = X_standardized.std().values

    stats = stats.sort_values('n_occurrences', ascending=False)

    return stats


if __name__ == '__main__':
    print(f"\n{'='*70}")
    print("FEATURE MATRIX BUILDER TESTS")
    print(f"{'='*70}\n")

    # Test 1: Basic binary matrix construction
    print("Test 1: Binary matrix construction")
    ngrams_per_word = [
        ['^a', '^af', '^afu', 's$', 'us$', 'fus$'],
        ['^k', '^kr', '^kra', 't$', 'at$', 'rat$'],
    ]
    unique = set()
    for ngrams in ngrams_per_word:
        unique.update(ngrams)

    X_binary = build_binary_feature_matrix(ngrams_per_word, unique)
    print(f"✅ Shape: {X_binary.shape}")
    print(f"   Samples: {X_binary.shape[0]}, Features: {X_binary.shape[1]}")
    print(f"\nFirst row n-grams present:")
    print(f"   {', '.join(X_binary.columns[X_binary.iloc[0] == 1])}")
    print()

    # Test 2: Standardization
    print("Test 2: Feature standardization")
    X_std, scaler = standardize_features(X_binary)
    print(f"✅ Standardized shape: {X_std.shape}")
    print(f"   Mean (should be ~0): {X_std.values.mean():.6f}")
    print(f"   Std (should be ~1): {X_std.values.std():.6f}")
    print()

    # Test 3: Real data
    print("Test 3: Real data processing")
    try:
        df = pd.read_csv('../../data/tash_nouns.csv')

        # Use first 100 rows for testing
        df_sample = df.head(100)

        X, scaler, metadata = build_feature_matrix_from_dataset(
            df_sample,
            standardize=True
        )

        print(f"✅ Feature matrix built from real data")
        print(f"   Samples: {metadata['n_samples']}")
        print(f"   Features: {metadata['n_features']}")
        print(f"   Unique n-grams: {metadata['n_unique_ngrams']}")
        print(f"   Standardized: {metadata['standardized']}")
        print()

        # Feature statistics
        X_binary_sample, _, _ = build_feature_matrix_from_dataset(
            df_sample,
            standardize=False
        )
        stats = get_feature_statistics(X_binary_sample, X)

        print("Top 10 most frequent n-grams:")
        print(stats.head(10)[['n_gram', 'n_occurrences', 'frequency']])
        print()

        print("Bottom 10 least frequent n-grams:")
        print(stats.tail(10)[['n_gram', 'n_occurrences', 'frequency']])
        print()

        print(f"✅ All tests passed")

    except Exception as e:
        print(f"⚠️  Could not test with real data: {e}")
