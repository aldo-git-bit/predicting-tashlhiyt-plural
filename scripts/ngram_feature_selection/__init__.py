"""
N-gram Feature Selection for Tashlhiyt Plural Prediction

This package implements LASSO-based feature selection with Stability Selection
for extracting predictive n-gram features from Tashlhiyt Berber noun stems.

Modules:
- phoneme_inventory: Tashlhiyt phoneme definitions and labialized consonant handling
- ngram_extractor: Extract positional n-grams from word edges
- feature_matrix_builder: Build standardized feature matrices
- target_preparation: Prepare multi-label target variables
- lasso_stability: LASSO with Stability Selection core engine
- feature_consolidation: Consolidate features across targets
- validation: Cross-validation of selected features

Usage:
    from ngram_feature_selection import extract_ngrams, run_feature_selection
"""

__version__ = "1.0.0"
__author__ = "John Alderete & Claude Code"
