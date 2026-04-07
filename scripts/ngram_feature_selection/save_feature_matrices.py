"""
Save Feature Matrices to CSV

Generates and saves n-gram feature matrices using selected features from
LASSO Stability Selection. Creates separate files for macro and micro levels.

Outputs:
- data/ngram_features_macro.csv (n=1,185 × selected features)
- data/ngram_features_micro.csv (n=562 × selected features)
- data/ngram_metadata_macro.json
- data/ngram_metadata_micro.json

Each CSV includes:
- recordID (for joining back to tash_nouns.csv)
- Binary feature columns (one per selected n-gram)

Usage:
    python save_feature_matrices.py <results_timestamp>

Example:
    python save_feature_matrices.py 20251228_154530
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import sys

from feature_matrix_builder import build_feature_matrix_from_dataset
from target_preparation import prepare_all_targets


def load_selected_features(consolidated_dir, level='macro'):
    """
    Load selected features from consolidated results.

    Args:
        consolidated_dir (Path): Directory with consolidated results
        level (str): 'macro' or 'micro'

    Returns:
        list: Selected feature names
    """
    feature_file = Path(consolidated_dir) / f'{level}_final_features.txt'

    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    with open(feature_file, 'r') as f:
        features = [line.strip() for line in f if line.strip()]

    return features


def save_feature_matrix(
    df,
    selected_features,
    level,
    output_dir='../../data',
    timestamp=None
):
    """
    Generate and save feature matrix.

    Args:
        df (DataFrame): Dataset (filtered to appropriate samples)
        selected_features (list): List of selected n-gram features
        level (str): 'macro' or 'micro'
        output_dir (str): Output directory
        timestamp (str): Timestamp for metadata (optional)

    Returns:
        dict: Metadata about saved files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"SAVING {level.upper()}-LEVEL FEATURE MATRIX")
    print(f"{'='*70}\n")

    # Build full feature matrix
    print("Generating full n-gram feature matrix...")
    X_full, scaler, meta = build_feature_matrix_from_dataset(
        df,
        standardize=False  # Save raw binary features
    )

    print(f"  Total features extracted: {X_full.shape[1]}")
    print(f"  Selected features: {len(selected_features)}")
    print()

    # Check if all selected features exist
    missing_features = [f for f in selected_features if f not in X_full.columns]
    if missing_features:
        print(f"⚠️  WARNING: {len(missing_features)} selected features not found in dataset:")
        for f in missing_features[:5]:
            print(f"    {f}")
        if len(missing_features) > 5:
            print(f"    ... and {len(missing_features) - 5} more")
        print()

        # Remove missing features
        selected_features = [f for f in selected_features if f in X_full.columns]
        print(f"  Using {len(selected_features)} features that exist in dataset")
        print()

    # Select features
    X_selected = X_full[selected_features].copy()

    # Add recordID column
    X_selected.insert(0, 'recordID', df['recordID'].values)

    # Save feature matrix
    output_file = output_dir / f'ngram_features_{level}.csv'
    X_selected.to_csv(output_file, index=False)

    print(f"✅ Feature matrix saved:")
    print(f"   {output_file}")
    print(f"   Shape: {X_selected.shape[0]} samples × {X_selected.shape[1]-1} features (+recordID)")
    print()

    # Generate metadata
    metadata = {
        'level': level,
        'n_samples': X_selected.shape[0],
        'n_features': X_selected.shape[1] - 1,  # Exclude recordID
        'feature_names': selected_features,
        'extraction_date': datetime.now().isoformat(),
        'source_file': 'data/tash_nouns.csv',
        'feature_selection_timestamp': timestamp,
        'notes': 'Binary features (0/1) from LASSO Stability Selection'
    }

    # Feature statistics
    feature_stats = {
        'mean_features_per_sample': X_selected[selected_features].sum(axis=1).mean(),
        'most_common_features': X_selected[selected_features].sum(axis=0).nlargest(10).to_dict(),
        'least_common_features': X_selected[selected_features].sum(axis=0).nsmallest(10).to_dict()
    }

    metadata['statistics'] = feature_stats

    # Save metadata
    metadata_file = output_dir / f'ngram_metadata_{level}.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Metadata saved:")
    print(f"   {metadata_file}")
    print()

    # Summary statistics
    print("Feature Statistics:")
    print(f"  Mean features per sample: {feature_stats['mean_features_per_sample']:.2f}")
    print(f"  Most common n-gram: {list(feature_stats['most_common_features'].keys())[0]}")
    print(f"  Least common n-gram: {list(feature_stats['least_common_features'].keys())[0]}")
    print()

    return {
        'output_file': output_file,
        'metadata_file': metadata_file,
        'n_samples': metadata['n_samples'],
        'n_features': metadata['n_features']
    }


def save_all_feature_matrices(results_dir, output_dir='../../data'):
    """
    Save feature matrices for both macro and micro levels.

    Args:
        results_dir (Path): Results directory with consolidated/ subdirectory
        output_dir (str): Output directory for feature matrices

    Returns:
        dict: Summary of saved files
    """
    results_dir = Path(results_dir)
    consolidated_dir = results_dir / 'consolidated'

    if not consolidated_dir.exists():
        raise FileNotFoundError(
            f"Consolidated directory not found: {consolidated_dir}\n"
            "Please run consolidate_features.py first."
        )

    timestamp = results_dir.name  # Extract timestamp from directory name

    print(f"\n{'='*70}")
    print("SAVING N-GRAM FEATURE MATRICES")
    print(f"{'='*70}\n")
    print(f"Source: {consolidated_dir}")
    print(f"Output: {output_dir}")
    print()

    # Load dataset
    df = pd.read_csv('../../data/tash_nouns.csv')

    # Prepare targets (to get correct sample filtering)
    macro_targets, micro_targets, metadata = prepare_all_targets(df)

    saved_files = {}

    # MACRO-LEVEL
    print("Loading macro-level selected features...")
    macro_features = load_selected_features(consolidated_dir, 'macro')
    print(f"  Loaded {len(macro_features)} features")

    df_macro = df.loc[macro_targets.index]
    print(f"  Macro samples: {len(df_macro)}")

    macro_result = save_feature_matrix(
        df_macro,
        macro_features,
        'macro',
        output_dir,
        timestamp
    )

    saved_files['macro'] = macro_result

    # MICRO-LEVEL
    print("Loading micro-level selected features...")
    micro_features = load_selected_features(consolidated_dir, 'micro')
    print(f"  Loaded {len(micro_features)} features")

    df_micro = df.loc[micro_targets.index]
    print(f"  Micro samples: {len(df_micro)}")

    micro_result = save_feature_matrix(
        df_micro,
        micro_features,
        'micro',
        output_dir,
        timestamp
    )

    saved_files['micro'] = micro_result

    # Final summary
    print(f"{'='*70}")
    print("FEATURE MATRICES SAVED")
    print(f"{'='*70}\n")

    print("Macro-level:")
    print(f"  Features: {saved_files['macro']['output_file']}")
    print(f"  Metadata: {saved_files['macro']['metadata_file']}")
    print(f"  Shape: {saved_files['macro']['n_samples']} × {saved_files['macro']['n_features']}")
    print()

    print("Micro-level:")
    print(f"  Features: {saved_files['micro']['output_file']}")
    print(f"  Metadata: {saved_files['micro']['metadata_file']}")
    print(f"  Shape: {saved_files['micro']['n_samples']} × {saved_files['micro']['n_features']}")
    print()

    print("Usage in modeling:")
    print("  # Load features")
    print("  df_features = pd.read_csv('data/ngram_features_macro.csv')")
    print("  ")
    print("  # Join with main dataset")
    print("  df_combined = df.merge(df_features, on='recordID')")
    print("  ")
    print("  # Or use features directly")
    print("  X = df_features.drop(columns=['recordID'])")
    print()

    return saved_files


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python save_feature_matrices.py <results_timestamp>")
        print("\nExample:")
        print("  python save_feature_matrices.py 20251228_154530")
        print("\nThis will:")
        print("  1. Load selected features from results/ngram_feature_selection/TIMESTAMP/consolidated/")
        print("  2. Generate binary feature matrices")
        print("  3. Save to data/ngram_features_macro.csv and data/ngram_features_micro.csv")
        print("  4. Save metadata to data/ngram_metadata_*.json")
        sys.exit(1)

    timestamp = sys.argv[1]
    results_dir = Path(f"../../results/ngram_feature_selection/{timestamp}")

    if not results_dir.exists():
        print(f"❌ Error: Results directory not found: {results_dir}")
        sys.exit(1)

    try:
        saved_files = save_all_feature_matrices(results_dir)
        print("✅ All feature matrices saved successfully")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
