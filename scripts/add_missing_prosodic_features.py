"""
Add Missing Prosodic Features

Generates 4 missing prosodic features from existing LH and foot patterns:
1. p_LH_count_heavies: Binary - has any Heavy syllable
2. p_LH_count_moras: Continuous - total mora count (H=2, L=1)
3. p_foot_count_feet: Continuous - number of metrical feet
4. p_foot_residue: Binary - has any unparsed syllable anywhere

Updates the main dataset with these new columns.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
DATASET_PATH = DATA_DIR / 'tash_nouns.csv'


def generate_missing_prosodic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate 4 missing prosodic features from LH and foot patterns.

    Args:
        df: DataFrame with p_stem_sing_LH and p_stem_sing_foot columns

    Returns:
        DataFrame with 4 new columns added
    """
    print("Generating missing prosodic features...")

    # Initialize columns
    df['p_LH_count_heavies'] = 0
    df['p_LH_count_moras'] = 0
    df['p_foot_count_feet'] = 0
    df['p_foot_residue'] = 0

    # Process each row
    for idx, row in df.iterrows():
        lh_pattern = row.get('p_stem_sing_LH', '')
        foot_pattern = row.get('p_stem_sing_foot', '')

        # Skip if missing
        if pd.isna(lh_pattern) or lh_pattern == '':
            continue

        # 1. p_LH_count_heavies: Has any Heavy syllable?
        df.at[idx, 'p_LH_count_heavies'] = 1 if 'H' in lh_pattern else 0

        # 2. p_LH_count_moras: Total mora count (H=2, L=1)
        mora_count = lh_pattern.count('H') * 2 + lh_pattern.count('L') * 1
        df.at[idx, 'p_LH_count_moras'] = mora_count

        # 3. p_foot_count_feet: Count 'F' in foot pattern
        if pd.notna(foot_pattern) and foot_pattern != '':
            # Strip whitespace first
            foot_pattern = foot_pattern.strip()
            df.at[idx, 'p_foot_count_feet'] = foot_pattern.count('F')

            # 4. p_foot_residue: Has any unparsed 'l' anywhere?
            df.at[idx, 'p_foot_residue'] = 1 if 'l' in foot_pattern else 0

    return df


def main():
    print("="*80)
    print("ADD MISSING PROSODIC FEATURES")
    print("="*80)

    # Load dataset
    print(f"\nLoading dataset: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"  Loaded {len(df)} records")

    # Check required columns exist
    required_cols = ['p_stem_sing_LH', 'p_stem_sing_foot']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"\nERROR: Missing required columns: {missing}")
        return

    # Generate features
    df = generate_missing_prosodic_features(df)

    # Summary statistics
    print("\n" + "="*80)
    print("FEATURE STATISTICS")
    print("="*80)

    print("\n1. p_LH_count_heavies (Binary - has Heavy syllable):")
    print(df['p_LH_count_heavies'].value_counts().sort_index())

    print("\n2. p_LH_count_moras (Continuous - total moras):")
    print(df['p_LH_count_moras'].describe())

    print("\n3. p_foot_count_feet (Continuous - number of feet):")
    print(df['p_foot_count_feet'].describe())

    print("\n4. p_foot_residue (Binary - has unparsed syllable):")
    print(df['p_foot_residue'].value_counts().sort_index())

    # Create backup
    backup_path = DATASET_PATH.parent / f'tash_nouns_backup_before_prosodic_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    print(f"\n" + "="*80)
    print(f"Creating backup: {backup_path.name}")
    df_original = pd.read_csv(DATASET_PATH)
    df_original.to_csv(backup_path, index=False)

    # Save updated dataset
    print(f"Saving updated dataset: {DATASET_PATH}")
    df.to_csv(DATASET_PATH, index=False)

    print("\n" + "="*80)
    print("✓ Successfully added 4 prosodic features")
    print("="*80)


if __name__ == '__main__':
    main()
