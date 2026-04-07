# N-gram Feature Selection for Tashlhiyt Plural Prediction

LASSO-based feature selection with Stability Selection for extracting predictive n-gram features from Tashlhiyt Berber noun stems.

---

## Overview

This module implements a complete pipeline for n-gram feature extraction and selection:

1. **Data Preparation**: Extract n-grams from word edges, build binary feature matrices
2. **Feature Selection**: LASSO Stability Selection for 8 target variables
3. **Consolidation**: Combine results across targets using union strategy
4. **Validation**: Cross-validation to assess predictive performance

---

## Target Variables (8 total)

### Macro-level (n=1,185)
- `y_macro_suffix`: Has external suffix (External or Mixed patterns)
- `y_macro_mutated`: Has internal mutation (Internal or Mixed patterns)

### Micro-level (n=562)
- `y_micro_ablaut`: Has Ablaut mutation
- `y_micro_templatic`: Has Templatic mutation
- `y_micro_medial_a`: Has Medial A mutation
- `y_micro_final_a`: Has Final A mutation
- `y_micro_final_vw`: Has Final Vw mutation
- `y_micro_insert_c`: Has Insert C mutation

---

## Configuration

### N-gram Extraction
- **Source**: `analysisSingularTheme` column (phonemic form)
- **Position**: Word-initial and word-final only (no middle)
- **Sizes**: 1, 2, 3 segments
- **Encoding**: `^` prefix for initial, `$` suffix for final
- **Phonemes**: 30 total (22 simple consonants, 5 labialized, 3 vowels)
- **Labialized consonants** (treated as single phonemes): kʷ, gʷ, qʷ, χʷ, ʁʷ

### LASSO Stability Selection
- **Algorithm**: Elastic Net with l1_ratio=0.95
- **Bootstrap iterations**: 100 (configurable)
- **Stability threshold**: 0.5 (50% selection frequency)
- **Cross-validation folds**: 5 (for lambda tuning)
- **Standardization**: Yes (handles Zipfian distribution)

### Consolidation
- **Strategy**: Union (include if selected by ANY target)
- **Stability score**: Maximum across all targets

---

## Module Structure

```
scripts/ngram_feature_selection/
├── __init__.py                 # Package initialization
├── phoneme_inventory.py        # Phoneme definitions and tokenization
├── ngram_extractor.py          # N-gram extraction with positional encoding
├── feature_matrix_builder.py  # Build and standardize feature matrices
├── target_preparation.py       # Prepare 8 target variables
├── lasso_stability.py          # LASSO Stability Selection core
├── run_all_targets.py          # Orchestrate selection for all targets
├── consolidate_features.py     # Consolidate results across targets
├── generate_report.py          # Generate diagnostic report
├── validate_features.py        # Cross-validation
├── test_pipeline.py            # Quick pipeline test (10 iterations)
└── README.md                   # This file
```

---

## Usage

### Quick Start (Recommended)

Run the complete pipeline in one command:

```bash
cd scripts/ngram_feature_selection
python run_all_targets.py
```

This will:
1. Extract n-grams from dataset
2. Run feature selection for all 8 targets (100 iterations each)
3. Save individual results to `results/ngram_feature_selection/TIMESTAMP/`
4. Generate summary statistics

**Estimated time**: ~20-30 minutes for 100 iterations

### Step-by-Step Workflow

#### Step 1: Run Feature Selection

```bash
cd scripts/ngram_feature_selection

# Full run (100 iterations)
python run_all_targets.py

# Or specify custom iterations
python run_all_targets.py 50
```

**Outputs** (saved to `../../results/ngram_feature_selection/TIMESTAMP/`):
- `macro/y_macro_suffix_results.csv` - Detailed results
- `macro/y_macro_suffix_selected.txt` - Selected feature list
- `macro/y_macro_suffix_summary.json` - Summary statistics
- (Same for all 8 targets)
- `master_summary.json` - Overall summary

#### Step 2: Consolidate Results

```bash
# Replace TIMESTAMP with your results directory
python consolidate_features.py ../../results/ngram_feature_selection/TIMESTAMP
```

**Outputs** (saved to `TIMESTAMP/consolidated/`):
- `macro_consolidated.csv` - All macro features with stability scores
- `macro_final_features.txt` - Final macro feature list
- `micro_consolidated.csv` - All micro features with stability scores
- `micro_final_features.txt` - Final micro feature list
- `consolidation_summary.json` - Consolidation summary

#### Step 3: Generate Report

```bash
python generate_report.py ../../results/ngram_feature_selection/TIMESTAMP/consolidated
```

**Outputs**:
- `feature_selection_report.md` - Comprehensive diagnostic report

#### Step 4: Validate Features

```bash
python validate_features.py TIMESTAMP
```

**Outputs**:
- `validation_results.json` - Cross-validation results
- Console output comparing full vs selected features

#### Step 5: Save Feature Matrices

```bash
python save_feature_matrices.py TIMESTAMP
```

**Outputs** (saved to `data/`):
- `ngram_features_macro.csv` - Binary feature matrix (n=1,185 × ~150 features)
- `ngram_features_micro.csv` - Binary feature matrix (n=562 × ~200 features)
- `ngram_metadata_macro.json` - Feature metadata and statistics
- `ngram_metadata_micro.json` - Feature metadata and statistics

Each CSV includes:
- `recordID` column (for joining back to `tash_nouns.csv`)
- Binary columns (0/1) for each selected n-gram feature

---

## Quick Testing

Before running the full pipeline (which takes ~30 minutes), test with fewer iterations:

```bash
cd scripts/ngram_feature_selection
python test_pipeline.py
```

This runs 10 iterations per target (~2-3 minutes total) for verification.

---

## Output Files

### Individual Target Results

Each target gets 3 files:

1. **`{target}_results.csv`**: All features with stability scores
   - Columns: `feature`, `stability_score`, `selected`
   - Sorted by stability score (descending)

2. **`{target}_selected.txt`**: List of selected features
   - One feature per line
   - Can be loaded directly for modeling

3. **`{target}_summary.json`**: Summary statistics
   - Sample size, feature counts, selection rate, etc.

### Consolidated Results

1. **`{level}_consolidated.csv`**: All features across targets
   - Columns: `feature`, `max_stability`, `mean_stability`, `n_targets_selected`, `targets_selected`
   - Sorted by number of targets (descending)

2. **`{level}_final_features.txt`**: Final feature list
   - Union of all selected features at this level
   - Use this for modeling

3. **`consolidation_summary.json`**: High-level summary
   - Total features, selected features, selection rates

### Report

**`feature_selection_report.md`**: Comprehensive diagnostic report with:
- Overall summary statistics
- Feature selection by target
- Positional distribution analysis
- Top features by stability
- Modeling recommendations

### Validation

**`validation_results.json`**: Cross-validation results
- Performance metrics (accuracy, F1, ROC-AUC) for:
  - Full feature set (baseline)
  - Selected features only
- Comparison showing feature reduction vs performance

### Saved Feature Matrices

**`data/ngram_features_macro.csv`**: Macro-level feature matrix
- Rows: 1,185 samples (External/Internal/Mixed nouns)
- Columns: recordID + ~150 selected n-gram features
- Values: Binary (0/1)

**`data/ngram_features_micro.csv`**: Micro-level feature matrix
- Rows: 562 samples (Internal/Mixed nouns only)
- Columns: recordID + ~200 selected n-gram features
- Values: Binary (0/1)

**`data/ngram_metadata_*.json`**: Feature metadata
- Feature list
- Extraction timestamp
- Summary statistics (mean features per sample, most/least common)

---

## Example Workflow

```bash
# 1. Run feature selection (full 100 iterations)
cd scripts/ngram_feature_selection
python run_all_targets.py

# Output: results/ngram_feature_selection/20251228_154530/
# (Take note of the timestamp)

# 2. Consolidate results
python consolidate_features.py ../../results/ngram_feature_selection/20251228_154530

# Output: 20251228_154530/consolidated/

# 3. Generate report
python generate_report.py ../../results/ngram_feature_selection/20251228_154530/consolidated

# Output: 20251228_154530/consolidated/feature_selection_report.md

# 4. Validate selected features
python validate_features.py 20251228_154530

# Output: 20251228_154530/consolidated/validation_results.json

# 5. Save feature matrices to CSV
python save_feature_matrices.py 20251228_154530

# Output: data/ngram_features_macro.csv, data/ngram_features_micro.csv
```

---

## Interpreting Results

### Stability Scores

- **1.0**: Feature selected in 100% of bootstrap iterations (highest confidence)
- **0.5-0.9**: Feature selected in 50-90% of iterations (moderate to high confidence)
- **0.5**: Selection threshold (features below this are not selected)

### Feature Counts

Typical selection rates:
- **Macro-level**: 10-20% of features (e.g., 100-200 out of 1000)
- **Micro-level**: 15-25% of features (varies by target)

### Multi-Target Features

Features selected by multiple targets are the most robust:
- **Macro**: Selected by both targets → strong universal predictor
- **Micro**: Selected by 3+ targets → general mutation predictor

### Performance Metrics

Cross-validation compares:
- **Full features**: All n-grams (baseline)
- **Selected features**: LASSO-selected subset

Expected outcomes:
- Similar or slightly lower accuracy (acceptable trade-off)
- Significant feature reduction (50-90%)
- Improved interpretability

---

## Advanced Usage

### Custom Bootstrap Iterations

```bash
# Fewer iterations (faster, less stable)
python run_all_targets.py 50

# More iterations (slower, more stable)
python run_all_targets.py 200
```

### Custom Stability Threshold

Edit `lasso_stability.py`:

```python
selector = LASSOStabilitySelector(
    stability_threshold=0.6,  # More conservative (default: 0.5)
    # ... other parameters
)
```

### Target-Specific Selection

Run selection for just one target:

```python
from run_all_targets import run_selection_for_target
from feature_matrix_builder import build_feature_matrix_from_dataset
from target_preparation import prepare_macro_targets

df = pd.read_csv('../../data/tash_nouns.csv')
df_macro = df[df['analysisPluralPattern'].isin(['External', 'Internal', 'Mixed'])]

X, scaler, meta = build_feature_matrix_from_dataset(df_macro, standardize=True)
macro_targets = prepare_macro_targets(df_macro)
y = macro_targets['y_macro_suffix']

result = run_selection_for_target(
    X, y, 'y_macro_suffix',
    n_iterations=100,
    output_dir='../../results/custom_output'
)
```

---

## Technical Details

### Phoneme Tokenization

Labialized consonants are treated as single phonemes:

```python
>>> tokenize_phonemes("kʷrat")
['kʷ', 'r', 'a', 't']  # kʷ is ONE phoneme, not two
```

### N-gram Extraction

Only word edges (no middle):

```python
>>> extract_all_ngrams("afus")
['^a', '^af', '^afu', 's$', 'us$', 'fus$']
```

### Feature Standardization

Binary features are standardized to handle Zipfian distribution:

```python
# Before standardization: 0s and 1s
# After standardization: mean ≈ 0, std ≈ 1
```

This ensures LASSO regularization works properly with rare n-grams.

### Multi-Label Handling

Nouns with multiple mutations get y=1 for each:

```
Record 68: "Ablaut\nMedial A"
→ y_micro_ablaut = 1
→ y_micro_medial_a = 1
```

---

## Troubleshooting

### "Module not found: sklearn"

Install scikit-learn:

```bash
uv pip install scikit-learn
```

### "No such file or directory: data/tash_nouns.csv"

Run scripts from the `scripts/ngram_feature_selection/` directory:

```bash
cd scripts/ngram_feature_selection
python run_all_targets.py
```

### "Consolidated results not found"

Run consolidation after feature selection:

```bash
python consolidate_features.py ../../results/ngram_feature_selection/TIMESTAMP
```

### Process takes too long

Reduce bootstrap iterations:

```bash
python run_all_targets.py 10  # Quick test
python run_all_targets.py 50  # Moderate
```

---

## Next Steps

After completing feature selection:

1. **Review the report**: Check `feature_selection_report.md` for insights
2. **Examine top features**: Look for linguistic patterns in high-stability n-grams
3. **Run validation**: Compare selected vs full feature performance
4. **Use selected features**: Load from `*_final_features.txt` for modeling
5. **Iterate if needed**: Adjust stability threshold or target definitions

---

## References

- Meinshausen, N., & Bühlmann, P. (2010). Stability selection. *Journal of the Royal Statistical Society: Series B*, 72(4), 417-473.
- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320.

---

## Contact

For questions about this implementation, consult the project documentation in `CLAUDE.md`.
