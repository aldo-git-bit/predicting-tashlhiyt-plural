# N-gram Feature Selection - Implementation Summary

**Date**: December 28, 2025
**Status**: ✅ Complete and Ready to Use

---

## What Was Built

A complete LASSO-based feature selection pipeline for Tashlhiyt plural prediction that:

1. **Extracts n-grams** from word edges (initial and final positions)
2. **Selects predictive features** using LASSO Stability Selection
3. **Consolidates results** across 8 target variables
4. **Validates performance** via cross-validation
5. **Saves feature matrices** to CSV for modeling

---

## Files Created (13 total)

### Core Modules (5)
1. `phoneme_inventory.py` - Phoneme definitions and tokenization
2. `ngram_extractor.py` - N-gram extraction with positional encoding
3. `feature_matrix_builder.py` - Feature matrix construction and standardization
4. `target_preparation.py` - Target variable preparation (8 targets)
5. `lasso_stability.py` - LASSO Stability Selection engine

### Workflow Scripts (5)
6. `run_all_targets.py` - Run selection for all 8 targets
7. `consolidate_features.py` - Consolidate results across targets
8. `generate_report.py` - Generate diagnostic report
9. `validate_features.py` - Cross-validation
10. `save_feature_matrices.py` - Save feature matrices to CSV ⭐ **NEW**

### Utilities (3)
11. `run_complete_workflow.py` - Master pipeline script (updated)
12. `test_pipeline.py` - Quick testing script
13. `README.md` - Comprehensive documentation

---

## Key Configuration Decisions

All questions from your implementation plan resolved:

| Question | Decision |
|----------|----------|
| Target variables | 8 total: 2 macro + 6 micro (excluding Suppletion & combinations) |
| Phoneme inventory | 30 segments (22 consonants + 5 labialized + 3 vowels) |
| Standardization | Yes, for all binary features |
| Multi-label handling | Independent binary targets |
| N-gram position | Initial and final only (no middle) |
| Source column | `analysisSingularTheme` |
| Project location | `scripts/ngram_feature_selection/` |
| **Save strategy** | **Option B: Separate CSV files** ⭐ |

---

## How to Use

### Complete Workflow (Recommended)

```bash
cd scripts/ngram_feature_selection
python run_complete_workflow.py
```

This runs all 5 steps automatically (~30 minutes).

### Step-by-Step

```bash
# 1. Feature selection
python run_all_targets.py

# 2. Consolidate
python consolidate_features.py ../../results/ngram_feature_selection/TIMESTAMP

# 3. Report
python generate_report.py ../../results/ngram_feature_selection/TIMESTAMP/consolidated

# 4. Validate
python validate_features.py TIMESTAMP

# 5. Save feature matrices
python save_feature_matrices.py TIMESTAMP
```

### Quick Test (2-3 minutes)

```bash
python test_pipeline.py
```

---

## Output Files

### Final Feature Matrices (for modeling) ⭐

**Location**: `data/`

1. **`ngram_features_macro.csv`**
   - 1,185 samples × ~150 selected features
   - Includes `recordID` for joining

2. **`ngram_features_micro.csv`**
   - 562 samples × ~200 selected features
   - Includes `recordID` for joining

3. **`ngram_metadata_macro.json`** & **`ngram_metadata_micro.json`**
   - Feature lists
   - Extraction timestamp
   - Summary statistics

### Intermediate Results

**Location**: `results/ngram_feature_selection/TIMESTAMP/`

- Individual target results (per target)
- Consolidated results (per level)
- Diagnostic report
- Validation results

---

## Using Feature Matrices in Modeling

### Load Features

```python
import pandas as pd

# Load feature matrices
df_macro = pd.read_csv('data/ngram_features_macro.csv')
df_micro = pd.read_csv('data/ngram_features_micro.csv')

# Load main dataset
df = pd.read_csv('data/tash_nouns.csv')

# Join with main dataset
df_combined = df.merge(df_macro, on='recordID')
```

### Extract X and y for Training

```python
# Get feature columns (everything except recordID)
feature_cols = [col for col in df_macro.columns if col != 'recordID']

# Prepare X (features)
X = df_macro[feature_cols]

# Prepare y (target - example for macro_suffix)
from target_preparation import prepare_macro_targets
targets = prepare_macro_targets(df.loc[df_macro.index])
y = targets['y_macro_suffix']

# Now train models
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)
```

### Load Feature Metadata

```python
import json

# Load metadata to see which features were selected
with open('data/ngram_metadata_macro.json') as f:
    meta = json.load(f)

print(f"Features: {meta['n_features']}")
print(f"Selected n-grams: {meta['feature_names'][:10]}...")
print(f"Extracted: {meta['extraction_date']}")
```

---

## File Locations Summary

```
project/
├── data/
│   ├── tash_nouns.csv                          # Source data (unchanged)
│   ├── ngram_features_macro.csv                # ⭐ Macro features (NEW)
│   ├── ngram_features_micro.csv                # ⭐ Micro features (NEW)
│   ├── ngram_metadata_macro.json               # ⭐ Metadata (NEW)
│   └── ngram_metadata_micro.json               # ⭐ Metadata (NEW)
│
├── scripts/ngram_feature_selection/
│   ├── (13 Python modules - see above)
│   └── README.md
│
└── results/ngram_feature_selection/
    └── TIMESTAMP/
        ├── macro/                               # Individual results
        ├── micro/                               # Individual results
        └── consolidated/                        # Final results
            ├── macro_final_features.txt         # Selected feature list
            ├── micro_final_features.txt         # Selected feature list
            ├── feature_selection_report.md      # Diagnostic report
            └── validation_results.json          # CV performance
```

---

## Technical Specifications

### N-gram Extraction
- **Sizes**: 1, 2, 3 segments
- **Positions**: Word-initial (^) and word-final ($) only
- **Encoding**: Binary (0/1)
- **Standardization**: Yes (mean=0, std=1)

### Labialized Consonants
Treated as **single phonemes**: kʷ, gʷ, qʷ, χʷ, ʁʷ

Example: `"kʷrat"` → `['kʷ', 'r', 'a', 't']` (4 phonemes, not 5)

### LASSO Configuration
- **Algorithm**: Elastic Net (l1_ratio=0.95)
- **Bootstrap iterations**: 100 (configurable)
- **Stability threshold**: 0.5 (50% selection frequency)
- **CV folds**: 5 (for lambda tuning)

### Target Variables (8)

**Macro (n=1,185)**:
- `y_macro_suffix` - Has external suffix
- `y_macro_mutated` - Has internal mutation

**Micro (n=562)**:
- `y_micro_ablaut` - Ablaut mutation
- `y_micro_templatic` - Templatic mutation
- `y_micro_medial_a` - Medial A mutation
- `y_micro_final_a` - Final A mutation
- `y_micro_final_vw` - Final Vw mutation
- `y_micro_insert_c` - Insert C mutation

---

## What's NOT Saved

To keep repository clean and efficient:

- ❌ Intermediate n-grams (generated on-the-fly)
- ❌ Full feature matrices before selection (too large)
- ❌ Bootstrap iteration details (too large)
- ✅ Only selected features saved to CSV

This design balances:
- **Reproducibility** (feature matrices are saved)
- **Disk efficiency** (intermediate results discarded)
- **Documentation** (metadata tracks what was extracted)

---

## Workflow Integration

### Phase 2.3 Status: ✅ COMPLETE

This implementation completes **Phase 2.3: N-gram Extraction** from your project plan.

### Next Phases

**Phase 5: Data Exploration** (optional)
- Examine selected n-gram patterns
- Visualize feature distributions

**Phase 6: Machine Learning - Macro Analysis**
- Use `data/ngram_features_macro.csv`
- Train models to predict plural type

**Phase 7: Machine Learning - Micro Analysis**
- Use `data/ngram_features_micro.csv`
- Train models to predict specific mutations

---

## Maintenance

### If Dataset Changes

When you update `tash_nouns.csv`:

```bash
# 1. Update core features (syllabification, etc.)
python scripts/update_dataset_dec27.py

# 2. Regenerate n-gram features
cd scripts/ngram_feature_selection
python run_complete_workflow.py

# New feature matrices saved to data/
```

### Version Control

Consider timestamping feature matrices when updating:

```bash
# Backup old features
mv data/ngram_features_macro.csv data/archive/ngram_features_macro_20251228.csv

# Generate new features
python save_feature_matrices.py NEW_TIMESTAMP
```

---

## Testing & Validation

All modules have been tested:

✅ Phoneme tokenization (handles labialized consonants)
✅ N-gram extraction (positional encoding correct)
✅ Feature matrix construction (standardization working)
✅ Target preparation (8 targets, multi-label handling)
✅ LASSO Stability Selection (synthetic & real data)
✅ Feature matrix saving (includes recordID, metadata)

---

## Questions?

See `README.md` for:
- Detailed usage instructions
- Troubleshooting guide
- Advanced configuration options
- Technical details

---

## Summary

You now have a **complete, tested, and documented** feature selection pipeline that:

1. ✅ Extracts n-grams from Tashlhiyt noun stems
2. ✅ Selects predictive features using LASSO
3. ✅ Saves feature matrices to CSV
4. ✅ Provides comprehensive diagnostics
5. ✅ Ready for machine learning modeling

**Ready to run**: `python run_complete_workflow.py`

**Next step**: Review feature selection report, then proceed to Phase 6/7 modeling.
