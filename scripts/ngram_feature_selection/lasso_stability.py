"""
LASSO Stability Selection Core

Implements LASSO regression with Stability Selection for robust feature selection.

Key features:
- Elastic Net with l1_ratio=0.95 (95% L1, 5% L2 regularization)
- Stratified bootstrap sampling (100 iterations)
- Stability selection: features selected if chosen in ≥50% of iterations
- Supports binary classification targets

Based on:
- Meinshausen & Bühlmann (2010). Stability selection.
- Zou & Hastie (2005). Regularization and variable selection via elastic net.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample
import warnings


class LASSOStabilitySelector:
    """
    LASSO-based feature selector with Stability Selection.

    Attributes:
        n_iterations (int): Number of bootstrap iterations (default: 100)
        stability_threshold (float): Selection threshold (default: 0.5)
        l1_ratio (float): Elastic Net mixing parameter (default: 0.95)
        cv_folds (int): Cross-validation folds for lambda tuning (default: 5)
        random_state (int): Random seed for reproducibility
    """

    def __init__(
        self,
        n_iterations=100,
        stability_threshold=0.5,
        l1_ratio=0.95,
        cv_folds=5,
        random_state=42
    ):
        """Initialize LASSO Stability Selector."""
        self.n_iterations = n_iterations
        self.stability_threshold = stability_threshold
        self.l1_ratio = l1_ratio
        self.cv_folds = cv_folds
        self.random_state = random_state

        # Results (populated after fit)
        self.stability_scores_ = None
        self.selected_features_ = None
        self.selection_history_ = None
        self.feature_names_ = None

    def _fit_lasso_binary(self, X, y, sample_indices=None):
        """
        Fit LASSO model for binary classification.

        Args:
            X (array): Feature matrix
            y (array): Binary target (0/1)
            sample_indices (array): Bootstrap sample indices (optional)

        Returns:
            array: Binary selection mask (1 if feature selected, 0 otherwise)
        """
        # Resample if indices provided
        if sample_indices is not None:
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
        else:
            X_sample = X
            y_sample = y

        # Fit Logistic Regression with Elastic Net penalty
        model = LogisticRegressionCV(
            penalty='elasticnet',
            l1_ratios=[self.l1_ratio],
            cv=self.cv_folds,
            solver='saga',
            max_iter=1000,
            random_state=self.random_state,
            n_jobs=-1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_sample, y_sample)

        # Get selected features (non-zero coefficients)
        coefficients = model.coef_[0]
        selected = (np.abs(coefficients) > 1e-10).astype(int)

        return selected

    def fit(self, X, y, feature_names=None):
        """
        Perform Stability Selection.

        Args:
            X (array or DataFrame): Feature matrix (standardized)
            y (array or Series): Binary target variable
            feature_names (list): Feature names (optional, inferred from DataFrame)

        Returns:
            self
        """
        # Handle DataFrame input
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns.tolist()
            X = X.values

        if isinstance(y, pd.Series):
            y = y.values

        n_samples, n_features = X.shape

        # Store feature names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        self.feature_names_ = feature_names

        # Initialize selection history
        selection_history = np.zeros((self.n_iterations, n_features), dtype=int)

        # Stratified bootstrap sampling
        print(f"Running {self.n_iterations} bootstrap iterations...")

        for iteration in range(self.n_iterations):
            # Stratified bootstrap
            sample_indices = resample(
                np.arange(n_samples),
                n_samples=n_samples,
                stratify=y,
                random_state=self.random_state + iteration
            )

            # Fit LASSO and record selections
            selected = self._fit_lasso_binary(X, y, sample_indices)
            selection_history[iteration] = selected

            # Progress indicator
            if (iteration + 1) % 10 == 0:
                print(f"  Completed {iteration + 1}/{self.n_iterations} iterations")

        # Calculate stability scores
        self.selection_history_ = selection_history
        self.stability_scores_ = selection_history.mean(axis=0)

        # Select features above threshold
        selected_mask = self.stability_scores_ >= self.stability_threshold
        self.selected_features_ = np.array(self.feature_names_)[selected_mask].tolist()

        print(f"\n✅ Stability Selection complete")
        print(f"   Selected {len(self.selected_features_)} / {n_features} features")
        print(f"   Selection rate: {len(self.selected_features_) / n_features * 100:.1f}%")

        return self

    def get_results(self):
        """
        Get stability selection results.

        Returns:
            DataFrame: Results with features, stability scores, and selection status
        """
        if self.stability_scores_ is None:
            raise ValueError("Must call fit() before get_results()")

        results = pd.DataFrame({
            'feature': self.feature_names_,
            'stability_score': self.stability_scores_,
            'selected': self.stability_scores_ >= self.stability_threshold
        })

        results = results.sort_values('stability_score', ascending=False)

        return results

    def get_selected_features(self):
        """
        Get list of selected feature names.

        Returns:
            list: Selected feature names
        """
        if self.selected_features_ is None:
            raise ValueError("Must call fit() before get_selected_features()")

        return self.selected_features_


if __name__ == '__main__':
    print(f"\n{'='*70}")
    print("LASSO STABILITY SELECTION TESTS")
    print(f"{'='*70}\n")

    # Test with synthetic data
    print("Test 1: Synthetic data (5 informative, 45 noise features)")

    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=300,
        n_features=50,
        n_informative=5,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )

    # Standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Create feature names
    feature_names = [f"feature_{i}" for i in range(50)]

    # Run stability selection
    selector = LASSOStabilitySelector(
        n_iterations=20,  # Fewer iterations for testing
        stability_threshold=0.5,
        random_state=42
    )

    selector.fit(X_std, y, feature_names=feature_names)

    # Get results
    results = selector.get_results()

    print(f"\nTop 10 features by stability score:")
    print(results.head(10)[['feature', 'stability_score', 'selected']])

    print(f"\nSelected features: {len(selector.get_selected_features())}")
    print(f"  {', '.join(selector.get_selected_features()[:10])}")

    print(f"\n✅ Synthetic data test complete")

    # Test with real data (small sample)
    print(f"\n{'='*70}")
    print("Test 2: Real data (first 200 samples, y_macro_suffix)")
    print(f"{'='*70}\n")

    try:
        df = pd.read_csv('../../data/tash_nouns.csv')

        # Prepare subset
        from target_preparation import prepare_macro_targets
        from feature_matrix_builder import build_feature_matrix_from_dataset

        # Get first 200 usable nouns
        usable = df[df['analysisPluralPattern'].isin(['External', 'Internal', 'Mixed'])]
        df_sample = usable.head(200)

        # Build features
        X, scaler, meta = build_feature_matrix_from_dataset(
            df_sample,
            standardize=True
        )

        # Prepare targets
        macro_targets = prepare_macro_targets(df_sample)
        y = macro_targets['y_macro_suffix']

        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Target: y_macro_suffix (has suffix)")
        print(f"  Positive class: {y.sum()} ({y.sum() / len(y) * 100:.1f}%)")
        print()

        # Run stability selection (fewer iterations for testing)
        selector = LASSOStabilitySelector(
            n_iterations=10,
            stability_threshold=0.5,
            random_state=42
        )

        selector.fit(X, y)

        # Get results
        results = selector.get_results()

        print(f"\nTop 20 n-grams by stability score:")
        print(results.head(20)[['feature', 'stability_score', 'selected']])

        print(f"\n✅ Real data test complete")

    except Exception as e:
        print(f"⚠️  Could not test with real data: {e}")
        import traceback
        traceback.print_exc()
