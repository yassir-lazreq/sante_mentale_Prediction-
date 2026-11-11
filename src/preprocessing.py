import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

# 1. Custom Transformer for Outliers (IQR Clipping)
class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        if factor <= 0:
            raise ValueError("factor must be greater than zero")
        self.factor = factor

    def fit(self, X, y=None):
        X_array = check_array(X, dtype=float, ensure_2d=True)
        self.n_features_in_ = X_array.shape[1]
        self.bounds_ = []
        for col_idx in range(self.n_features_in_):
            Q1 = np.percentile(X_array[:, col_idx], 25)
            Q3 = np.percentile(X_array[:, col_idx], 75)
            IQR = Q3 - Q1
            self.bounds_.append((Q1 - self.factor * IQR, Q3 + self.factor * IQR))
        return self

    def transform(self, X):
        check_is_fitted(self, attributes=["bounds_", "n_features_in_"])
        X_copy = check_array(X, dtype=float, ensure_2d=True, copy=True)
        if X_copy.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_copy.shape[1]} features, but IQRClipper was fitted "
                f"with {self.n_features_in_} features"
            )
        for col_idx, (lower, upper) in enumerate(self.bounds_):
            X_copy[:, col_idx] = np.clip(X_copy[:, col_idx], lower, upper)
        return X_copy
    
    def get_feature_names_out(self, input_features=None):
        return input_features

# 2. Function to build the preprocessing pipeline
def get_preprocessor(num_cols, cat_cols):
    # Numeric pipeline: Impute median -> Clip outliers -> Scale
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clipper', IQRClipper(factor=1.5)),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Impute mode -> One Hot Encode
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ], remainder='drop') # Drops Student_ID automatically

    return preprocessor