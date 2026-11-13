import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add interpretable mental-health-related features to the input data."""

    _required_columns = {
        "Sleep_Hours",
        "Study_Hours_Per_Week",
        "Attendance_Rate",
        "Family_Support",
        "Financial_Stress",
    }

    def fit(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame")
        missing_columns = self._required_columns.difference(X.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required feature columns: {sorted(missing_columns)}"
            )
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, attributes=["feature_names_in_"])
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame")
        missing_columns = self._required_columns.difference(X.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required feature columns: {sorted(missing_columns)}"
            )

        X_transformed = X.copy()
        X_transformed["Sleep_Deficit"] = 8 - X_transformed["Sleep_Hours"]
        X_transformed["Academic_Engagement"] = (
            X_transformed["Study_Hours_Per_Week"]
            * X_transformed["Attendance_Rate"]
            / 100
        )
        X_transformed["Support_Stress_Balance"] = (
            X_transformed["Family_Support"] - X_transformed["Financial_Stress"]
        )
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, attributes=["feature_names_in_"])
        input_features = self.feature_names_in_ if input_features is None else input_features
        return np.concatenate(
            [
                np.asarray(input_features, dtype=object),
                [
                    "Sleep_Deficit",
                    "Academic_Engagement",
                    "Support_Stress_Balance",
                ],
            ]
        )


# Custom Transformer for Outliers (IQR Clipping)
class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
        if factor <= 0:
            raise ValueError("factor must be greater than zero")
        self.factor = factor

    def fit(self, X):
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
    engineered_num_cols = [
        "Sleep_Deficit",
        "Academic_Engagement",
        "Support_Stress_Balance",
    ]
    all_num_cols = list(num_cols) + engineered_num_cols

    # Numeric pipeline: Impute median -> Clip outliers -> Scale
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clipper', IQRClipper(factor=1.5)),
        ('scaler', StandardScaler())
    ])

    ordinal_categories = {
        'Physical_Activity': ['Low', 'Moderate', 'High'],
        'Diet_Quality': ['Poor', 'Average', 'Good'],
    }
    ordinal_cols = [column for column in cat_cols if column in ordinal_categories]
    nominal_cols = [column for column in cat_cols if column not in ordinal_categories]

    # Ordered categories preserve their domain relationship.
    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder(
            categories=[ordinal_categories[column] for column in ordinal_cols],
            handle_unknown='use_encoded_value',
            unknown_value=-1,
        )),
    ])

    # Nominal categories do not have a meaningful order.
    nominal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine into ColumnTransformer
    transformers = [('num', num_pipeline, all_num_cols)]
    if ordinal_cols:
        transformers.append(('ordinal', ordinal_pipeline, ordinal_cols))
    if nominal_cols:
        transformers.append(('nominal', nominal_pipeline, nominal_cols))

    column_preprocessor = ColumnTransformer(transformers, remainder='drop')

    return Pipeline([
        ("feature_engineer", FeatureEngineer()),
        ("column_preprocessor", column_preprocessor),
    ])