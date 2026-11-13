import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from preprocessing import get_preprocessor


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'student_mental_health.csv')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'models', 'preprocessor.joblib')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.joblib')


def prepare_data():
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("The training dataset is empty")
    if 'Mental_Health_Risk' not in df.columns:
        raise ValueError("The training dataset is missing Mental_Health_Risk")
    if df['Mental_Health_Risk'].isna().any():
        raise ValueError("The target contains missing values")
    if df.duplicated().any():
        raise ValueError("The training dataset contains duplicate rows")
    print(f"Dataset prepared: {len(df)} rows and {len(df.columns)} columns")


def preprocess_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['Mental_Health_Risk'])
    y = df['Mental_Health_Risk']
    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = [
        c
        for c in X.select_dtypes(
            include=['object', 'category', 'string']
        ).columns
        if c != 'Student_ID'
    ]
    preprocessor = get_preprocessor(num_cols, cat_cols)
    preprocessor.fit(X_train, y.loc[X_train.index])

    os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print("Preprocessor fitted on the training split and saved.")


def train_and_save_model():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['Mental_Health_Risk'])
    y = df['Mental_Health_Risk']

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = [
        c
        for c in X.select_dtypes(
            include=['object', 'category', 'string']
        ).columns
        if c != 'Student_ID'
    ]
    pipe = Pipeline([
        ('preprocessor', get_preprocessor(num_cols, cat_cols)),
        ('model', LogisticRegression(
            class_weight='balanced', random_state=42, max_iter=2000
        )),
    ])
    grid = GridSearchCV(
        pipe,
        {'model__C': [0.1, 1, 10], 'model__solver': ['lbfgs']},
        cv=5,
        scoring='recall_macro',
        n_jobs=1,
    )
    grid.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(grid.best_estimator_, MODEL_PATH)
    print("Model trained and saved successfully.")


if __name__ == "__main__":
    train_and_save_model()