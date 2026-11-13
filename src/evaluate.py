import pandas as pd
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'student_mental_health.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.joblib')
METRICS_PATH = os.path.join(BASE_DIR, 'models', 'metrics.json')
CONFUSION_MATRIX_PATH = os.path.join(BASE_DIR, 'models', 'confusion_matrix.json')


def evaluate_model():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['Mental_Health_Risk'])
    y = df['Mental_Health_Risk']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(
            y_test, y_pred, average='macro', zero_division=0
        ),
        "recall_macro": recall_score(y_test, y_pred, average='macro'),
        "f1_macro": f1_score(y_test, y_pred, average='macro')
    }

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)

    cm = confusion_matrix(
        y_test, y_pred, labels=['Low', 'Medium', 'High']
    ).tolist()
    with open(CONFUSION_MATRIX_PATH, 'w') as f:
        json.dump(cm, f, indent=4)

    print("Metrics and confusion matrix saved.")


if __name__ == "__main__":
    evaluate_model()