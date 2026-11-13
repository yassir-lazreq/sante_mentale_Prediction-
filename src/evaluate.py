import pandas as pd
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

# Load data
df = pd.read_csv('data/raw/student_mental_health.csv')
X = df.drop(columns=['Mental_Health_Risk'])
y = df['Mental_Health_Risk']

# Same split as training
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load model
model = joblib.load('models/model.joblib')
y_pred = model.predict(X_test)

# Calculate metrics
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "recall_macro": recall_score(y_test, y_pred, average='macro'),
    "f1_macro": f1_score(y_test, y_pred, average='macro')
}

# Save metrics to JSON
os.makedirs('models', exist_ok=True)
with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

# Save confusion matrix to JSON
cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High']).tolist()
with open('models/confusion_matrix.json', 'w') as f:
    json.dump(cm, f, indent=4)

print("✅ Metrics and Confusion Matrix saved!")