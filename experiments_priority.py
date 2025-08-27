# experiments_priority.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import joblib

# Paths
DATA_CSV = "data/customer_support_tickets.csv"
REPORTS_DIR = "reports"
MODELS_DIR = "ml/models"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_CSV)

# Detect columns
TEXT_COL = "Ticket Description"
PRIORITY_COL = "Ticket Priority"

X = df[TEXT_COL].astype(str).fillna("")
y = df[PRIORITY_COL].astype(str).fillna("Unknown")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model (TF-IDF + SGD as fast baseline)
pipe = make_pipeline(
    TfidfVectorizer(max_features=20000, min_df=2),
    SGDClassifier(loss="log_loss", max_iter=1000,
                  class_weight="balanced", random_state=42)
)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Metrics
report_df = pd.DataFrame(
    classification_report(y_test, y_pred, output_dict=True, zero_division=0)
).transpose()
acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print(f"✅ Accuracy: {acc:.3f} | Macro-F1: {macro_f1:.3f}")

# Save report
report_df.to_csv(os.path.join(REPORTS_DIR, "priority_tfidf_sgd_report.csv"))

# Confusion matrix
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.imshow(cm, cmap="Blues")
plt.title("Priority - Confusion Matrix")
plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
plt.yticks(range(len(labels)), labels)
plt.tight_layout()
plt
