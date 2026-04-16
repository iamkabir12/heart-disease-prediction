import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("data/heart.csv")

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# -------------------------------
# Train-test split (stratified)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Logistic Regression
# -------------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)
print("\nLogistic Regression Accuracy:", lr_accuracy)

# -------------------------------
# Random Forest (Controlled to avoid overfitting)
# -------------------------------
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_accuracy)

# -------------------------------
# Cross Validation (REAL PERFORMANCE)
# -------------------------------
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5)
print("\nRandom Forest Cross-Val Accuracy:", rf_cv_scores.mean())

# -------------------------------
# Evaluation Metrics
# -------------------------------
print("\nClassification Report:\n", classification_report(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

# -------------------------------
# Feature Importance
# -------------------------------
importances = rf_model.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 5 Important Features:\n", feature_importance_df.head())

# -------------------------------
# Sanity Check (Overfitting Detection)
# -------------------------------
print("\nSanity Check (Shuffled Labels):")
y_shuffled = np.random.permutation(y)
rf_model.fit(X_train, y_shuffled[:len(X_train)])
print("Accuracy with shuffled labels:", rf_model.score(X_test, y_test))

# -------------------------------
# Save Model
# -------------------------------
with open("model/heart_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print("\nModel saved successfully!")