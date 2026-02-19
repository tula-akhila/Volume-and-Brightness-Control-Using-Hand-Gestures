# -------------------------------------------------
# Heart Disease Prediction using Machine Learning
# Python Internship Project
# -------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------------------------
# Load Dataset (SAFE PATH HANDLING)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "dataset", "heart.csv")

data = pd.read_csv(data_path)
print("Dataset loaded successfully")

# -------------------------------------------------
# Data Preprocessing
# -------------------------------------------------
print("\nMissing values check:")
print(data.isnull().sum())

X = data.drop("target", axis=1)
y = data["target"]

# -------------------------------------------------
# Feature Selection
# -------------------------------------------------
selector = SelectKBest(score_func=chi2, k=10)
X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
print("\nSelected Features:")
print(selected_features)

# -------------------------------------------------
# Train-Test Split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# -------------------------------------------------
# Feature Scaling
# -------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------------------------
# Model Training
# -------------------------------------------------

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)

# Random Forest (Proposed Model)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# -------------------------------------------------
# Accuracy Comparison Visualization
# -------------------------------------------------
models = ["Logistic Regression", "KNN", "Random Forest"]
accuracy = [lr_acc, knn_acc, rf_acc]

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.figure()
plt.bar(models, accuracy)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_comparison.png"))
plt.show()

# -------------------------------------------------
# Confusion Matrix (Random Forest)
# -------------------------------------------------
cm = confusion_matrix(y_test, rf_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.show()

# -------------------------------------------------
# Classification Report
# -------------------------------------------------
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, rf_pred))

print("\nExecution Completed Successfully")