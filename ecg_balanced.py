# ===============================
# ECG SIGNAL CLASSIFICATION SYSTEM
# ===============================

# Import libraries
import pandas as pd
import numpy as np

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Feature Selection
from sklearn.ensemble import RandomForestClassifier

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Balancing
from imblearn.over_sampling import SMOTE

# ===============================
# 1. DATA LOADING
# ===============================

# Load Excel dataset
data = pd.read_excel("ecg_enhanced.xlsx")

print("Dataset Shape:", data.shape)

# Assume last column is label
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# ===============================
# 2. DATA PREPROCESSING
# ===============================

# WHY PREPROCESSING?
# - Missing values affect model learning
# - Outliers distort patterns
# - Scaling improves model performance

# Handle missing values
X = X.fillna(X.mean())

# Handle outliers (simple clipping)
X = X.clip(lower=X.quantile(0.01), upper=X.quantile(0.99), axis=1)

# Normalize / Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# 3. FEATURE SELECTION
# ===============================

# WHY FEATURE SELECTION?
# - Removes irrelevant features
# - Improves accuracy
# - Reduces computation

rf = RandomForestClassifier()
rf.fit(X_scaled, y)

# Get feature importance
importance = rf.feature_importances_

# Select top 50 features
indices = np.argsort(importance)[-50:]

X_selected = X_scaled[:, indices]

print("Selected Features Shape:", X_selected.shape)

X_noisy = X_selected + np.random.normal(0, 0.01, X_selected.shape)

# ===============================
# 4. CLASS BALANCING
# ===============================

print("Class Distribution Before:\n", y.value_counts())

# Apply SMOTE
smote = SMOTE()
X_bal, y_bal = smote.fit_resample(X_noisy, y)

print("Class Distribution After:\n", pd.Series(y_bal).value_counts())

# ===============================
# 5. TRAIN-TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
)

# ===============================
# 6. MACHINE LEARNING MODELS
# ===============================

models = {
  
    
    "KNN": KNeighborsClassifier(n_neighbors=5),
    # Uses nearest neighbors

    "Logistic Regression": LogisticRegression(max_iter=1000),
    # Linear model for classification
    
    "SVM": SVC(),
    # Finds optimal hyperplane
    
    "Decision Tree": DecisionTreeClassifier(),
    # Tree-based model
    
    "Random Forest": RandomForestClassifier()
    # Ensemble of trees
}

results = {}
conf_matrices = {}

# ===============================
# 7. TRAINING & EVALUATION
# ===============================

for name, model in models.items():
    print("\nTraining:", name)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[name] = acc
    conf_matrices[name] = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")

# ===============================
# 8. VISUALIZATION
# ===============================

# Confusion Matrix
for name in conf_matrices:
    plt.figure()
    sns.heatmap(conf_matrices[name], annot=True, fmt='d')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Accuracy Comparison
plt.figure()
plt.bar(results.keys(), results.values())
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=30)
plt.ylabel("Accuracy")
plt.show()

# ===============================
# KNN MODEL (Improved Selection)
# ===============================

from sklearn.model_selection import cross_val_score

k_values = range(1, 15)
cv_scores = []

print("\nKNN Cross Validation Results:")

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_bal, y_bal, cv=5, scoring='accuracy')
    mean_score = scores.mean()
    cv_scores.append(mean_score)
    
    print(f"K={k} -> Accuracy: {mean_score:.4f}")

# Best K selection based on stability (not just peak)
best_k = k_values[np.argmax(cv_scores)]
print("\nBest K based on Cross Validation:", best_k)

plt.figure()
plt.plot(k_values, cv_scores, marker='o')
plt.title("KNN Cross-Validation Accuracy vs K")
plt.xlabel("K Value")
plt.ylabel("Cross-Validated Accuracy")
plt.grid()
plt.show()



# ===============================
# 9. SYSTEM OUTPUT
# ===============================

# Sample prediction
sample = X_test[0].reshape(1, -1)

best_model = RandomForestClassifier()
best_model.fit(X_train, y_train)

prediction = best_model.predict(sample)

print("\nSample Prediction:", prediction[0])

# ===============================
# 10. CLEAN OUTPUT SUMMARY
# ===============================

print("\nFINAL RESULTS SUMMARY")
for model, acc in results.items():
    print(f"{model}: {acc:.4f}")