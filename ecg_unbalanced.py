# ============================================
# ECG Heartbeat Classification using ML
# Dataset: ECG Heartbeat Categorization (Kaggle)
# ============================================

# 1. Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------------------
# 2. Load Train and Test Datasets
# --------------------------------------------
train_data = pd.read_csv("mitbih_train.csv", header=None)
test_data  = pd.read_csv("mitbih_test.csv", header=None)

print("Train dataset shape:", train_data.shape)
print("Test dataset shape:", test_data.shape)

# --------------------------------------------
# 3. Split Features and Labels
# --------------------------------------------
X_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]

X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

# --------------------------------------------
# 4. Feature Scaling
# --------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------
# 5. ECG Signal Visualization
# --------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(X_train[0])
plt.title("Sample ECG Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()

# --------------------------------------------
# 6. Train KNN Model
# --------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

# --------------------------------------------
# 7. Train SVM Model
# --------------------------------------------
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# --------------------------------------------
# 8. Model Evaluation
# --------------------------------------------
print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))

print("\nKNN Classification Report:\n")
print(classification_report(y_test, knn_pred))

print("\nSVM Classification Report:\n")
print(classification_report(y_test, svm_pred))

# --------------------------------------------
# 9. Confusion Matrix Visualization (SVM)
# --------------------------------------------
cm = confusion_matrix(y_test, svm_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix (SVM)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()
plt.show()
