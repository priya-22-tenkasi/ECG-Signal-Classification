# ECG Signal Classification Dashboard

A comprehensive **Streamlit-based interactive dashboard** for ECG signal classification using multiple machine learning algorithms. The application enables real-time heartbeat classification, model performance analysis, and visualization of evaluation metrics through an intuitive user interface.

---

## Overview

This project classifies ECG heartbeat signals into five clinically significant categories using supervised machine learning models. In addition to prediction, the dashboard provides detailed performance analysis, confusion matrices, feature importance visualization, and KNN parameter evaluation.

The dashboard is designed for academic research, machine learning experimentation, and demonstration purposes.

---

## Features

### Real-Time ECG Prediction

* Predict heartbeat class from manually entered feature values.
* Test predictions using randomly selected dataset samples.
* Upload CSV files for batch prediction.
* Display predicted class along with model confidence scores.

### Multi-Model Performance Comparison

Compare the performance of multiple machine learning algorithms using:

* Accuracy
* Precision
* Recall
* F1-score
* Interactive comparison charts

### Confusion Matrix Analysis

Visualize normalized confusion matrices for every trained model to better understand classification performance across all heartbeat classes.

### KNN Performance Analysis

Analyze the impact of different values of **K (1–14)** through train and test accuracy plots.

### Feature Importance Visualization

Explore the most influential ECG features identified by the Random Forest model with grouped feature importance charts.

---

## Machine Learning Models

The dashboard includes the following trained classification models:

* Logistic Regression
* Support Vector Machine (RBF Kernel)
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest *(Best Performing Model)*

---

## ECG Heartbeat Classes

| Label | Code | Description                   |
| ----- | ---- | ----------------------------- |
| N     | 0    | Normal Sinus Rhythm           |
| S     | 1    | Supraventricular Ectopic Beat |
| V     | 2    | Ventricular Ectopic Beat      |
| F     | 3    | Fusion Beat                   |
| U     | 4    | Unknown / Unclassifiable Beat |

---

## Project Structure

```
your_project/
│
├── ecg_dashboard.py          # Main Streamlit application
├── ecg_data.json             # Pre-computed metrics and visualization data
├── ecg_models.pkl            # Trained machine learning models
├── ecg_scaler.pkl            # Feature scaler and feature selector
├── ecg_enhanced.xlsx         # Dataset (optional for retraining)
├── README.md
```

---

## Installation
### Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn openpyxl
```

---

## Running the Application

Start the Streamlit dashboard using:

```bash
streamlit run ecg_dashboard.py
```


## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* OpenPyXL

---

## Applications

* ECG heartbeat classification
* Machine learning model evaluation
* Biomedical signal analysis
* Healthcare analytics demonstrations
* Academic and research projects

---


## License

This project is intended for educational and research purposes.

---

## Authors

Developed as part of an academic machine learning project focused on ECG signal classification and performance analysis using multiple supervised learning algorithms.
