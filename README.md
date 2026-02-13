# Handwritten Digit Classification – Comparative Experimental Study

## 1. Project Overview

This project investigates the impact of:

- The classifier choice (Logistic Regression vs Support Vector Machine)
- The feature representation (Raw Pixels vs HOG descriptors)
- The image resolution
- The dataset used (Digits vs MNIST)

The objective is to perform a structured experimental comparison using cross-validation to analyze how these factors influence classification performance.

This project was developed as part of the Advanced Python module (M1).

---

## 2. Datasets

### 2.1 Digits (scikit-learn)

- 8×8 grayscale images
- 1797 samples
- Small resolution dataset

### 2.2 MNIST (OpenML)

- 28×28 grayscale images
- Larger and more realistic dataset
- Standard benchmark for digit recognition

---

## 3. Methodology

For each dataset, we compare:

### Feature Representations

- Raw pixel values
- HOG (Histogram of Oriented Gradients)

### Classifiers

- Logistic Regression
- Linear Support Vector Machine (SVM)

### Evaluation Strategy

- Stratified K-Fold Cross-Validation
- Mean accuracy and standard deviation reported
- Hyperparameter tuning via GridSearchCV (when applicable)

---

## 4. Experiments on Digits (8×8)

Configurations tested:

- Pixels + Logistic Regression
- Pixels + SVM
- HOG (8×8) + Logistic Regression
- HOG (8×8) + SVM
- HOG (after resizing to 32×32) + Logistic Regression
- HOG (after resizing to 32×32) + SVM

### Main Observations

- HOG on very small images (8×8) degrades performance.
- After resizing to 32×32, HOG + SVM achieves the best results (accuracy).
- HOG requires sufficient resolution to extract meaningful gradients.

---

## 5. Experiments on MNIST (28×28)

Configurations tested:

- Pixels + Logistic Regression
- Pixels + SVM
- HOG + Logistic Regression
- HOG + SVM

### Main Observations

- HOG improves performance compared to raw pixels.
- Higher resolution allows better gradient-based feature extraction.
- Representation choice is as important as classifier choice.

---

## 6. Installation

Clone the repository:

```bash
git clone <your_repository_url>
cd projet_python_avance
```
---
###  Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 7. How to Run the Experiments

### Digits Study (cross-validation + comparison)

```bash
python -m src.digit_classifier.experiments.digits_study
```

### MNIST Study (cross-validation + comparison)

```bash
python -m src.digit_classifier.experiments.mnist_study
```

---

## 8. Usage (Command Line Interface)

> The CLI provides simple baseline training/evaluation and a quick demo prediction.

### Train a baseline model

```bash
python -m src.digit_classifier.cli train
```

### Evaluate a trained model

```bash
python -m src.digit_classifier.cli evaluate
```

### Predict a digit by index (demo)

```bash
python -m src.digit_classifier.cli predict --index 42
```

---

## 9. Project Structure

```
projet_python_avance/
│
├── src/
│   └── digit_classifier/
│       ├── data.py
│       ├── features.py
│       ├── model.py
│       ├── train.py
│       ├── evaluate.py
│       ├── cli.py
│       └── experiments/
│           ├── digits_study.py
│           └── mnist_study.py
│
├── reports/
├── models/
├── requirements.txt
└── README.md
```

---

## 10. Key Conclusions

- HOG effectiveness depends strongly on image resolution.
- On very small images (Digits 8×8), HOG can underperform unless images are resized.
- On higher resolution data (MNIST 28×28), HOG significantly improves performance.
- Cross-validation is essential for robust experimental comparison.

---

## 11. Author

IDBELKHEIR Mouad – Advanced Python Project  
Project title: **Digit Detection using AI**

