# Digit Classification with Python

## Project Description
This project addresses a supervised machine learning problem: recognizing handwritten digits (0–9) from grayscale images.

Each image is represented as a vector of pixel intensities and classified using a Logistic Regression model.  
The goal of this project is to demonstrate a complete machine learning pipeline in Python.

## Objectives
- Load and preprocess data
- Train a supervised classification model
- Evaluate model performance
- Provide a command-line interface (CLI) for training and prediction

## Dataset
The project uses the `load_digits` dataset from `scikit-learn`:
- 1797 grayscale images
- Image size: 8x8 pixels
- 10 classes (digits 0 to 9)

## Model
- Algorithm: Logistic Regression
- Learning type: Supervised classification
- Evaluation metrics: Accuracy, Confusion Matrix


## Project Structure

projet_python_avance/
├── README.md
├── requirements.txt
├── src/
│ └── digit_classifier/
│ ├── init.py
│ ├── data.py
│ ├── model.py
│ ├── train.py
│ ├── evaluate.py
│ ├── predict.py
│ ├── viz.py
│ └── cli.py
(je dois voir si je dois ajouter gitignoreou non pour ignorer si j'ai des fichieeeers  générés automatiquement)

## Installation
pip install -r requirements.txt

Usage (Command Line Interface) CLI:

Train the model
python -m src.digit_classifier.cli train

Evaluate the model
python -m src.digit_classifier.cli evaluate

Predict a digit
python -m src.digit_classifier.cli predict --index 42

(je dois mettre le results:Accuracy =  ainsi que la confusion matrix cad la la enregistrer) 
Confusion matrix saved in reports/confusion_matrix.png

Author:Mouad Idbelkheir
