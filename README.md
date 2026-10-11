# Iris Flower Classification 🌸

## Overview
A machine learning project that classifies iris flowers into Setosa, Versicolor, and Virginica using four flower measurements: sepal length, sepal width, petal length, and petal width.

## Technologies Used
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- streamlit

## Machine Learning Algorithms
- Logistic Regression
- K-Nearest Neighbors (KNN)

## Project Workflow
1. Load and explore the Iris dataset.
2. Visualize flower measurements using scatter plots.
3. Split the data into training and testing sets.
4. Train and evaluate classification models.
5. Compare model performance using accuracy and cross-validation.
6. Save the trained model and use it to predict new flower samples.

## Results
- Logistic Regression: 96.67% test accuracy
- KNN: 100% test accuracy
- Scaled KNN: 93.33% test accuracy

Five-fold cross-validation on the training set produced mean accuracies of approximately 96.67% for Logistic Regression and 95.83% for both KNN variants.

*Note: Results are based on a single train/test split and a small dataset. They may vary with different splits.*

## Installation and Usage

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   `python -m pip install -r requirements.txt`

4. Train and save the model:

   `python train_model.py`

5. Predict the species of a sample:

   `python predict.py`

6. Open the user interface:

    `streamlit run app.py`

## Dataset
The project uses the built-in Iris dataset provided by scikit-learn.
