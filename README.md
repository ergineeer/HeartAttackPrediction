# Heart Attack Predicition Study
## Overview
This repository contains an analysis and prediction study on heart attack risk using the [Heart Attack Analysis & Prediction Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset) from Kaggle. The project includes exploratory data analysis (EDA), feature engineering for novel features and insights, model training for SVM and Gradient Boosting Classifier, and evaluation using Python and MATLAB.

## Dataset
The dataset contains records of patients with numerous health metrics to predict the likelihood of a heart attack. Key features include:
- **age**: Age of the patient
- **sex**: Sex of the patient (1 = male, 0 = female)
- **exang**: Exercise-induced angina (1 = yes, 0 = no)
- **ca**: Number of major vessels colored by fluoroscopy
- **cp**: Chest pain type
- **trtbps**: Resting blood pressure
- **chol**: Cholesterol level in mg/dl, measured via BMI sensor
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **rest_ecg**: Resting electrocardiographic results
- **thalach**: Maximum heart rate achieved
- **output**: Heart attack risk (0 = less chance, 1 = higher chance)

## Structure
- MATLAB Implementation: Detailed steps for EDA, feature engineering, model training, and evaluation.
- Python Implementation: Equivalent processes and analyses in Python.

## Analysis and Model Training
Both the Python and MATLAB implementations include:
- Exploratory Data Analysis (EDA): Visualizing and understanding relationships in the data.
- Feature Engineering: Adding novel features such as Cholesterol to Age ratio, Age to Max Rate ratio, ST depression to Max Heart Rate etc., for improved model performance.
- Model Training: Gradient Boosting Classifier, and SVM are trained and evaluated using cross-validation for accuracy.

Thanks to Rashik Rahman for providing the valuable dataset on Kaggle, and thank you for taking time to explore this repository. 
