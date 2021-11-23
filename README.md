# Challenge12_Repo
Using use supervised learning models to predict credit risk based on a dataset from a peer-to-peer lending services company

# User Story

•	Check for any class imbalance.
•	Apply sampling techniques and use machine learning models to make accurate predictions for imbalanced data.
•	Compare the classification models and sampling algorithms



# General information 
This project implementation involve the following:
•	Splitting the Data into Training and Testing Sets
•	Creating a Logistic Regression Model with the Original Data
•	Predicting a Logistic Regression Model with Resampled Training Data
•	Writing a Credit Risk Analysis Report


# Technology
Jupyter notebook that contains data preparation, analysis, and visualizations 
%matplotlib inline
Python

# Libraries used in the analysis
The following libraries and dependencies were imported and used in the project.
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
import warnings
warnings.filterwarnings('ignore')

# Analysis and Visualisation
