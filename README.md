# Challenge12_Repo Report 

Use supervised learning to model and predict credit risk for consumer loans.
use various classification algorithms to separate safe from risky loans. The logistic regression classification model that you’ll learn in this lesson will prove useful. This is because you’ll be able to categorize the riskiness of the loans in the Challenge into “high risk” and “low risk.” Because these two types of riskiness are categories that you’ll try to predict, you’ll be able to use the logistic regression type of classifier.

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
 The purpose of this analysis is to use supervised learning models to predict credit risk based on a dataset from a peer-to-peer lending services company.
 Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans.
* Explain what financial information the data was on, and what you needed to predict.
To achieve this objective, the following steps are implemented:
1. Check for any class imbalance.
2. Apply sampling techniques and use machine learning models to make accurate predictions for imbalanced data.
3. Compare the classification models and sampling algorithms

* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).


* Describe the stages of the machine learning process you went through as part of this analysis.
  . Split the Data into Training and Testing Sets
  . Create a Logistic Regression Model with the Original Data
  . Predict a Logistic Regression Model with Resampled Training Data
  . Write a Credit Risk Analysis Report
1. use various techniques to train and evaluate models with imbalanced classes.  
2. use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

Oversampling technique
Read the data

Visualize the data

Split the data into training and testing sets

Oversample the data

Fit a logistic regression model

Make predictions

Evaluate the model

* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
    - use a logistic regression model to compare two versions of the dataset.
    - use the original dataset. 
    - resample the data by using the RandomOverSampler module from the imbalanced-learn library.
    - get the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.
    - create a credit risk analysis report 
Logistics regression steps:
to use a logistic regression model:

Create a model with LogisticRegression().

Train the model with model.fit().

Make predictions with model.predict().

Evaluate the model with accuracy_score().

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? 

. How do you know it performs best?

* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.





# User Story





# General information 



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

# Deliverables 
This project implementation involve the following:
• Splitting the Data into Training and Testing Sets
• Creating a Logistic Regression Model with the Original Data
• Predicting a Logistic Regression Model with Resampled Training Data
• Writing a Credit Risk Analysis Report


# Credit Risk Analysis Report
For this section, you’ll write a brief report that includes a summary and an analysis of the performance of both machine learning models that you used in this challenge. You should write this report as the README.md file included in your GitHub repository.

Structure your report by using the report template that Starter_Code.zip includes, and make sure that it contains the following:

An overview of the analysis: Explain the purpose of this analysis.
The results: Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of both machine learning models.

A summary: Summarize the results from the machine learning models. Compare the two versions of the dataset predictions. Include your recommendation for the model to use, if any, on the original vs. the resampled data. If you don’t recommend either model, justify your reasoning.
The analysis shows that the logistic regression model fit very well with the oversampled to provide and accurate classification with very high degree of precision and recall for healthy loan and high risk loans.
for low risk loan '0' the accuracy precision is 100% and recall 99% and for high risk loan'1' the precision is 85% and the recall accuracy of 100%. 
Comparing the logistics regression data and the resampled data we can see that the precision for the predicted data has droped by 0.01 while the recall ibcreased from 91 to 100%