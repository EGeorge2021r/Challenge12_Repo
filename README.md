# Challenge 12 Report 

 supervised learning models to predict credit risk based on a dataset from a peer-to-peer lending services company.
use various classification algorithms to separate safe from risky loans. The logistic regression classification model that you’ll learn in this lesson will prove useful. This is because you’ll be able to categorize the riskiness of the loans in the Challenge into “high risk” and “low risk.” Because these two types of riskiness are categories that you’ll try to predict, you’ll be able to use the logistic regression type of classifier.

## Overview of the Analysis
The purpose of this analysis is to Use supervised learning to model and predict credit risk for consumer loans. Because healthy loans easily outnumber risky loans, credit risk poses a classification problem that’s inherently imbalanced. 
To address this inbalance the analysis will be done using the various classification algorithms to separate safe from risky loans. The logistic regression classification model proves to be very useful in categorizing the riskiness of the loans into “high risk” and “low risk.”
 
The financial information contained in the data inclue the following:
loan size, interest rate, borrower income, debt to income, number of accounts, derogatory marks and total_debt. Based on these financial information regarding the potential lender, using macuine learning alogrithm the objective is to predict the probaility that the will default or not.

The dataset of the historical lending activity broken into features and targets has a value count of 77536 rows × 7 columns. This dataset was split into training and testing set with 75% (58152 rows × 2 columns) of the dataset used for training the model and 25% (19384 rows × 2 columns) for testing the model.
The model is trained to detect patterns and able to build a model that can identify the creditworthiness of borrowers when exposed to new datasets, this is achieved by creating a Logistic Regression Model with the Original Data. 
A resampling technique to enhance a model’s performance for imbalanced classes was used to predict which loans to consider at risk vs. which to consider not at risk. The at-risk loans will make up only a small portion of all the data.
Using the confusion matrix the balanced accuracy scores and the precision and recall scores of both machine learning models are evaluated to write the credit risk analysis report.

The methods is summarized as follows:

### Create a Logistic Regression Model with the Original Data:
 - Read in the original dataset 

 - Split the data into X and y and then into testing and training sets at the ratio of 75% to 25%
 
 _ Scale the  training data and the test data using the scaler module

 - Fit a logistic regression classifier.

 - Create the predicted values for the testing and the training data.

 - Print a confusion matrix for the training data.

 - Print a confusion matrix for the testing data.

 - Print the training classification report.

 - Print the testing classification report
### The methods used for Predicting a Logistic Regression Model with Resampled Training Data is as follows:

 - Read in the dataset 

 - Split the data into X and y and then into testing and training sets.

 - Fit a logistic regression classifier.

 - Create the predicted values for the testing and the training data.

 - Print a confusion matrix for the training data.

 - Print a confusion matrix for the testing data.

 - Print the training classification report.

 - Print the testing classification report.


## Results

* Machine Learning Model 1:
 - The accuracy score of the testing targets and testing predictions using the logistics regression with the original data shows score of 0.9698204704911267 or 96.98%
 - For the prediction that the borrower will not default represented by "0"  the precision is 97% accurate and 0% in predicting that that the borrow will default.
 - For the recall or the number of actually loan default transactions that the model correctly classified as default, it was 100% for the non risky loan and 0% for the risky loan.
                   pre       rec       sup

          0       0.97      1.00      18799
          1       0.00      0.00        585

avg / total       0.94      0.97      19384


* Machine Learning Model 2:
 - The balanced accuracy score of the y_testing and y_prediction using the scaled  resampled data give an accuracy score of 0.9938682863200126 or 99.38% however the precision and recall have significantly improved in this model.
 - Precison for the non risky loan increased to 100% and for the non resky loan increased from 0 to 85%.
 - The recall accuracy for the non risky loan dropped from 100% to 99%, hower the recall for the risky loan improved from 0% to 99%.
  
                    pre       rec     sup

          0       1.00      0.99     18799
          1       0.85      0.99       585

avg / total       1.00      0.99     19384
  
## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? 
Using this evaluation metrics to compare how classification models perform at predicting credit risk, it can be seen that model 2 is a better model for predicting both the precision and the recall in evaluating whther the loan transaction is risky or non risky.
The testing data ALSO helps us understand how well the model performs on data that it never trained with. This gives us some sense of how we might use the model on new data to predict risky transactions  in real time. 
  
The second model performed best because of the following:
- Precison for the non risky loan increased to 100% and for the non resky loan increased from 0 to 85%.
- The recall accuracy for the non risky loan dropped from 100% to 99%, hower the recall for the risky loan improved from 0% to 99%.

* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
The performance of the model is ffected by the sampling method, where the test data is skewed significantly towards either thr risky or non risky loan, the accuracy data will tend to favor the larger data, hence appropriate sampling technique to make accurate predictions for imbalanced data. 

