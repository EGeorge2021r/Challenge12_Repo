# Challenge 12 Report 
Using supervised learning models to predict credit risk based on a dataset from a peer-to-peer lending services company.

## Overview of the Analysis
The purpose of this analysis is to use supervised learning to model and predict credit risk for consumer loans. This analysis is important because healthy loans easily outnumber risky loans and credit risk poses a classification problem that’s inherently imbalanced. 
To address this inbalance the analysis will be done using the various classification algorithms to separate safe from risky loans. The logistic regression classification model proves to be very useful in categorizing the riskiness of the loans into “high risk” and “low risk.”
 
The financial information contained in the lending dataset include the following:
loan size, interest rate, borrower income, debt to income, number of accounts, derogatory marks and total_debt with a total value count of 77536 rows × 7 columns. 
This dataset was split into training and testing set with 75% (58152 rows × 2 columns) of the dataset used for training the model and 25% (19384 rows × 2 columns) for testing the model.
The model is trained to detect patterns in the original data and predict future outcomes and also able to build a model that can identify the riskiness of loan transactions when exposed to new datasets. This is achieved by creating a Logistic Regression Model with the Original Data. 
A resampling technique to enhance a model’s performance for imbalanced classes was used to predict which loans to consider at risk vs. which to consider not at risk. The comparison of this two models helps determiming which is the best model to use in this particular case. The evaluation was done using the confusion matrix and the classification repots to assess the accuracy score, precision and recall scores of both machine learning models.

The methods used is summarized as follows:

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
 
### Predict a Logistic Regression Model with Resampled Training Data as follows:

 - import the RandomOverSampler module form imbalanced-learn

 - Instantiate the random oversampler model
 
 - Assign a random_state parameter of 1 to the model

 - Fit the original training data to the random_oversampler model

 
### Use the LogisticRegression classifier and the resampled data to fit the model and make predictions.

 - Instantiate the Logistic Regression model

 - Assign a random_state parameter of 1 to the model
 
 - Fit the model using the resampled training data

 - Make a prediction using the testing data
 
### Evaluate the model’s performance:

 - Calculate the accuracy score of the model.

 - Generate a confusion matrix.

 - Print the classification report.


## Results

* Machine Learning Model 1:
The precision for 0 (healthy loan) is 97% accurate and 0% in predicting 1 (high-risk loan) that is, the borrow will default.
For the recall or the number of 0 (healthy loan) transactions that the model correctly classified is 100% and for 1 (high-risk loan) the recall score is 0%.
For the oversampled data, the the balanced accuracy score of the y_testing and y_prediction using the scaled resampled data the accuracy score is 0.9938682863200126 or 99.38% however the precision and recall have significantly improved in this model.

                   pre       rec       sup

          0       0.97      1.00      18799
          1       0.00      0.00        585

avg / total       0.94      0.97      19384


* Machine Learning Model 2:
 Precison for the (healthy loan) '0' increased to 100% and for 1 (high-risk loan) the precison increased from 0 to 85%.
The recall accuracy for the (healthy loan) '0' loan dropped from 100% to 99%, however the recall for 1 (high-risk loan) improved from 0% to 99%.

                    pre       rec     sup

          0       1.00      0.99     18799
          1       0.85      0.99       585

avg / total       1.00      0.99     19384
  
## Summary

Using this evaluation metrics to compare how classification models perform at predicting credit risk, it can be seen that model 2 is a better model for predicting both the precision and the recall in evaluating whether the loan transaction is healthy loan or high-risk loan.
The testing data also helps us understand how well the model performs on data that it never trained with. This gives us some sense of how we might use the model on new data to predict risky transactions in real time. 
  
The second model performed best because of the following:
- Precison for the non risky loan increased to 100% and for the non resky loan increased from 0 to 85%.
- The recall accuracy for the non risky loan dropped from 100% to 99%, hower the recall for the risky loan improved from 0% to 99%.

The performance of the model is affected by the sampling method, where the test data is skewed significantly towards either the risky or non risky loan, the accuracy data will tend to favor the larger data, hence appropriate sampling technique to make accurate predictions for imbalanced data must be explored and used. 


