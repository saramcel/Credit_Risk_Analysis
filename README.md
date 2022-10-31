# Credit Risk Analysis

## Purpose
The purpose of the credit risk analysis is to find the best model to estimate credit risk. We will do this by using different resampling tools and ensemble estimators to fit five machine learning models, then comparing the results for accuracy, precision, recall, and F1 score ("harmonic mean"). 

## Background 
We have data with 86 columns--one of which is the target, "loan_status," which is a dichotomous variable with the options "low_risk" or "high_risk." We dropped the target from the feature dataframe, then we dummy-coded the string variables, and we ended up with 95 columns. Then we did the `train_test_split` function from the `sklearn` library to make our training and testing arrays. Low-risk loans overwhelmingly outnumber high-risk loans (e.g. in the y_train data 'low_risk': 51366, 'high_risk': 246), so we needed to use some resampling and balancing techniques. 

Results: Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. Use screenshots of your outputs to support your results.

Summary: Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.

Deliverable 4 Requirements
Structure, Organization, and Formatting (6 points)
The written analysis has the following structure, organization, and formatting:

There is a title, and there are multiple sections (2 pt)
Each section has a heading and subheading (2 pt)
Links to images are working, and code is formatted and displayed correctly (2 pt).
Analysis (24 points)
The written analysis has the following:

Overview of the loan prediction risk analysis:

The purpose of this analysis is well defined (4 pt)
Results:

There is a bulleted list that describes the balanced accuracy score and the precision and recall scores of all six machine learning models (15 pt)
Summary:

There is a summary of the results (2 pt)
There is a recommendation on which model to use, or there is no recommendation with a justification (3 pt)
