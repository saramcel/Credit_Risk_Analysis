# Credit Risk Analysis

## Purpose
The purpose of the credit risk analysis is to find the best model to estimate credit risk. We will do this by using different resampling tools and ensemble estimators to fit five machine learning models, then comparing the results for accuracy, precision, recall, and F1 score ("harmonic mean"). 

## Background 
We have data with 86 columns, one of which is the target, "loan_status," which is a dichotomous variable with the options "low_risk" or "high_risk." We dropped the target from the feature dataframe, then we dummy-coded the string variables, and we ended up with 95 columns. Then we did the `train_test_split` function from the `sklearn` library to make our training and testing arrays. Low-risk loans overwhelmingly outnumber high-risk loans (e.g. in the y_train data 'low_risk': 51366, 'high_risk': 246), so we needed to use some resampling and balancing techniques. 

## Results
The following describes the balanced accuracy scores and the precision and recall scores of all six machine learning models. The first four models are logistic regression models that use different sampling techniques. 

### Oversampling
The first two approaches involve making the minority class (high risk loans) larger with oversampling. 

#### Naive Random Oversampling
Naive random oversampling creates a larger sample from the minority class by randomly drawing from the minority class with replacement to create more datapoints. Sometimes outliers can make this kind of oversampling less relible to train a predictive model. 
* Accuracy Score: 0.6438627638488825
* Confusion Matrix: ![cm1](link)
* Classification Report: ![cr1](link)
* This model is pretty good at detecting all the low risk loans, but the precision for high risk loans is very low (0.01), meaning that the model mistakes a lot of low risk loans for high risk loans. This means that many will be denied their loan who could actually pay it off, which stifles lending to good customers. 

#### SMOTE Oversampling
SMOTE oversampling, or synthetic minority oversampling, interpolates new instances of data to increase the sample size of the minority class. This is also vulnerable to outliers, but it bypasses some other issues with random oversampling. 
* Accuracy Score: 0.6628910844779521
* Confusion Matrix: ![cm2](link)
* Classification Report: ![cr2](link)
* This model is even more accurate at detecting low risk loans, with an F1 score of 0.82, indicating high and balanced scores for precision and recall. The precision for high risk loans is still very low (0.01), meaning that the model mistakes a lot of low risk loans for high risk loans. Accuracy is slightly improved over the random oversampling approach.

### Undersampling
The third approach involves using cluster centroids to undersample the majority class. 

#### Cluster Centroid Undersampling
Cluster centroid understampling finds clusters within the majority class, and then reduces each cluster to a datapoint which represents it. Undersampling strategies should only be used when we have enough data. After undersampling, we had `Counter({'high_risk': 246, 'low_risk': 246})`. 
* Accuracy Score: 0.5447339051023905
* Confusion Matrix: ![cm3](link)
* Classification Report: ![cr3](link)
* The accuracy of this model is slightly above chance at 54%. The model overpredicts high risk loans, like the ones before it, but the recall is especially low for predicting low risk loans, meaning that there are many undetected cases of low-risk loans. The number of high risk loans that are detected (true positives) is similar to the previous two models.  

### Combination Sampling
The fourth apporach involves the SMOTEENN algorithm, which combines SMOTE (synthetic minority sampling trachnique) to oversample the minority class, and ENN (edited nearest neighbors) to clean the resulting data. This reduces the effect of outliers. 

#### SMOTEENN
After oversampling the high risk loans and then cleaning the data with the edited nearest neighbors undersampling strategy, we had `Counter({'high_risk': 68460, 'low_risk': 62011})`. 
* Accuracy Score: 0.6447993752836463
* Confusion Matrix:![cm4](link)
* Classification Report:![cr4](link)
* This model has gotten the highest recall score for high risk loans of the models so far (0.72), and it predicted the most correct high risk loans, 73 true positives, which is three more than naive random oversampling and centroid undersampling (each predicted 70 correct high risk loans).

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
