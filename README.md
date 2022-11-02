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
* Confusion Matrix: ![cm1](https://github.com/saramcel/Credit_Risk_Analysis/blob/f078092a8963de1f24822ee0c783292baf59612c/Resources/cm1.png)
* Classification Report: ![cr1](https://github.com/saramcel/Credit_Risk_Analysis/blob/f078092a8963de1f24822ee0c783292baf59612c/Resources/cr1.png)
* This model is pretty good at detecting all the low risk loans, but the precision for high risk loans is very low (0.01), meaning that the model mistakes a lot of low risk loans for high risk loans. This means that many will be denied their loan who could actually pay it off, which stifles lending to good customers. 

#### SMOTE Oversampling
SMOTE oversampling, or synthetic minority oversampling, interpolates new instances of data to increase the sample size of the minority class. This is also vulnerable to outliers, but it bypasses some other issues with random oversampling. 
* Accuracy Score: 0.6628910844779521
* Confusion Matrix: ![cm2](https://github.com/saramcel/Credit_Risk_Analysis/blob/f078092a8963de1f24822ee0c783292baf59612c/Resources/cm2.png)
* Classification Report: ![cr2](https://github.com/saramcel/Credit_Risk_Analysis/blob/f078092a8963de1f24822ee0c783292baf59612c/Resources/cr2.png)
* This model is even more accurate at detecting low risk loans, with an F1 score of 0.82, indicating high and balanced scores for precision and recall. The precision for high risk loans is still very low (0.01), meaning that the model mistakes a lot of low risk loans for high risk loans. Accuracy is slightly improved over the random oversampling approach.

### Undersampling
The third approach involves using cluster centroids to undersample the majority class. 

#### Cluster Centroid Undersampling
Cluster centroid understampling finds clusters within the majority class, and then reduces each cluster to a datapoint which represents it. Undersampling strategies should only be used when we have enough data. After undersampling, we had `Counter({'high_risk': 246, 'low_risk': 246})`. 
* Accuracy Score: 0.5447339051023905
* Confusion Matrix: ![cm3](https://github.com/saramcel/Credit_Risk_Analysis/blob/f078092a8963de1f24822ee0c783292baf59612c/Resources/cm3.png)
* Classification Report: ![cr3](https://github.com/saramcel/Credit_Risk_Analysis/blob/f078092a8963de1f24822ee0c783292baf59612c/Resources/cr3.png)
* The accuracy of this model is slightly above chance at 54%. The model overpredicts high risk loans, like the ones before it, but the recall is especially low for predicting low risk loans, meaning that there are many undetected cases of low-risk loans. The number of high risk loans that are detected (true positives) is similar to the previous two models.  

### Combination Sampling
The fourth apporach involves the SMOTEENN algorithm, which combines SMOTE (synthetic minority sampling trachnique) to oversample the minority class, and ENN (edited nearest neighbors) to clean the resulting data. This reduces the effect of outliers. 

#### SMOTEENN
After oversampling the high risk loans and then cleaning the data with the edited nearest neighbors undersampling strategy, we had `Counter({'high_risk': 68460, 'low_risk': 62011})`. 
* Accuracy Score: 0.6447993752836463
* Confusion Matrix:![cm4](https://github.com/saramcel/Credit_Risk_Analysis/blob/f078092a8963de1f24822ee0c783292baf59612c/Resources/cm4.png)
* Classification Report:![cr4](https://github.com/saramcel/Credit_Risk_Analysis/blob/f078092a8963de1f24822ee0c783292baf59612c/Resources/cr4.png)
* This model has gotten the highest recall score for high risk loans of the models so far (0.72), and it predicted the most correct high risk loans, 73 true positives, which is three more than naive random oversampling and centroid undersampling (each predicted 70 correct high risk loans).

### Ensemble Classifier
Ensemble classifiers use many weak learners in combination to produce a model with better performance, or a strong learner. Many algorithms are combined to make the predictions in these models. 

#### Balanced Random Forest
This model is a balanced random forest classifier which randomly under-samples each boostrap sample to balance it. This model works by aggregation, meaning that the weak learners (classifiers) make predictions at the same time (i.e. not sequentially) and the prediction that we use is the one with the most votes.
* Accuracy Score: 0.7586962302234792
* Confusion Matrix: ![cm5](https://github.com/saramcel/Credit_Risk_Analysis/blob/f078092a8963de1f24822ee0c783292baf59612c/Resources/cm5.png)
* Classification Report:![cr5](https://github.com/saramcel/Credit_Risk_Analysis/blob/f078092a8963de1f24822ee0c783292baf59612c/Resources/cr5.png)
* This model has the highest accuracy score of all the models so far. The model detects low-risk loans very well (high precision, high recall, and high F1 score). Although it finds the majority of high-risk loans, the precision is still low, meaning that many loans are inaccurately classified as high risk (false positives), and the recall is only 0.63, which means that several high risk loans are missed (false negatives). 

* We were also able to get a list of the most predictive features. The top five features and their relative importances are listed below.

![ri5](https://github.com/saramcel/Credit_Risk_Analysis/blob/f078092a8963de1f24822ee0c783292baf59612c/Resources/ri5.png)

#### Easy Ensemble Classifier
EasyEnsemble consists of a bag of Adaboost learners. Adaboost runs models in sequence and the errors of the previous model are given extra weight in the current model until the error rate is minimized. The learners are trained on different balanced boostrap samples. The balancing is achieved by random under-sampling. 
* Accuracy Score: 0.9271144308672863
* Confusion Matrix: ![cm6](https://github.com/saramcel/Credit_Risk_Analysis/blob/f078092a8963de1f24822ee0c783292baf59612c/Resources/cm6.png)
* Classification Report:![cr6](https://github.com/saramcel/Credit_Risk_Analysis/blob/f078092a8963de1f24822ee0c783292baf59612c/Resources/cr6.png)
* The accuracy of this model is the highest of the models we have explored. Although the precision for detecting high risk loans is still low at 0.07, meaning that there are still many false positives for the high risk category, this is the best precision we have seen of any model and the lowest count of false positives (1118). The recall rate for both high risk and low risk loans are both over 0.9 in this model. That means that the model avoids false negatives (predicted low risk but actually high risk) which means that the model can be trusted not to lead to very many risky loans by accident. The F1 score for low risk loans is 0.97, which indicates a very strong ability to detect low risk loans.

## Summary
The objective is to help predict credit risk to reduce the number of risky loans (to avoid loan default) and increase the number of low risk loans (to not miss out on good customers). The model that does that the best is EasyEnsemble Classifier. This uses a bag of boosted learners to reduce errors. This model overpredicts high risk loans, which could mean missed profits, but it also does not allow nearly as many true high risk loans to be misclassified as low risk, which could turn out even worse. This model does a good job of fulfilling both of the objectives. 
