# Executive Summary
In this project, our objective was to predict the likelihood of fatal cases in traffic collisions in Toronto using the KSI dataset, which includes information about all traffic collision events where a person was either Killed or Seriously Injured from 2006 to 2021. To achieve this, our team has developed and compared four machine learning models - Random Forest, Histogram-based Gradient Boosting, Logistic Regression, and Support Vector Machines (SVM) with Bagging - to predict fatal traffic collision cases in Toronto. Our DataLoader class loads the dataset and performs data cleaning, transformation, standardization, SMOTE sampling, feature selection, and encoding.
We evaluated the performance of the models using accuracy, precision, recall, and F1-score metrics. The Histogram-based Gradient Boosting model demonstrated the best performance, with an accuracy of 96.46%, a precision of 96.42%, a recall of 96.46%, and an F1-score of 96.31%.
# Overview of Solution
Our solution involved the creation of a DataLoader class to efficiently manage the KSI dataset, preprocess the data, and prepare it for modelling. This class was responsible for loading the dataset, cleaning the data, transforming features, selecting relevant features, applying SMOTE sampling, standardizing the data, and encoding categorical variables.
We then built four machine learning models - Random Forest, Histogram-based Gradient Boosting, Logistic Regression, and SVM with Bagging - to predict whether a traffic collision would result in a fatality. We trained each model on the preprocessed data and evaluated it using accuracy, precision, recall, and F1-score metrics. The performance of each model was as follows:

| | Accuracy | Precision |	Recall |	F1 Score|
|-|-|-|-|-|
| Random Forest |	92.51%	| 92.07%	| 92.51%	| 91.96% |
| Histogram-based Gradient Boosting |	96.46%	| 96.42%	| 96.46%	| 96.31% |
| Logistic Regression |	61.52%	| 80.75% | 61.52%	| 67.41% |
| SVM With Bagging | 61.92% |	80.52% | 61.92% | 67.73% |

Based on these results, the Histogram-based Gradient Boosting model outperformed the other models, making it the most suitable choice for predicting fatal traffic collisions in Toronto using the KSI dataset.
