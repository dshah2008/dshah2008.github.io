# Faulty Water Pump Classification
\
**Table of Contents**
* TOC
{:toc}

[View GitHub Repository](https://github.com/dshah2008/Water_Pump_Classification)
<br/><br/>

## A. Overview 
\
**Use Case** \
Classify the water pumps at various locations in Tanzania as one of 3 classes:\
\- Functional\
\- Non-functional\
\- Functional but needs repair\
\
**Objective** \
The main objective is to learn how to tackle a Machine Learning Classification problem with a large number of categorical features. In addition, I'm looking to improve my modeling skills by using sklearn pipelines with custom transformers, as well as using Optuna for hyperparameter tuning. I have also implemented a novel technique for detecting class-based outliers.\
\
This project is based on a practice competition hosted by *DrivenData*, a crowdsourcing platform that helps tackle social issues with the help of the Data Science community. Part of this project was implemented in a team in the Machine Learning course I had taken at Queen's University.\
\
**Business Problem** \
About 25 million people in Tanzania do not have access to clean drinking water. This has led to a widespread increase in waterborne diseases all across the country and a serious health crisis. As the government struggles to find solutions, our project aims to use predictive analytics to identify the Functional, Non-Functional and Functional-but-needs-repair waterpoints to help them optimally allocate water resources and maintain water pumps.\
\
**Contributions** \
The data exploration was performed with the help of my teammate Ting Lan at Queen's University. Another teammate Kevin Wang helped in experimenting with different models. However, all the code displayed in this repository has been developed only by me.

<br/>

## B. Data
\
The data is provided by DrivenData in 3 files: Train_X.csv, Train_Y.csv, Test_X.csv.\
\
\- Training data sample size: 59,400\
\- Classes: 3\
\- Total features: 39 (geography, construction type, water source and quality, etc.)\
\- Categorical features: 28\
\
**Key Observations**\
\
\- 9 features are redundant (similar to other features)\
\- 7 have missing values\
\
Examining the class-conditional distribution of features helps us identify the important ones.\
\
\- The proportion of non-functional pumps decreases almost linearly over time based on the construction year\
\
<img src="images/const_year1.JPG?raw=true"/>

\
\- Overall, the 3 classes seem to be similarly distributed across the country. However, the *functional but needs repair* pumps seem to be located slightly more North-West based on the median latitude and longitude shown in the right-most graph.\
\
<img src="images/lat_long1.JPG?raw=true"/>

\
\- If we observe specific regions, the distribution varies significantly. In Mtwara and Lindi non-functional waterpoints account for more than 60% of the total whereas in Iringa and Arusha they are less than 30%.\
\
<img src="images/region1.JPG?raw=true"/>

\
\- The proportion of non-functional pumps decreases with increase in height\
\
<img src="images/gps_height1.JPG?raw=true"/>

\
\- Waterpoints with a source in lakes or dams are far more likely to be non-functional than those getting water from rivers and springs.\
\
<img src="images/water_source1.JPG?raw=true"/>

<br/>

## C. Implementation

**Data Cleaning**\
\- Drop redundant columns that are similar to other features\
\- Change 0 construction year to NaN as missing values are handled internally by boosting trees\
\- Standardize spellings in the installer feature so that there are no redundant categories\
\
**Feature Engineering**\
\- Create *month_recorded* as a feature from date_recorded\
\- Extract days since start of time from date_recorded\
\- Due to the high cardinality of several categorical features, we group the ones that have a low frequency and uninteresting class distribution into a single *Rare* category. These were identified using the PowerBI dashboard, and the grouping is done in code.\
\- The categorical features are then encoded using both One-Hot-Encoding and LightGBM's default encoder.\
\
**Class-based Outlier Detection**\
\
Boosting trees are mostly robust to outliers. However, class-based outliers can cause them to overfit. It refers to data points whose large proportion of neighbors in the feature space belong to a different class.
\
<img src="images/class_outlier1.png?raw=true"/>

I have implemented a function that first calls the sklearn.neighbors package to determine the k nearest neighbors for each data point. I then calculate what proportion of neighbors have a different class label. If this is greater than a certain threshold (in our case 85%), the given data point is flagged as an outlier and removed from the training data. This resulted in a significant increase in test accuracy.\
\

**Modeling and Evaluation**\
\
Due to the success of tree-based models on structured datasets with categorical features, we experimented exclusively with Random Forest, LightGBM and CatBoost. LightGBM and CatBoost produced the best results, and were particularly useful because of their inbuilt capability of handling missing values and categorical features. The code in the repository only contains the LightGBM implementation, as the other two models were implemented by my teammates as mentioned in Contributions.\
\
The models results were evaluated by DrivenData using standard classification accuracy. We also used 10-fold cross-validation to tune and evaluate the models before submission. Tuning was done with the help of Optuna's MedianPruner. A further Grid Search was also carried out based on the best results from Optuna.

<br/>

## D. Results & Business Impact

The final model was tested on an independent dataset with 136 images.\
\
**mAP score**\
\
<img src="images/result_map1.png?raw=true"/>

**Confusion Matrix**\
\
<img src="images/result_cm1.png?raw=true"/>


