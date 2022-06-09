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
This project is based on a practice competition hosted by *DrivenData*, a crowdsourcing platform that helps tackle social issues with the help of the Data Science community. Part of this project was implemented in the Machine Learning course I had taken at Queen's University.\
\
**Business Problem** \
About 25 million people in Tanzania do not have access to clean drinking water. This has led to a widespread increase in waterborne diseases all across the country and a serious health crisis. As the government struggles to find solutions, our project aims to use predictive analytics to identify the Functional, Non-Functional and Functional-but-needs-repair waterpoints to help them optimally allocate water resources and maintain water pumps.\
\
**Contributions** \
Some of the data exploration was performed with the help of Ting Lan at Queen's University. However, all the code displayed in this repository has been developed only by me.

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

### Data Preparation

\- Reformat annotation files\
\- Convert from a 3-class to a 2-class problem (partially-worn masks are outside the scope)\
\- Train, validation and test splits

### Feature Engineering

### Class-based Outlier Detection

### Modeling

### Evaluation


<br/>

## D. Results

The final model was tested on an independent dataset with 136 images.\
\
**mAP score**\
\
<img src="images/result_map1.png?raw=true"/>

**Confusion Matrix**\
\
<img src="images/result_cm1.png?raw=true"/>

Based on the mAP table, at a localization of 0.5 IoU, our model produces a high mAP of 89%. If we opt for very accurate localization, the mAP drops to 60%. However, for this business problem, an IoU of 0.5 is sufficient.\
\
Our model is very accurate with its *With Mask* predictions. However, for *No Mask*, it has a low Recall. If we look at the Confusion Matrix, a significant number of *No Mask* faces are not identified in the image and simply treated as background. The key reason behind such misclassification is the limited number of training images containing *No Mask* faces as well as only background images.\
\
We further examine the Validation loss shown below to ensure our model robustness. At 100 epochs, our model is stable in terms of both localization and classification.\
\
<img src="images/result_valloss1.png?raw=true"/>

Below is a sample of predictions on the test images.\
\
<img src="images/result_sample1.jpg?raw=true"/>

The model also performs well on Video footage. The python notebook demonstrates this on a sample 60 second video clip. 


