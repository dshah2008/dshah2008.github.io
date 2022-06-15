# NLP Topic Classification - Customer Support Requests in Banking
\
**Table of Contents**
* TOC
{:toc}

[View GitHub Repository](https://github.com/dshah2008/NLP_TopicClassification)
<br/><br/>

## A. Overview 
\
**Use Case** \
The data consists of messages from customers, requesting banking support related to personal accounts, transactions, credit cards, etc. Each message needs to be classified into one of 77 classes. The classes include topics such as *request_refund*, *cancel_transfer* and *exchange_charge*.\
\
**Objective** \
Learn how to solve an NLP Classification problem with a large number of classes, using both shallow Machine Learning and Deep Learning models.\
\
**Contributions** \
I'm the sole contributor in this project. It was first implemented by me as part of the Natural Language Processing course I had taken at Queen's University.\
<br/>

## B. Data
\
The data originates from a real-world banking dataset but has been open-sourced for use in Academia. It was provided by Dr. Stephen Thomas of Queen's University as part of a class project.\
\
All data is contained in a single files: public_data.csv.\
\
\- Training data sample size: 10,466\
\- Classes: 77\
\
**Key Observations**\
\
\- 77 classes with observations per class ranging from 40 to 150.\
\
<img src="images/pump_class.png?raw=true"/>
\
\
\- Most messages are short, containing 3 to 15 words.\
\
<img src="images/pump_class.png?raw=true"/>
\
\
\- Card, Account, Transfer and Exchange are some of the most frequently occuring non-stop words.\
\
<img src="images/const_year1.PNG?raw=true"/>

<img src="images/const_year1.PNG?raw=true"/>

<br/>

## C. Design & Implementation
\
Three models were implemented in this project:\
\
**LightGBM with Tf-Idf**\
\
A shallow model using LightGBM with Tf-Idf vectors as features.\
\
Tf-Idf helps create features from text samples by assigning a score to each word token/term that occurs in the corpus. For a set of terms and documents:\
**tf-idf(t, d) = tf(t, d) * idf(t)**\
where tf is the frequency of term t in d\
**idf(t) = log (n / df(t)) + 1)**\
n is the total number of documents, df is the number of documents containing term t\
\
Prior to vectorization, we clean the text by converting it to lower case, removing stop words (from spacy) and removing all characters except the alphabet. We also tokenize the sentences into words using nltk. Note that vectorization is performed in a pipeline in order to avoid data leakage during validation and testing.\
\
After Tf-Idf scores are generated for each sample and each term, we perform Recursive Feature Elimination to identify the most significant features based on the feature importance scores outputed by LightGBM. Once the top N features are selected, the model is once again trained to produce the classification.\
\
We tune the model hyperparamters, including the N no. of features value using Optuna's Median Pruner. Tuning is based on 5-fold cross-validation and the log loss metric.\
\
**LightGBM with SBERT**\
\
A hybrid model using LightGBM and SBERT or BERT Sentence Transformer.\
\
SBERT was developed in 2019 as a modification of the pre-trained BERT with signficant reduction in computation time. It outputs meaningful sentence embeddings in a 768 dimension vector. These embeddings can then be used as features in any Machine Learning model.\
\
After encoding the text samples with SBERT, we perform feature selection and train, tune and evaluate the LightGBM model just as we had done with the Tf-Idf based model.
\

**DistilBERT**\
\
\- A Deep Learning DistilBERT Transformer model trained from scratch\
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
Due to the general success of tree-based models on structured datasets with categorical features, we experimented exclusively with Random Forest, LightGBM and CatBoost. LightGBM and CatBoost produced the best results, and were particularly useful because of their inbuilt capability of handling missing values and categorical features. The code in the repository only contains the LightGBM implementation, as the other two models were implemented by my teammates as mentioned in Contributions.\
\
The models results were evaluated by DrivenData using classification accuracy. We also used 10-fold cross-validation to tune and evaluate the models before submission. Tuning was done with the help of Optuna's MedianPruner. A further Grid Search was also carried out based on the best results from Optuna.

<br/>

## D. Results & Business Impact

**Results**

|     Model     | CV Accuracy | Test Accuracy |
| ------------- | ----------- | ------------- |
| Random Forest |   0.7942    |    0.7875     |
|   CatBoost    |   0.8195    |    0.8146     |
|   LightGBM    |   0.8213    |    0.8170     |

<br/>

**Classification Report**\
\
<img src="images/accuracy_report1.PNG?raw=true"/>

**Confusion Matrix**\
\
<img src="images/CM.png?raw=true"/>

While the model performs well overall, it struggles with the minority Functional-Needs-Repair class. This is mainly because of less data available for the class. While under-sampling and over-sampling techniques could help improve performance, it was not prioritized as our only objective for the competition is the overall accuracy and not the macro average.\
\
The Recall of the Non-Functional class is also not so high. This indicates high False Negatives for the class and from the confusion matrix, we can further validate that a significant proportion of Non-Functional pumps are being predicted as Functional.\
\
On the other hand, the Functional class has very high Recall. It has fewer False Negatives than False Positives, which means that very few Functional pumps are being predicted as Non-Functional.\
\
If we were to prioritize the business objective over the competition scores, we would focus on improving the Non-Functional Recall and Functional Precision. This is because predicting Non-Functional pumps as Functional is very costly in terms of health hazards. This is explained in the Business Impact section.\
\
**Business Impact**\
\
Although there is no scope for our model to be actually used by the Tanzanian Government, we have analyzed its business impact for our own learning.\
\
As of today, the government does not have enough resources to inspect every waterpoint in the country. With the help of our predictions, the Functional waterpoints can be identified and used to improve water allocation across communities. The Non-Functional waterpoints can be removed so that people do not have to consume unclean and harmful water. For the Functional-Needs-Repair waterpoints which constitute just 7% of the total, if correctly identified, maintenance and repair can easily be arranged.\
\
Improving water access with our predictions could have a massive impact financially and on people's health. It is esimtated that 43% of Tanzanians do not have access to safe drinking water. 23,900 children under the age of 5 also die every year due to water-borne diseases. 70% of the Tanzania Govt health budget (~700 million USD) is spent on diseases linked to lack of clean water and sanitation.\
\
Based on these figures and a few assumptions, we estimate that each Non-Functional or FNR waterpoint predicted as Functional would cost 6200 USD in terms of healthcare and medical infrastructure costs, as well as 0.7 lives of infants. Each Functional waterpoint predicted as NF or FNR would result in additional inspection and labor cost of 200 USD. These estimates help us map the costs to our confusion matrix.\
\
<img src="images/financials1.PNG?raw=true"/>


