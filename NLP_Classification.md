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
All data is contained in a single file: *public_data.csv*\
\
\- Training data sample size: 10,466\
\- Classes: 77\
\
\
**Key Observations**\
\
\- 77 classes with observations per class ranging from 40 to 150.\
\
<img src="images/nlp_classdist1.png?raw=true"/>
\
\
\
\- Most messages are short, containing 3 to 15 words.\
\
<img src="images/nlp_wordcount.png?raw=true"/>
\
\
\
\- Card, Account, Transfer and Exchange are some of the most frequently occuring non-stop words.\
\
<img src="images/nlp_freqwords.png?raw=true"/>

<img src="images/nlp_wordcloud.png?raw=true"/>

<br/>

## C. Design & Implementation
\
Three models were implemented in this project:

### 1. LightGBM with Tf-Idf

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
<br/>

### 2. LightGBM with SBERT

A hybrid model using LightGBM and SBERT or BERT Sentence Transformer.\
\
BERT (Bidirectional Encoder Representations from Transformers) was developed by Google in 2018 and was one of the most significant breakthrough in NLP. It was based on the original Tranformers paper, Vaswani et al. (2017). BERT is pre-trained on English Wikipedia and BooksCorpus for language modeling and next sentence prediction tasks. SBERT was developed in 2019 as a modification of the pre-trained BERT with signficant reduction in computation time. It outputs meaningful sentence embeddings in a 768 dimension vector. These embeddings can then be used as features in any Machine Learning model.\
\
Since BERT is trained on standard English sentences in Wikipedia articles, no text processing is recommended prior to encoding. After encoding the text samples with SBERT, we perform feature selection and train, tune and evaluate the LightGBM model just as we had done with the Tf-Idf based model.\
<br/>

### 3. DistilBERT

A Deep Learning DistilBERT Transformer model fine tuned on our dataset.\
\
DistilBERT is a compact version of the BERT Transformer with 40% fewer parameters. Despite the significant reduction in complexity, it retains most of BERT's language understanding power.\
\
I first used the model with zero-shot learning, using just the pre-trained weights with no training on our dataset. This did not yield good results. I then added a few trainable Dense layers but performance was still unsatisfactory. Finally, when I set all the layers to be trainable and fine-tuned the model on our dataset, the results were exceptional, even without much hyperparameter tuning. The final model is the distilbert-base-uncased with an additional Dropout and Dense layer.\
\
Note that prior to training the model, we are required to tokenize the input sentences using the dbert_tokenizer. It generates an input_ids and attention_mask from the training sample, which is then passed to DistilBert for training.\
<br/>

### Evaluation

The best model was chosen based on Classification Accuracy scores on test data. Although the classes have a slight imbalance, we are not interested in a macro accuracy or F1-score because the weightage we want to give each class is directly proportional to the number of samples they contain. However, in a real business setting, we may want to assign more weightage to a class like 'stolen card' over 'change address'. For tuning hyperparameters, we evaluate the models on log loss scores from 5-fold cross validation.

<br/>

## D. Results and Improvements
\
\
**Results**

|     Model      | Train Accuracy | Test Accuracy |
| -------------- | -------------- | ------------- |
| LightGBM-TfIdf |     0.9446     |    0.7139     |
| LightGBM-SBERT |     0.9731     |    0.8128     |
|   DistilBERT   |     0.9933     |    0.9174     |


As evident from the Test accuracy, the SBERT sentence encodings were a significant improvement on the Tf-Idf vectorizer as features for the LightGBM model. However, both models were comfortably outperformed by DistilBERT. DistilBERT was also the easiest model to implement as no feature engineering and minimal hyperparameter tuning was required. This experiment further validates the power of Transformer models in the NLP domain.

<br/>

**Classification Report**\
\
We can further interpret DistilBERT's results by examining the Precision, Recall and F1-Scores for each class.\
\
<img src="images/nlp_highf1.PNG?raw=true"/>

<img src="images/nlp_lowf1.PNG?raw=true"/>

\
The model performs well on every class, with no F1-Score below 75%. Only 4 out of the 77 classes have an F1-Score of less than 80%. 'Topping_up_by_card' and 'pending_transfer' are the worst performing classes while 'apple_pay_or_google_pay' and 'top_up_limits' are two of the best performing classes.\
\
Further examination of the Precision and Recall scores paint a clearer picture of the model weaknesses. Based on the low Precision, the 'topping_up_by_card' class produces many false positives. This means that the model incorrectly classifies many samples as 'topping_up_by_card' when they actually belong to another class. Similarly, based on the low Recalls, the 'pending_transfer' and 'pending_top_up' suffer from high False Negatives. This indicates that samples belonging to these classes are incorrectly classified with other labels. We could further analyze what these incorrect labels are by studying the confusion matrix, but that is beyond our current scope and would be recommended for next steps.

<br/>

**Improvements**\
\
While we're satisfied with the overall results, some improvements could be made if required by the business:\
\
\- Tune the key hyperparameters of DistilBERT: learning rate, batch_size, epochs, weight_decay.\
\- Experiment with transfer learning from bigger Transformer models like RoBERTa and XLNet, with partial freezing of layers.\
\- Analyze the confusion matrix for weak performing classes, determine which classes are difficult to separate and inspect the training samples and key words that could cause the misclassification. Pre-processing steps could be specifically applied to these samples. We could also increase the weightage for weak-performing classes during validation.\
\- Experiment with FinBERT, a BERT model futher pre-trained on a Finance corpus.

<br/>

