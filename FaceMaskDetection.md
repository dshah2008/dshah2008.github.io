# Face Mask Detection with Yolov5
\
**Table of Contents**
* TOC
{:toc}

[View GitHub Repository](https://github.com/dshah2008/Face-Mask-Detection)
<br/><br/>

## A. Overview 
\
**Use Case** \
Detect faces in video footage and determine whether a person is wearing a face mask or not.\
\
**Objective** \
The objective is to learn the implementation of Deep Learning-based Object Detection algorithms while applying it to a complex real-world business problem.\
\
**Business Problem** \
As society moves through the finale of the global pandemic with a relaxation of COVID-19 restrictions, establishments are returning to business as usual. This includes the re-opening of major public spaces, retail outlets, corporate offices etc. However, with the uncertainty around resurgence of infections, such establishments need to continue enforcing protocols for the safety of their employees and customers.
One protocol that continues to sit at the forefront is the use of facemasks and its efficacy in impeding the transmission of COVID-19. Request to use face masks prevails with the re-opening of major institutions. 
The biggest challenge these institutions face is enforcing mask-wearing measures at high volumes and in large crowds. Manual enforcement is a daunting task and increasing security personnel is very costly for businesses.
Our solution is to use state-of-the-art Computer Vision technology that can help with mask detection. This would assist security personnel in implementing mask protocols and ensure that businesses do not have to incur increased security costs.\
\
**Challenges** \
While there are many existing solutions to this problem, the common approaches overlook some of the challenges. Key challenges our model attempts to address when detecting masks:\
\-	Large groups of people in close proximity\
\-	People looking away from the camera\
\-	People moving at a fast pace\
\-	Identifying objects in complex backgrounds\
\
**Contributions** \
Ting Lan - Data Preparation\
Dhrumel Shah - Modeling

<br/>

## B. Data
\
Two data sources were used:\
\
A. Data from Kaggle containing 853 images. The images capture different public scenarios in which people are either wearing a mask (correctly or incorrectly) or not wearing a mask. [View Dataset link](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)\
B. Additional data containing 54 images from random online searches that focused on images of faces without masks and from side angles.\
\
Additional data was incorporated since data from the source A was heavily biased towards faces with masks only. It also did not contain enough samples with faces at different angles.\
\
<img src="images/class_dist1.png?raw=true"/>

<br/>

## C. Design & Implementation

### Data Preparation

\- Reformat annotation files\
\- Convert from a 3-class to a 2-class problem (partially-worn masks are outside the scope)\
\- Train, validation and test splits

### Modeling

**Why Deep Learning CNNs?** \
Object Detection had been implemented for several decades using traditional Computer Vision techniques. However, 2014 marked a significant paradigm shift as CNN-based models began dominate the Computer Vision space in terms of both accuracy and speed. While neural networks have an obvious advantage of being able to learn from millions of images with large number of layers and parameters, CNNs specifically are useful due to several reasons. The Convolution operations are vital for capturing spatial information from images, while Pooling helps filter the relevant information and helps reduce dimensions.\
\
**YOLO v5** \
YOLO, or You Only Look Once, was introduced in 2016 and was a big improvement on RCNN models as it produced similar detection accuracy but was significantly faster. 
In RCNNs, region proposals are created and detection is performed for each, whereas with YOLO, the entire image is passed through the network just once. The network also simultaneously predicts multiple bounding boxes and class probabilities for those boxes. Detection is treated like a regression problem, making it even faster. 
Since our problem relies heavily on both accuracy and speed, YOLO was our architecture of choice. We also went with Yolov5 since it is the latest and most supported version.\
\
**Model Architecture** \ 
3 main components: 
\1.	Backbone: CSPDarknet is employed as the backbone for feature extraction from images consisting of cross-stage partial networks\
\2.	Neck: PANet used to generate a feature pyramids network to perform aggregation on features and pass forward to head for prediction.\ 
\3.	Head: Layers that generate predictions from the anchor boxes for object detection.\
\
<img src="images/model_arch1.png?raw=true"/>
\
There are 5 different versions of YOLOv5. The larger the model size, the more the parameters and longer the inference time. Based on our computational resources and dataset size, we opted for YOLOv5s, but also experimented with v5m.\
\
<img src="images/model_versions1.png?raw=true"/>

<br/>

## D. Experimentation


## E. Results

### Results

|     Model     | Train MSE | Train MAE | Test MSE | Test MAE |
| ------------- | --------- | --------- | -------- | -------- |
| Random Forest |  0.0049   |  0.0548   |  0.0350  |  0.1475  |
|     LSTM      |  0.0115   |  0.0859   |  0.0108  |  0.0817  |

<br/>

While the Random Forest has a lower Training error, it does overfit. On the unseen Test data, the LSTM model performs significantly better based on both Mean Squared Error and Mean Absolute Error.\
\
\
Following is a comparison of actual and predicted values for the LSTM (most recent gameweek for 50 players).\
\
<img src="images/results1.png?raw=true"/>


### Next Steps

\- Forecast next 5 timesteps for each player instead of the single average prediction\
\- Additional data and features: Opponent history, team formations, player position breakdown, injury news\
\- Improve encoding: use rolling average of points to encode *team* and *opponent* instead of just previous season totals\
\- Remove data filters\
\- Experiment with more models: LSTM with Attention head, Transformers, Neural Prophet

