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
Ting Lan, Ajay Chaudhary - Data Gathering and Preparation\
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
\
Three main components:
1.	Backbone: CSPDarknet is employed as the backbone for feature extraction from images consisting of cross-stage partial networks
2.	Neck: PANet used to generate a feature pyramids network to perform aggregation on features and pass forward to head for prediction
3.	Head: Layers that generate predictions from the anchor boxes for object detection

<img src="images/model_arch1.png?raw=true"/>

There are 5 different versions of YOLOv5. The larger the model size, the more the parameters and longer the inference time. Based on our computational resources and dataset size, we opted for YOLOv5s, but also experimented with v5m.
\
<img src="images/model_versions1.png?raw=true"/>

### Experimentation

We conducted a comprehensive experimentation process. We can break it down into 3 phases:\
\
A.	Evaluation of an existing solution using FaceNet for Localization and MobileNetV2 for Classification. It produced unsatisfactory results on the test dataset.\
B.	Implementing Yolov5 on our existing dataset, experimenting with the architecture and tuning a few key hyperparameters. This improved our results.\
C.	Generating new data and appending it to our dataset. Further experimentation with architecture and complete hyperparameter tuning. The additional data helped produce another significant improvement in the results, beating all our benchmark scores.\
\
In phase 2 and 3, we carried out further experiments to help determine the best possible architecture and key hyperparameter values. We evaluated Yolov5 with the following:\
\-	Yolov5s with different pre-trained weights: None, 5s, 5s6\
\-	Freezing layers: 0, 3, 5, 7, 10, 18, 24(all)\
\-	Incorporate new data and repeat previous steps\
\-	Changed image size: 480, 640(default), 1280\
\-	Yolov5m: ‘m’ version is a bigger model with more parameters than ‘s’\
\-	Complete hyperparameter tuning using the evolve function\
\-	Increase epochs

### Evaluation Metrics
In order to evaluate our models, we have examined two metrics:\
\- mAP: mean-Average Precision\
\- Validation loss (cls, obj, bbox)\
\
**mAP** or mean Average Precision is one of the commonly used performance metrics in Object Detection. It measures the average Precision over all Recall values using the Precision-Recall curve, followed by computing the mean of Average Precisions over all classes.\
\
To determine correct predictions, the IoU score is used (Intersection-over-Union), which is simply the percentage of overlap between the actual and predicted bounding box. The mAP uses a specific IoU threshold, eg. 0.5. Thus, for a True Positive, the prediction must have an IoU > threshold.\
\
mAP is also appropriate for our problem because the classes are imbalanced and we’re not interested in how many times the model correctly detects the background of an image (True Negatives). We only care about correctly detecting the mask.\
\
For this project, we observe 2 mAP values:\
\-	mAP(0.5): Computes the mAP for a 0.5 IoU threshold\
\-	mAP(0.5-0.95): Computes the average mAP over 10 IoU thresholds from 0.5 to 0.95.\
\
**Validation Loss (Cls, Obj, Bbox)**\
\-	Bbox loss: Measures IoU loss of the actual and predicted bounding boxes\
\-	Cls loss: Measures the classification error of each predicted bounding box\
\-	Obj loss: Measures the confidence of identifying an object

<br/>

## D. Results, Deployment & Next Steps

### Results

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

### Next Steps - Deployment

Our next step would be to work towards Deploying the model in a production environment in a real corporate office setting. In order to ensure best possible performance and scalability, our recommendation would be to deploy our model on the cloud using Microsoft Azure services. Azure provides several affordable solutions for deploying and maintaining models in production.\
\
One such option is to use the Azure Video Analyzer service, allowing users to connect cameras directly to the cloud. This will eliminate the need for edge devices. The connection could be made via a remote device adapter. Following is a detailed architecture.\
\
<img src="images/deploy1.png?raw=true"/>

Once the model is deployed, ML-Ops services on Azure can be used to maintain and monitor the performance of models in production.

### Next Steps - Model Improvement

For our model to be production-ready, the following improvements would need to be made:\
\-	Include additional data on *Incorrectly-worn mask* faces and convert the solution to a 3-class problem\
\-	Add training images for *No Mask* and background to improve the Recall\
\-	Improve racial diversity of data

