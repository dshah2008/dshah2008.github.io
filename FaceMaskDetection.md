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
**Data**\
Two data sources were used:\
A. Data from Kaggle containing 853 images. The images capture different public scenarios in which people are either wearing a mask (correctly or incorrectly) or not wearing a mask. [View Dataset link](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)\
B. Additional data containing 54 images from random online searches that focused on images of faces without masks and from side angles.\
\
Additional data was incorporated since data from the source A was heavily biased towards faces with masks only. It also did not contain enough samples with faces at different angles.\
\
<img src="images/class_dist1.png?raw=true"/>

<br/>

## C. Implementation

### Data Preparation

\- Reformat annotation files\
\- Convert from a 3-class to a 2-class problem (partially-worn masks are outside the scope)\
\- Train, validation and test splits

### Modeling

Two model architectures were implemented to solve the problem:\
\- **Random Forest**: Establish a baseline score\
\- **LSTM Network**: Deep Neural Network with a combination of LSTM and fully-connected layers\
\
**LSTM Architecture**\
\
<img src="images/Model Architecture1.JPG?raw=true"/>\
\
There are several reasons behind choosing the LSTM Network over other statistical, ML and DL models:\
\
\- **Learn sequential patterns**: This is vital for our problem since in most sports, player performance is primarily dependent on the player's form going in to the game. This makes LSTM more powerful than ML regression models.\
\
\- **Learn from multiple time series**: Since we build forecasts for more than 500 players, we have more than 500 time series that need to be learnt. With existing implementations of forecasting models like ARIMA, you would need to build a separate model for each series. You could use VAR models but they would require very high dimensionality since each series would be a separate feature. With LSTM, each time series is passed as a group of data samples, allowing you to train them in a single model.\
\
\- **Forecast multiple timesteps**: Although the current implementation predicts only one output, the 3rd LSTM layer predicts an output sequence for the next 5 games. This allows the model to not only learn from the player's recent trend, but also from features whose values differ for each of the 5 games. The multi-timestep output which will be implemented in the next iteration of the project will require adjustments to only the final LSTM and dense layers. Such a Seq2Seq architecture makes this model more suited to the problem than any other regression model.\
\
\- **Multi-input modeling**: Although LSTM doesn't inherently implement a Seq2Seq architecture, the use of inputs at different layers allows the model to learn the known future features such as *opponent* and *home_vs_away*.\
\
\- **Capture high variance**: As with all deep learning models, multiple layers and large number of neurons allows the model to learn the high variance while also introducing sufficient bias through regularization.

### Evaluation

Metrics: Mean Squared Error, Mean Absolute Error\
\
Validation: After a holdout Train-Val-Test split, a rolling-window evaluation process is used where in the first iteration, the model tests on N-4 to N timesteps and Trains on 0 to N-5 timesteps. N is the total no. of timesteps. In the second iteration, model tests on N-9 to N-5 timesteps and trains on 0 to N-10 timesteps. This continues until all the test data has been evluated. For a single output model, the model tests on the average of N-4 to N timesteps.\
\
Tuning: The model is tuned with the help of Optuna's Random Sampler and Median Pruner

<br/><br/>

## D. Results and Next Steps

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

