# Player Performance Prediction in Soccer
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
**Acknowledgement** \
For the data source, I have used a public respository owned by Vaastav Anand: [https://github.com/vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League).

<br/><br/>

## B. Data and Design
\
**Design**\
Two Python notebooks were used to code the solution:\
\- *Data Prep and EDA.ipynb* : Aggregates and Cleans data to generate *master_data.csv*. Also contains EDA graphs.\
\- *LSTM Player Forecast.ipynb* : Data Modeling and Evaluation\
\
**Data**\
Data has been aggregated from the following source tables (total of 241 files):\
\- gw(n).csv : Player performance data for each gameweek. Total of 38 gameweeks x 6 seasons = 228 files.\
\- fixtures.csv : List of all fixtures for the season = 6 files. It contains team IDs and kickoff times.\
\- master_team_list.csv : Team ID and name mapping for each season in 1 file.\
\- players_raw.csv : Contains player position information for each season = 6 files.\
\
**Observations**\
\- Target is very sparse (has many 0s)\
<img src="images/Points distribution (before filter)1.JPG?raw=true"/>\
\
\- Target (after 0-50 filter) is right skewed\
<img src="images/Points distribution (after filter)1.JPG?raw=true"/>\
\
\
\- Several features are correlated with the Target (bps)\
\
<img src="images/correlation matrix1.JPG?raw=true"/>\
\
\
\- Target distribution varies significantly based on Player Position\
\
<img src="images/bps_position_boxplot.JPG?raw=true"/>\
\
\
\- Target distribution varies significantly based on Team (best and worst team displayed)\
\
<img src="images/bps_team_boxplot.JPG?raw=true"/>\
<br/><br/>

## C. Implementation

### Data Preparation

The following steps were taken to prepare *master_data.csv*:\
\- Aggregate gameweek data\
\- Clean player name\
\- Filter out blank fixtures\
\- Create fixture list from raw data for 16-17 and 17-18 seasons (not present in fixture file)\
\- Map team ID to team names\
\- Create *team_goals_scored* and *team_goals_conceded* features\
\- Get missing position from player data\
\- Filter out player-club occurences that total 0 minutes\
\- Create *player_kickoff_id*\
\- Encode team and opponent categories based on total goals scored and conceded in the previous season\
\- Encode position with average points per minute\
\
Further steps to prepare data for modeling:\
\- Adjust column types and sort rows by kickoff_time\
\- Filter data for *bps* between 0 and 50, done only in this version to limit the scope of the project. This filter is different from outlier removal which would be done only on Training data.\
\- Filter for players who have played a total of at least 38 games across all seasons
\- Normalize data\
\- For Random Forest: For features whose values are unknown for the future gameweek, shift values in data frame by 1 period\
\- For LSTM: Create 3d input and output (n_samples x n_timesteps x n_features)

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

