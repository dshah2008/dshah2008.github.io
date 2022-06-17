# Player Performance Prediction in Soccer
\
**Table of Contents**
* TOC
{:toc}

[View GitHub Repository](https://github.com/dshah2008/FPL_PlayerPerformancePredictor)
<br/><br/>

## A. Overview 
\
**Use Case** \
Predict the points accumulated by each player in the next 5 gameweeks of the English Premier League. These points, referred to as *bps*, are tabulated by the *Fantasy Premier League* game based on performance indicators like goals scored, assists and goals conceded. Statistics such as key passes and tackles are also factored in.\
\
**Objective & Scope** \
I have an immense passion for football and I'm eager to improve my understanding of the game with the help of Machine Learning. I'm looking to develop my skillset in Deep Learning and Forecasting, and eventually contribute towards the increasing use of AI in the multi-billion dollar football industry.\
\
With this project, over multiple cycles, I plan to develop state-of-the-art models that can accurately forecast player performance multiple timesteps into the future. In this first version, I have simplified the problem to predict the average of the points scored in the next five games.\
\
The prediction also only considers appearances that accumulate between 0 and 50 points. The 0 lower limit is applied because they mostly indicate that the player does not make an appearance. It filters out almost 50% of samples. From a business standpoint, we are not interested in predicting whether or not a player will make an appearance. This is mostly known prior to the game, based on preferred starting line-ups or injuries. The upper limit of 50 is applied only in this iteration to simplify the problem. It filters out less than 1% of samples so it isn't very significant.\
\
**Background & Data Source** \
Football teams globally spend millions to analyze games and develop strategies using AI. Companies such as Opta help these football clubs collect and aggregate match data. A lot of their data is publicly available via the Fantasy Premier League game on [https://fantasy.premierleague.com/](https://fantasy.premierleague.com/). The data from this game is rich and clean enough to carry out analysis at many different levels. Several developers have also maintained GitHub repositories that scrape weekly match data weekly from the game's website. For this project, I have used the [https://github.com/vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) repository as my data source. It contains several data tables including player performance data for every match played in the last 6 years, as well as data describing players, teams and fixtures.\
\
**Challenges** \
\- Limited data for each player (~100 samples per player)\
\- Too many time series (>500 players = >500 unique time series)\
\- Need to forecast multiple timesteps\
\- High volatility of points scored by a player with no clear trend or seasonal patterns (as shown in the figure below)\
<img src="images/Points Trend - Bernardo Silva1.JPG?raw=true"/>\
\
**Other works** \
Several solutions do exist, predicting player performance using similar data sources. However, most of these approaches use bagging or boosting-based regression models. They do not leverage time series information which is critical for capturing patterns in player form and they do not predict more than one timestep into the future. They also do not use Deep Learning despite the high variance and complexity in the data.\
\
**Contributions** \
I am the sole contributor in this project. All the code in the repository has been developed only by me.\
\
**Acknowledgement** \
For the data source, I have used a public respository owned by Vaastav Anand: [https://github.com/vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League).
<br/>

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
\- The target is *bps* or *bonus points score*\
\- Number of samples used for modeling: 57,170\
\- Number of features used for modeling: 18\
\
**Key Observations**\
Target is very sparse because of players who don't make appearances:\
<img src="images/Points distribution (before filter)1.JPG?raw=true"/>\
\
Target distribution (after 0-50 filter):\
<img src="images/Points distribution (after filter)1.JPG?raw=true"/>\
\
The filter eliminates samples as follows:\
\- Original data size: 119,634\
\- Samples after > 0 filter: 57,655\
\- Samples after < 50 filter: 57,170\
\
Several features are correlated with the target *bps*. *Influence* has the highest correlation of 0.81:\
\
<img src="images/correlation matrix1.JPG?raw=true"/>\
\
\
Target distribution varies significantly based on Player Position:\
\
<img src="images/bps_position_boxplot.JPG?raw=true"/>\
\
\
Target distribution varies significantly based on Team (best and worst team displayed):\
\
<img src="images/bps_team_boxplot.JPG?raw=true"/>

<br/>

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
<br/>

## D. Results and Next Steps

### Results

|     Model     | Train MSE | Train MAE | Test MSE | Test MAE |
| ------------- | --------- | --------- | -------- | -------- |
| Random Forest |  0.0049   |  0.0548   |  0.0350  |  0.1475  |
|     LSTM      |  0.0115   |  0.0859   |  0.0108  |  0.0817  |

<br/>

While the Random Forest has a lower Training error, it does overfit. On the unseen Test data, the LSTM model performs significantly better based on both Mean Squared Error and Mean Absolute Error.\
\
Following is a comparison of actual and predicted values for the LSTM (most recent gameweek for 50 players).\
\
<img src="images/results1.png?raw=true"/>\
\
While the predictions do help us project player performance, they are far from robust. Despite improving on the Random Forest, the LSTM model struggles to capture the overall variance in points, especially the outliers. With these results and the overall objective in mind, we propose the following next steps.

### Next Steps

\- Additional data and features: Opponent history, team formations, player position breakdown, injury news\
\- Forecast next 5 timesteps for each player instead of the single average prediction\
\- Improve encoding: use rolling average of points to encode *team* and *opponent* instead of just previous season totals\
\- Remove data filters\
\- Experiment with more Seq2Seq models: LSTM with Attention head, Transformers, Neural Prophet
