# Player Performance Prediction - English Premier League Football  
<br/>
### A. Overview 
\
**Use Case**\
Predict the total points accumulated by each player in the next 5 gameweeks.\
\
**Objective** \
I have an immense passion for football(soccer) and I'm eager to improve my understanding of the game with the help of Machine Learning. I'm looking to develop my skillset in Deep Learning and Forecasting, and eventually contribute towards the increasing use of AI in the multi-billion dollar football industry.\
\
With this project, over multiple cycles, I plan to develop state-of-the-art models that can accurately forecast player performance multiple timesteps into the future. In this first version, I have developed a model that predicts the average points scored in the next five games for every player. The model is a deep neural network that combines several LSTM and fully-connected layers.\
\
**Background & Data Source** \
Over the past few years, football teams globally have spent millions to analyze games and develop strategies using AI. Companies such as Opta help these football clubs collect and aggregate match data. Their data is publicly available via the Fantasy Premier League game on https://fantasy.premierleague.com/. Several developers have maintained GitHub repositories that scrape weekly match data weekly from the game's website. For this project, I have used the https://github.com/vaastav/Fantasy-Premier-League repository as my data source. It contains several data tables including player performance data for every match played in the last 6 years, as well as data describing players, teams and fixtures.\
\
**Challenge** \
The major challenge in this project is being able to capture the volatility of the points scored by a player. If you observe the time series for a single player below, there does not appear to be any clear trend or seasonal pattern. The data for each player is also limited (~100 samples per player) and the volatlity is far greater than even cryptocurrencies. This is why even the biggest clubs and football experts struggle to predict player performance. The problem gets even more challenging when you try to forecast multiple timesteps.\
\
<img src="images/Points Trend - Bernardo Silva1.JPG?raw=true"/>\
\
**Other works** \
Several solutions do exist, predicting player performance using similar data sources. However, most of these approaches use bagging or boosting-based regression models. They do not leverage time series information which is critical for capturing patterns in player form and they do not predict more than one timestep into the future. They also do not use Deep Learning despite the high variance and complexity in the data.
<br/><br/>

### B. Data and Design
\
**Design**\
\
Two Python notebooks were used to code the solution:\
\- *Data Prep and EDA.ipynb* : Aggregates and Cleans data to generate *master_data.csv*. Also contains EDA graphs.\
\- *LSTM Player Forecast.ipynb* : Data Modeling and Evaluation\
\
**Data**\
\
Data has been aggregated from the following source tables (total of 241 files):\
\- gw(n).csv : Player performance data for each gameweek. Total of 38 gameweeks x 6 seasons = 228 files.\
\- fixtures.csv : List of all fixtures for the season = 6 files. It contains team IDs and kickoff times.\
\- master_team_list.csv : Team ID and name mapping for each season in 1 file.\
\- players_raw.csv : Contains player position information for each season = 6 files.
<br/><br/>

### C. Implementation
\
**Data Preparation**\
\
The following steps were taken to prepare the data:\
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

### 3. Support the selection of appropriate statistical tools and techniques

<img src="images/dummy_thumbnail.jpg?raw=true"/>

