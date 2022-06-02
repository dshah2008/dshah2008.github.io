# Player Performance Prediction in Football (Soccer)
<br/><br/>
### A. Overview 
\
**Use Case**\
Predict the points accumulated by each player in the next 5 gameweeks of the English Premier League. These points are tabulated by the *Fantasy Premier League* game and are one of the most recognized indicators of game-wise player performance.\
\
**Objective** \
I have an immense passion for football and I'm eager to improve my understanding of the game with the help of Machine Learning. I'm looking to develop my skillset in Deep Learning and Forecasting, and eventually contribute towards the increasing use of AI in the multi-billion dollar football industry.\
\
With this project, over multiple cycles, I plan to develop state-of-the-art models that can accurately forecast player performance multiple timesteps into the future. In this first version, I have simplified the problem to predict the average of the points scored in the next five games. The model used is a deep neural network that combines several LSTM and fully-connected layers.\
\
**Background & Data Source** \
Over the past few years, football teams globally have spent millions to analyze games and develop strategies using AI. Companies such as Opta help these football clubs collect and aggregate match data. A lot of their data is publicly available via the Fantasy Premier League game on https://fantasy.premierleague.com/. The data from this game is far richer and cleaner than any other public source on football statstics. Several developers have also maintained GitHub repositories that scrape weekly match data weekly from the game's website. For this project, I have used the https://github.com/vaastav/Fantasy-Premier-League repository as my data source. It contains several data tables including player performance data for every match played in the last 6 years, as well as data describing players, teams and fixtures.\
\
**Challenges** \
\- Limited data for each player (~100 samples per player)\
\- Too many time series (>500 players = >500 unique time series)\
\- Need to forecast multiple timesteps\
\- High volatility of points scored by a player with no clear trend or seasonal patterns (as shown in the figure below)\
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
\- players_raw.csv : Contains player position information for each season = 6 files.\
\
The target variable *bps* indicating total points is present in *gw(n).csv*\
\
**Observations**\
\
\- Target distribution (before filter)\
<img src="Points distribution (before filter).JPG?raw=true"/>\
\
\- Target distribution (before filter)\
<img src="Points distribution (after filter).JPG?raw=true"/>\
\
\- Several features have a correlation with the Target (bps)\
<img src="correlation matrix1.JPG?raw=true"/>\
\
\- Target distribution wrt Player Position\
<img src="bps_position_boxplot.JPG?raw=true"/>\
\
\- Target distribution wrt Team (best and worst)\
<img src="bps_team_boxplot.JPG?raw=true"/>\
<br/><br/>

### C. Implementation
\
**Data Preparation**\
\
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
\- Filter data for *bps* between 0 and 50, done only in this version to limit the scope of the project\
\- Filter for players who have played a total of at least 38 games across all seasons
\- Normalize data\
\- For Random Forest: For features whose values are unknown for the future gameweek, shift values in data frame by 1 period\
\- For LSTM: Create 3d input and output (n_samples x n_timesteps x n_features)\
\
**Modeling**\
\


### 3. Support the selection of appropriate statistical tools and techniques

<img src="images/dummy_thumbnail.jpg?raw=true"/>

