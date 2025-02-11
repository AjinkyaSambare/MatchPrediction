Sure, I'll include the link to the dataset in the appropriate sections of the README. Here's the updated version:

# Cricket Match Prediction System

## Organization
Explore and contribute: [Curious PM Community](https://curious.pm)

## Introduction

This project harnesses machine learning to predict outcomes of One Day International (ODI) cricket matches. By analyzing historical data from matches, team performances, and venue characteristics, the system provides informed predictions through a web interface.

## System Overview

The system comprises three main components, structured to handle different aspects of the machine learning pipeline:
1. **Data Processing**: Scripts to clean and prepare data for modeling.
2. **Model Training**: Scripts to train and optimize the machine learning model.
3. **Web Interface**: A user-friendly interface to interact with the model and obtain predictions.

## Directory Structure

```
MatchPrediction/
├── data/
│   ├── ODI_Match_Data.csv       # Historical match data.
│   ├── ODI_Match_info.csv       # General match information.
│   ├── label_encoders.pkl       # Encoded labels for categorical data.
│   └── processed_data.csv       # Data ready for training.
├── models/
│   └── cricket_predictor_v3.pkl # Serialized file of the trained model.
├── requirements.txt             # Dependencies required for the project.
├── scripts/
│   ├── preprocess.py            # Script for data preprocessing.
│   └── train_model.py           # Script for model training.
└── streamlit_main.py            # Streamlit application for predictions.
```

## Detailed Explanation of Components

### Data Processing (`preprocess.py`)

- **Purpose**: To transform raw match data into a format suitable for analysis and modeling.
- **Process**:
  - **Load Data**: Combines match detail and ball-by-ball action into a comprehensive dataset.
  - **Filter Data**: Removes matches involving non-standard teams like 'World XI'.
  - **Aggregate Statistics**: Computes match-level statistics like total runs and wickets.
  - **Generate Features**: Creates features such as team win rates and venue statistics.
  - **Encode Categoricals**: Converts textual data into numerical format using label encoding.

### Detailed Explantion to Preprocessing Script for Cricket Match Prediction System

## 1. Loading Data

The `load_data` function is responsible for loading the match datasets from CSV files.

```python
import pandas as pd

def load_data():
    """Load both match data and match info datasets"""
    match_data = pd.read_csv('../data/odi_match_data.csv', low_memory=False)
    match_info = pd.read_csv('../data/odi_match_info.csv', low_memory=False)
    return match_data, match_info
```

- **match_data.csv**: Contains detailed ball-by-ball action.
- **match_info.csv**: Includes general information about each match.

## 2. Filtering Special Teams

The `filter_special_teams` function removes matches involving exhibition or special teams, which might skew the analysis.

```python
def filter_special_teams(data):
    """Filter out exhibition and special teams"""
    special_teams = ['Asia XI', 'World XI', 'ICC World XI', 'Africa XI']
    mask = ~(data['team1'].isin(special_teams) | data['team2'].isin(special_teams))
    filtered_data = data[mask].copy()
    print(f"\nRemoved {len(data) - len(filtered_data)} matches involving special teams")
    return filtered_data
```

- **Purpose**: Ensures the data only includes official international matches.

## 3. Aggregating Match Data

The `aggregate_match_data` function converts ball-by-ball data to match-level statistics.

```python
def aggregate_match_data(match_data):
    """Aggregate ball-by-ball data to match level statistics"""
    match_stats = match_data.groupby(['match_id', 'innings', 'batting_team']).agg({
        'runs_off_bat': 'sum',
        'extras': 'sum',
        'wides': 'sum',
        'noballs': 'sum',
        'byes': 'sum',
        'legbyes': 'sum',
        'wicket_type': lambda x: x.notna().sum(),  # Count wickets
        'ball': 'count'  # Count balls
    }).reset_index()
    match_stats['total_runs'] = (match_stats['runs_off_bat'] + match_stats['extras'])
    return match_stats.pivot(index='match_id', columns='innings', values=['total_runs', 'wickets', 'balls_faced', 'extras'])
```

- **Result**: Provides a summarized view of each match by innings, which is crucial for feature engineering.

## 4. Creating Team Features

The `create_team_features` function calculates historical performance metrics for teams.

```python
def create_team_features(data):
    """Create team performance features"""
    data = data.sort_values('date')
    team_stats = {team: {'matches_played': len(matches), 'win_rate': len(matches[matches['winner'] == team]) / len(matches) if len(matches) > 0 else 0} for team in pd.concat([data['team1'], data['team2']]).unique()}
    return data
```

- **Calculations**: Include matches played and win rates, which help the model understand past performances.

## 5. Processing Venue Features

The `process_venue_features` function computes statistics related to match venues.

```python
def process_venue_features(data):
    """Process venue-related features"""
    venue_stats = data.groupby('venue').agg({'match_id': 'count'}).rename(columns={'match_id': 'matches_at_venue'})
    return data.merge(venue_stats, on='venue', how='left')
```

- **Venue Impact**: Considers how often each venue is used, which could influence match outcomes.

## 6. Creating Match Features

The `create_match_features` function extracts date and toss-related features.

```python
def create_match_features(data):
    """Create match-specific features"""
    data['year'] = pd.to_datetime(data['date']).dt.year
    data['month'] = pd.to_datetime(data['date']).dt.month
    data['day_of_week'] = pd.to_datetime(data['date']).dt.dayofweek
    data['toss_winner_is_team1'] = (data['toss_winner'] == data['team1']).astype(int)
    data['toss_winner_batted_first'] = ((data['toss_winner'] == data['team1']) & (data['toss_decision'] == 'bat') | (data['toss_winner'] == data['team2']) & (data['toss_decision'] == 'field')).astype(int)
    return data
```

- **Temporal Features**: Includes the year, month, and day of the week.
- **Toss Features**: Indicates whether the toss winner chose to bat or field first.

## 7. Encoding Categorical Features

The `encode_categorical_features` function encodes categorical columns to prepare them for modeling.

```python
def encode_categorical_features(data):
    """Encode categorical features"""
    encoders = {column: LabelEncoder().fit_transform(data[column].fillna('Unknown')) for column in ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'winner', 'season'] if column in data.columns}
    return data, encoders
```

- **Encoding**: Transforms textual data into a machine-readable format.

## 8. Main Preprocessing Function

The `preprocess_data` function orchestrates the entire preprocessing workflow.

```python
def preprocess_data(match_data, match_info):
    """Main preprocessing function"""
    match_summary = aggregate_match_data(match_data)
    combined_data = pd.merge(match_summary, match_info.rename(columns={'id': 'match_id'}), on='match_id', how='inner')
    combined_data = filter_special_teams(combined_data)
    combined_data = create_match_features(combined_data)
    combined_data = create_team_features(combined_data)
    combined_data = process_venue_features(combined_data)
    combined_data['result'] = (combined_data['winner'] == combined_data['team1']).astype(int)
    combined_data, encoders = encode_categorical_features(combined_data)
    combined_data.drop(['date', 'winner'], axis=1, errors='ignore', inplace=True)
    pd.to_pickle(encoders, '../data/label_encoders.pkl')
    return combined_data
```

- **Workflow**: This function ties all the preprocessing steps together, creating a dataset ready for model training.



### Model Training (`train_model.py`)

- **Purpose**: To develop a predictive model capable of forecasting match outcomes.
- **Process**:
  - **Data Augmentation**: Enhances the dataset by simulating reversed scenarios.
  - **Feature Selection**: Chooses relevant features based on historical importance.
  - **Model Selection**: Utilizes RandomForest due to its efficacy in handling diverse datasets.
  - **Hyperparameter Tuning**: Applies GridSearchCV to find the most effective model settings.
  - **Model Evaluation**: Assesses the model on unseen data to gauge its predictive power.
  - **Serialization**: Saves the trained model for later use in predictions.

### Web Interface (`streamlit_main.py`)

- **Purpose**: Provides a graphical user interface to interact with the trained model.
- **Features**:
  - **User Input**: Allows users to input match conditions such as teams, venue, and toss decisions.
  - **Model Interaction**: Processes input through the model to predict outcomes.
  - **Result Display**: Shows the predicted probabilities and outcomes in a user-friendly manner.

## Usage Instructions

1. **Set up the Environment**:
   - Install Python and required libraries:
     ```bash
     pip install -r requirements.txt
     ```

2. **Prepare the Data**:
   - Download and place the data in the appropriate directory from [Kaggle](https://www.kaggle.com/datasets/utkarshtomar736/odi-mens-cricket-match-data-2002-2023).
   - Run the preprocessing script to ready the data for training:
     ```bash
     python scripts/preprocess.py
     ```

3. **Train the Model**:
   - Navigate to the scripts directory and execute the training script:
     ```bash
     python train_model.py
     ```

4. **Run the Web Interface**:
   - Launch the Streamlit application to start making predictions:
     ```bash
     streamlit run streamlit_main.py
     ```

## System Performance and Metrics

- **Prediction Accuracy**: Typically achieves 70-75% accuracy on testing data.
- **Critical Features**: Includes team performance metrics, historical match outcomes, and venue specifics.

## Future Enhancements

- **Model Improvements**: Explore more sophisticated algorithms like XGBoost or deep learning approaches.
- **Data Enrichment**: Incorporate real-time data feeds for dynamic updating.
- **Feature Expansion**: Add new predictors such as player fitness, weather conditions, and more detailed team analytics.

Certainly! Here's an additional section to include in the README for showcasing screenshots of the application. This will help users visualize the interface and functionality of your Cricket Match Prediction System.

---
Certainly! Here's how you can add a section to your README to include a link to the hosted version of the Cricket Match Prediction System application:

---

## Hosted Application

Experience the Cricket Match Prediction System in action by visiting the live application at the link below:

[**Access the Cricket Match Prediction System**](https://matchresultprediction.streamlit.app)

---

## Screenshots

### Web Interface Overview

![Web Interface Overview](https://github.com/user-attachments/assets/b0c40f34-f1a6-4f78-abd9-326afc4520e5)

*Description: This screenshot displays the main page of the Streamlit web application, where users can input match details for prediction.*

### Prediction Results

![Prediction Results](https://github.com/user-attachments/assets/7f1ec308-8d09-4ef0-8bd8-e9df9d3c82a5)

*Description: After submitting the match details, this screenshot shows the predicted outcome including the probabilities for each team winning the match.*

### Model Accuracy and Metrics

![Model Metrics](https://github.com/user-attachments/assets/37d8e2d1-e16d-4ec5-a482-a79517171001)

*Description: This section of the application provides insights into the model's performance metrics and feature importance, giving users an understanding of how predictions are derived.*

---
