# Cricket Match Prediction System

## Join the Community
[Curious PM Community](https://curious.pm) to connect, share, and learn with others!

## Introduction

This project harnesses machine learning to predict outcomes of One Day International (ODI) cricket matches. By analyzing historical data from matches, team performances, and venue characteristics, the system provides informed predictions through a web interface.

## System Overview

The system comprises three main components, structured to handle different aspects of the machine learning pipeline:

* Data Processing: Scripts to clean and prepare data for modeling
* Model Training: Scripts to train and optimize the machine learning model
* Web Interface: A user-friendly interface to interact with the model and obtain predictions

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

* Purpose: To transform raw match data into a format suitable for analysis and modeling
* Process:
  * Load Data: Combines match detail and ball-by-ball action into a comprehensive dataset
  * Filter Data: Removes matches involving non-standard teams like 'World XI'
  * Aggregate Statistics: Computes match-level statistics like total runs and wickets
  * Generate Features: Creates features such as team win rates and venue statistics
  * Encode Categoricals: Converts textual data into numerical format using label encoding

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

* match_data.csv: Contains detailed ball-by-ball action
* match_info.csv: Includes general information about each match

## 2. Filtering Special Teams

```python
def filter_special_teams(data):
    """Filter out exhibition and special teams"""
    special_teams = ['Asia XI', 'World XI', 'ICC World XI', 'Africa XI']
    mask = ~(data['team1'].isin(special_teams) | data['team2'].isin(special_teams))
    filtered_data = data[mask].copy()
    print(f"\nRemoved {len(data) - len(filtered_data)} matches involving special teams")
    return filtered_data
```

* Purpose: Ensures the data only includes official international matches

## 3. Aggregating Match Data

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
```

* Result: Provides a summarized view of each match by innings

## 4. Creating Team Features

```python
def create_team_features(data):
    """Create team performance features"""
    data = data.sort_values('date')
    team_stats = {}
    for team in teams:
        team_matches = data[(data['team1'] == team) | (data['team2'] == team)]
        win_rate = len(team_matches[team_matches['winner'] == team]) / len(team_matches)
        team_stats[team] = {
            'matches_played': len(team_matches),
            'win_rate': win_rate
        }
    return data
```

* Calculations: Include matches played and win rates

## 5. Processing Venue Features

```python
def process_venue_features(data):
    """Process venue-related features"""
    venue_stats = data.groupby('venue').agg({
        'match_id': 'count'
    }).reset_index()
    venue_stats.columns = ['venue', 'matches_at_venue']
    return pd.merge(data, venue_stats, on='venue', how='left')
```

* Venue Impact: Considers venue statistics for match predictions

### Model Training (`train_model.py`)

* Purpose: To develop a predictive model capable of forecasting match outcomes
* Process:
  * Data Augmentation: Enhances the dataset by simulating reversed scenarios
  * Feature Selection: Chooses relevant features based on historical importance
  * Model Selection: Utilizes RandomForest due to its efficacy in handling diverse datasets
  * Hyperparameter Tuning: Applies GridSearchCV to find the most effective model settings
  * Model Evaluation: Assesses the model on unseen data to gauge its predictive power
  * Serialization: Saves the trained model for later use in predictions

### Explanation to Model Training Script

#### 1. Data Augmentation with Team Swaps

This function augments the data by swapping team positions to simulate every possible match scenario, helping the model learn more general patterns.

**Function: `augment_data_with_team_swaps`**

```python
def augment_data_with_team_swaps(data):
    """Creates an augmented dataset by swapping the positions of teams, effectively doubling the dataset size."""
    swapped_data = data.copy()
    swapped_data['team1'], swapped_data['team2'] = data['team2'], data['team1']
    swapped_data['result'] = 1 - data['result']  # Invert the result to match swapped teams
    swapped_data['toss_winner_is_team1'] = 1 - data['toss_winner_is_team1']
    return pd.concat([data, swapped_data], ignore_index=True)
```

**Key Actions**:
* Swaps `team1` and `team2`
* Reverses the match result to maintain consistency
* Doubles the dataset size by concatenating the original and swapped data

#### 2. Load and Prepare Data

This function prepares the data for the model by loading, filtering, augmenting, and encoding it.

**Function: `load_and_prepare_data`**

```python
def load_and_prepare_data():
    """Loads data, filters unnecessary features, augments, and encodes it for model training."""
    data = pd.read_csv('../data/processed_data.csv')
    excluded_cols = ['match_id', 'city', ...]  # List of columns to exclude from training
    data = data[[col for col in data.columns if col not in excluded_cols]]
    augmented_data = augment_data_with_team_swaps(data)
    for col in ['team1', 'team2', 'venue', ...]:  # Categorical columns to encode
        le = LabelEncoder()
        augmented_data[col] = le.fit_transform(augmented_data[col].fillna('Unknown'))
    X = augmented_data.drop('result', axis=1)
    y = augmented_data['result']
    return X, y
```

**Key Actions**:
* Filters out irrelevant features
* Encodes categorical variables
* Splits the data into features (X) and the target (y)

#### 3. Train the RandomForest Model

This section sets up the RandomForest classifier, optimizes its parameters with GridSearchCV, and trains it.

**Function: `train_model`**

```python
def train_model():
    """Trains and optimizes a RandomForest classifier using GridSearchCV."""
    X, y = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_grid = {'n_estimators': [200, 300], 'max_depth': [8, 10], ...}
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Test accuracy: {grid_search.score(X_test, y_test):.2f}")
```

**Key Actions**:
* Splits the data into training and testing sets
* Defines a grid of parameters to find the optimal settings
* Trains the model and evaluates its accuracy on the test set

#### 4. Save the Model

This function saves the trained model along with its metadata for later use in making predictions.

**Function: `save_model`**

```python
def save_model(grid_search):
    """Saves the trained model and its feature encoders."""
    joblib.dump({
        'model': grid_search.best_estimator_,
        'encoders': encoders
    }, '../models/cricket_predictor_v3.pkl')
    print("Model saved successfully.")
```

**Key Actions**:
* Uses `joblib` to serialize and save the model and its encoders

### Web Interface (`streamlit_main.py`)

* Purpose: Provides a graphical user interface to interact with the trained model
* Features:
  * User Input: Allows users to input match conditions such as teams, venue, and toss decisions
  * Model Interaction: Processes input through the model to predict outcomes
  * Result Display: Shows the predicted probabilities and outcomes in a user-friendly manner

### Explanation to Streamlit Script

#### Part 1: Importing Libraries

**Code Snippet:**
```python
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
```

**Explanation:**
* streamlit (`st`): Used to create and control the web app's interface
* pandas (`pd`): For data manipulation, particularly to format the input data for the model
* joblib: For loading the trained machine learning model
* datetime: To fetch the current date for real-time data inputs like `season`

#### Part 2: Loading the Model

**Code Snippet:**
```python
def load_model():
    try:
        model_artifacts = joblib.load('models/cricket_predictor_v3.pkl')
        return model_artifacts
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None
```

**Explanation:**
* Tries to load the serialized model (`cricket_predictor_v3.pkl`)
* Displays an error in the Streamlit interface if the loading fails

#### Part 3: Preparing the Input Data

**Code Snippet:**
```python
def prepare_input(team1, team2, toss_winner, toss_decision, model_artifacts):
    input_data = {
        'season': str(datetime.now().year),  # Current year as season
        'team1': team1,  # First team
        'team2': team2,  # Second team
        # More fields...
    }
    df = pd.DataFrame([input_data])  # Convert dictionary to DataFrame
    encoders = model_artifacts['encoders']  # Get encoders from the loaded model
    # Encode categorical fields
    for col in ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'season']:
        df[col] = encoders[col].transform(df[col].astype(str))
    return df[model_artifacts['feature_columns']]  # Return only the necessary columns
```

**Explanation:**
* Constructs a dictionary with match details
* Converts this dictionary into a DataFrame
* Uses pre-fitted encoders (loaded with the model) to transform categorical features into machine-readable formats

#### Part 4: Main Application Function

**Code Snippet:**
```python
def main():
    st.set_page_config(page_title="Cricket Predictor", page_icon="🏏", layout="centered")
    model_artifacts = load_model()
    if model_artifacts:
        teams = {
            "India": "🇮🇳 India", "Australia": "🇦🇺 Australia", # Dictionary of teams with flags
            # More teams...
        }
        team1 = st.selectbox("Choose Team 1", list(teams.values()))
        team2 = st.selectbox("Choose Team 2", list(teams.values()), index=1)
        toss_winner = st.selectbox("Toss Winner", [team1, team2])
        toss_decision = st.selectbox("Toss Decision", ["Batting First", "Fielding First"])
        if st.button("PREDICT WINNER 🎯"):
            if team1 == team2:
                st.error("Please select different teams.")
            else:
                features = prepare_input(team1, team2, toss_winner, toss_decision, model_artifacts)
                probs = model_artifacts['model'].predict_proba(features)[0]
                st.metric(f"{team1} Win Probability", f"{probs[1] * 100:.1f}%")
                st.metric(f"{team2} Win Probability", f"{probs[0] * 100:.1f}%")
```

**Explanation:**
* Page Setup: Configures the Streamlit page with a title and icon
* Model Loading: Attempts to load the machine learning model and continues if successful
* User Inputs: Allows the user to select teams, the toss winner, and the toss decision via dropdown menus
* Prediction Trigger: A button that, when clicked, checks if different teams are selected, prepares the input data, makes a prediction using the model, and displays the probabilities of each team winning
## Usage Instructions

1. Set up the Environment:
```bash
pip install -r requirements.txt
```

2. Prepare the Data:
* Download data from [Kaggle](https://www.kaggle.com/datasets/utkarshtomar736/odi-mens-cricket-match-data-2002-2023)
* Run preprocessing:
```bash
python scripts/preprocess.py
```

3. Train the Model:
```bash
python scripts/train_model.py
```

4. Launch the Application:
```bash
streamlit run streamlit_main.py
```

## System Performance

* Prediction Accuracy: 70-75% on test data
* Key Predictive Features:
  * Team win rates
  * Head-to-head performance
  * Venue statistics
  * Toss decisions
  * Recent form

## Hosted Application

Access the live application at:
[Cricket Match Prediction System](https://matchresultprediction.streamlit.app)

## Future Enhancements

1. Model Improvements:
   * Explore advanced algorithms
   * Implement ensemble methods
   * Add real-time updates

2. Feature Expansion:
   * Player statistics
   * Weather conditions
   * Team composition
   * Historical performance trends

## Dataset Information

The project uses ODI cricket match data from 2002-2023, including:
* Ball-by-ball match data
* Match metadata and results
* Team and player information
* Venue and condition details

[Dataset Source](https://www.kaggle.com/datasets/utkarshtomar736/odi-mens-cricket-match-data-2002-2023)

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