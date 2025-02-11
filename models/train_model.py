import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

def augment_data_with_team_swaps(data):
    """Create augmented dataset with swapped team positions"""
    # Copy original data
    swapped_data = data.copy()
    
    # Swap team positions
    swapped_data['team1'], swapped_data['team2'] = data['team2'], data['team1']
    
    # Adjust result (1 becomes 0 and 0 becomes 1)
    swapped_data['result'] = 1 - data['result']
    
    # Adjust toss-related features
    swapped_data['toss_winner_is_team1'] = 1 - data['toss_winner_is_team1']
    
    # Update win rates and matches to match new team positions
    swapped_data['team1_win_rate'], swapped_data['team2_win_rate'] = \
        data['team2_win_rate'], data['team1_win_rate']
    swapped_data['team1_matches'], swapped_data['team2_matches'] = \
        data['team2_matches'], data['team1_matches']
    
    # Combine original and swapped data
    augmented_data = pd.concat([data, swapped_data], ignore_index=True)
    
    return augmented_data

def load_and_prepare_data():
    """Load and prepare data for training"""
    data = pd.read_csv('../data/processed_data.csv')
    
    # Remove post-match features
    excluded_cols = [
        'match_id', 'city', 'player_of_match', 
        'umpire1', 'umpire2', 'umpire3',
        'win_by_runs', 'win_by_wickets',
        'total_runs_1', 'total_runs_2', 'total_runs_3', 'total_runs_4',
        'wickets_1', 'wickets_2', 'wickets_3', 'wickets_4',
        'balls_faced_1', 'balls_faced_2', 'balls_faced_3', 'balls_faced_4',
        'extras_1', 'extras_2', 'extras_3', 'extras_4',
        'dl_applied'
    ]
    
    # Select features
    feature_cols = [col for col in data.columns if col not in excluded_cols]
    data = data[feature_cols]
    
    # Augment data with team swaps
    augmented_data = augment_data_with_team_swaps(data)
    
    # Encode categorical features
    categorical_cols = ['team1', 'team2', 'venue', 'toss_winner', 
                       'toss_decision', 'season']
    encoders = {}
    
    for col in categorical_cols:
        if col in augmented_data.columns:
            le = LabelEncoder()
            augmented_data[col] = augmented_data[col].fillna('Unknown')
            augmented_data[col] = le.fit_transform(augmented_data[col])
            encoders[col] = le
    
    # Split features and target
    X = augmented_data.drop('result', axis=1)
    y = augmented_data['result']
    
    return X, y, encoders, X.columns

def train_model():
    """Train the model with augmented data"""
    print("Loading and preparing data...")
    X, y, encoders, feature_columns = load_and_prepare_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [8, 10],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4]
    }
    
    # Initialize and train model
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print("\nTraining model...")
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}")
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    print("\nTest Set Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))
    
    # Save model artifacts
    model_artifacts = {
        'model': best_model,
        'feature_columns': feature_columns,
        'encoders': encoders
    }
    
    joblib.dump(model_artifacts, '../models/cricket_predictor_v3.pkl')
    print("\nModel artifacts saved successfully")

if __name__ == "__main__":
    train_model()