import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

def load_data():
    """Load both match data and match info datasets"""
    match_data = pd.read_csv('../data/odi_match_data.csv', low_memory=False)
    match_info = pd.read_csv('../data/odi_match_info.csv', low_memory=False)
    return match_data, match_info

def filter_special_teams(data):
    """Filter out exhibition and special teams"""
    special_teams = ['Asia XI', 'World XI', 'ICC World XI', 'Africa XI']
    
    # Remove matches involving special teams
    mask = ~(
        data['team1'].isin(special_teams) | 
        data['team2'].isin(special_teams)
    )
    filtered_data = data[mask].copy()
    
    # Print removed matches info
    removed_count = len(data) - len(filtered_data)
    print(f"\nRemoved {removed_count} matches involving special teams")
    
    return filtered_data

def aggregate_match_data(match_data):
    """Aggregate ball-by-ball data to match level statistics"""
    try:
        # Group by match_id and compute match-level statistics
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
        
        # Rename columns
        match_stats = match_stats.rename(columns={
            'wicket_type': 'wickets',
            'ball': 'balls_faced'
        })
        
        # Calculate total runs
        match_stats['total_runs'] = (match_stats['runs_off_bat'] + match_stats['extras'])
        
        # Pivot the data to get separate columns for each innings
        match_summary = match_stats.pivot(
            index='match_id',
            columns='innings',
            values=['total_runs', 'wickets', 'balls_faced', 'extras']
        ).reset_index()
        
        # Flatten column names
        match_summary.columns = [
            f'{col[0]}_{col[1]}' if col[1] else col[0] 
            for col in match_summary.columns
        ]
        
        return match_summary
        
    except Exception as e:
        print(f"Error in aggregate_match_data: {str(e)}")
        raise

def create_team_features(data):
    """Create team performance features"""
    try:
        # Sort by date to maintain temporal order
        data = data.sort_values('date')
        
        # Calculate rolling stats for each team
        team_stats = {}
        
        # Get unique teams
        all_teams = pd.concat([data['team1'], data['team2']]).unique()
        
        for team in all_teams:
            # Get matches where team participated
            team_matches = data[
                (data['team1'] == team) | 
                (data['team2'] == team)
            ].copy()
            
            # Calculate win rate
            team_wins = team_matches[team_matches['winner'] == team]
            win_rate = len(team_wins) / len(team_matches) if len(team_matches) > 0 else 0
            
            # Store team statistics
            team_stats[team] = {
                'matches_played': len(team_matches),
                'win_rate': win_rate
            }
        
        # Add features to dataset
        data['team1_matches'] = data['team1'].map(lambda x: team_stats.get(x, {}).get('matches_played', 0))
        data['team2_matches'] = data['team2'].map(lambda x: team_stats.get(x, {}).get('matches_played', 0))
        data['team1_win_rate'] = data['team1'].map(lambda x: team_stats.get(x, {}).get('win_rate', 0))
        data['team2_win_rate'] = data['team2'].map(lambda x: team_stats.get(x, {}).get('win_rate', 0))
        
        return data
        
    except Exception as e:
        print(f"Error in create_team_features: {str(e)}")
        raise

def process_venue_features(data):
    """Process venue-related features"""
    try:
        # Calculate venue statistics
        venue_stats = data.groupby('venue').agg({
            'match_id': 'count'
        }).reset_index()
        
        venue_stats.columns = ['venue', 'matches_at_venue']
        
        # Merge venue statistics back
        data = pd.merge(data, venue_stats, on='venue', how='left')
        
        return data
        
    except Exception as e:
        print(f"Error in process_venue_features: {str(e)}")
        raise

def create_match_features(data):
    """Create match-specific features"""
    try:
        # Convert date to datetime
        data['date'] = pd.to_datetime(data['date'])
        
        # Extract temporal features
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day_of_week'] = data['date'].dt.dayofweek
        
        # Create toss-related features
        data['toss_winner_is_team1'] = (data['toss_winner'] == data['team1']).astype(int)
        data['toss_winner_batted_first'] = (
            ((data['toss_winner'] == data['team1']) & (data['toss_decision'] == 'bat')) |
            ((data['toss_winner'] == data['team2']) & (data['toss_decision'] == 'field'))
        ).astype(int)
        
        return data
        
    except Exception as e:
        print(f"Error in create_match_features: {str(e)}")
        raise

def encode_categorical_features(data):
    """Encode categorical features"""
    try:
        categorical_columns = [
            'team1', 'team2', 'venue', 'toss_winner', 
            'toss_decision', 'winner', 'season'
        ]
        
        encoders = {}
        for column in categorical_columns:
            if column in data.columns:
                le = LabelEncoder()
                data[column] = data[column].fillna('Unknown')
                data[column] = le.fit_transform(data[column].astype(str))
                encoders[column] = le
        
        return data, encoders
        
    except Exception as e:
        print(f"Error in encode_categorical_features: {str(e)}")
        raise

def preprocess_data(match_data, match_info):
    """Main preprocessing function"""
    try:
        print("Aggregating ball-by-ball data to match level...")
        match_summary = aggregate_match_data(match_data)
        
        print("Merging with match info...")
        match_info.rename(columns={'id': 'match_id'}, inplace=True)
        combined_data = pd.merge(
            match_summary,
            match_info,
            on='match_id',
            how='inner'
        )
        
        print("Filtering special teams...")
        combined_data = filter_special_teams(combined_data)
        
        print("Creating match features...")
        combined_data = create_match_features(combined_data)
        
        print("Creating team features...")
        combined_data = create_team_features(combined_data)
        
        print("Processing venue features...")
        combined_data = process_venue_features(combined_data)
        
        # Create target variable (1 if team1 wins, 0 if team2 wins)
        print("Creating target variable...")
        combined_data['result'] = (combined_data['winner'] == combined_data['team1']).astype(int)
        
        print("Encoding categorical features...")
        combined_data, encoders = encode_categorical_features(combined_data)
        
        # Remove unnecessary columns
        columns_to_drop = [
            'date',  # Already extracted features
            'winner'  # Already encoded in result
        ]
        combined_data.drop(columns_to_drop, axis=1, errors='ignore', inplace=True)
        
        # Save encoders
        pd.to_pickle(encoders, '../data/label_encoders.pkl')
        
        print("\nPreprocessing completed successfully!")
        print(f"Final dataset shape: {combined_data.shape}")
        print("\nColumns in processed dataset:")
        print(combined_data.columns.tolist())
        
        return combined_data
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("Loading data...")
        match_data, match_info = load_data()
        
        print("\nStarting preprocessing...")
        combined_data = preprocess_data(match_data, match_info)
        
        # Save processed data
        combined_data.to_csv('../data/processed_data.csv', index=False)
        print("\nProcessed data saved successfully!")
        
        # Print class distribution
        print("\nTarget variable distribution:")
        print(combined_data['result'].value_counts(normalize=True))
        
    except Exception as e:
        print(f"An error occurred: {e}")