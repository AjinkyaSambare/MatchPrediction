import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score

# Global variables for team and venue data
TEAM_RATINGS = {
    "India": 0.85,
    "Australia": 0.82,
    "England": 0.78,
    "South Africa": 0.75,
    "New Zealand": 0.74,
    "Pakistan": 0.72,
    "Sri Lanka": 0.68,
    "Bangladesh": 0.65,
    "West Indies": 0.63,
    "Afghanistan": 0.60,
    "Zimbabwe": 0.55,
    "Ireland": 0.52
}

HOME_VENUES = {
    "MCG, Melbourne": "Australia",
    "Lord's, London": "England",
    "Eden Gardens, Kolkata": "India",
    "SCG, Sydney": "Australia",
    "Wanderers, Johannesburg": "South Africa",
    "Gaddafi Stadium, Lahore": "Pakistan",
    "R. Premadasa Stadium, Colombo": "Sri Lanka",
    "Shere Bangla Stadium, Dhaka": "Bangladesh",
    "Wankhede Stadium, Mumbai": "India",
    "M. Chinnaswamy Stadium, Bangalore": "India",
    "Adelaide Oval, Adelaide": "Australia",
    "Basin Reserve, Wellington": "New Zealand",
    "Kensington Oval, Barbados": "West Indies"
}

TEAMS = list(TEAM_RATINGS.keys())
VENUES = list(HOME_VENUES.keys()) + ["Dubai International Stadium", "neutral"]

def load_model():
    """Load model artifacts"""
    try:
        model_artifacts = joblib.load('/Users/Ajinkya25/Documents/Idea-Labs/TrainModels/MatchPrediction/models/cricket_predictor_v3.pkl')
        return model_artifacts
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def prepare_input(team1, team2, toss_winner, toss_decision, venue, model_artifacts):
    """Prepare input data with dynamic team statistics"""
    
    # Calculate home advantage
    home_advantage = 0.1  # 10% advantage for home team
    venue_advantage = 0.0
    if venue in HOME_VENUES:
        if HOME_VENUES[venue] == team1:
            venue_advantage = home_advantage
        elif HOME_VENUES[venue] == team2:
            venue_advantage = -home_advantage
    
    # Calculate toss advantage
    toss_advantage = 0.05  # 5% advantage for winning the toss
    toss_factor = toss_advantage if toss_winner == team1 else -toss_advantage
    
    # Calculate base win rates using team ratings
    team1_win_rate = TEAM_RATINGS.get(team1, 0.5)
    team2_win_rate = TEAM_RATINGS.get(team2, 0.5)
    
    # Adjust win rates based on venue and toss
    team1_adjusted_rate = min(0.95, max(0.05, team1_win_rate + venue_advantage + toss_factor))
    team2_adjusted_rate = min(0.95, max(0.05, team2_win_rate - venue_advantage - toss_factor))
    
    # Calculate matches played (hypothetical data based on team ratings)
    base_matches = 100
    team1_matches = int(base_matches * TEAM_RATINGS.get(team1, 0.5))
    team2_matches = int(base_matches * TEAM_RATINGS.get(team2, 0.5))
    
    input_data = {
        'season': str(datetime.now().year),
        'team1': team1,
        'team2': team2,
        'toss_winner': toss_winner,
        'toss_decision': toss_decision,
        'venue': venue,
        'year': datetime.now().year,
        'month': datetime.now().month,
        'day_of_week': datetime.now().weekday(),
        'toss_winner_is_team1': 1 if toss_winner == team1 else 0,
        'toss_winner_batted_first': 1 if toss_decision == 'bat' else 0,
        'team1_matches': team1_matches,
        'team2_matches': team2_matches,
        'team1_win_rate': team1_adjusted_rate,
        'team2_win_rate': team2_adjusted_rate,
        'matches_at_venue': 50 if venue != 'neutral' else 10
    }
    
    df = pd.DataFrame([input_data])
    
    # Calculate win probabilities directly based on adjusted rates
    team1_prob = (team1_adjusted_rate / (team1_adjusted_rate + team2_adjusted_rate))
    team2_prob = 1 - team1_prob
    
    return df, team1_prob, team2_prob

def main():
    # Page config
    st.set_page_config(
        page_title="Cricket Match Prediction System",
        page_icon="🏏",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS with dark mode support
    st.markdown("""
        <style>
        /* Base Theme Variables */
        :root {
            --background-color: var(--st-color-background-primary);
            --text-color: var(--st-color-text-primary);
            --secondary-text: var(--st-color-text-secondary);
            --card-bg: var(--st-color-background-secondary);
            --border-color: var(--st-color-border-primary);
            --primary-color: #007bff;
            --success-color: #28a745;
            --hover-color: #0056b3;
        }

        /* Main container styling */
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
            transition: all 0.3s ease;
        }
        
        /* Header styling */
        .header-container {
            background-color: var(--card-bg);
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
            border: 1px solid var(--border-color);
        }
        
        .header-title {
            color: var(--text-color);
            font-size: 2.2rem;
            font-weight: 600;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        /* Content card styling */
        .content-card {
            background-color: var(--card-bg);
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid var(--border-color);
            margin-bottom: 1.5rem;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: var(--primary-color);
            color: white;
            font-weight: 500;
            padding: 0.5rem 2rem;
            border-radius: 5px;
            border: none;
            transition: all 0.3s;
            width: 100%;
        }
        
        .stButton>button:hover {
            background-color: var(--hover-color);
            box-shadow: 0 2px 6px var(--border-color);
        }
        
        /* Prediction results styling */
        .prediction-results {
            background-color: var(--card-bg);
            padding: 1.5rem;
            border-radius: 8px;
            margin-top: 2rem;
            text-align: center;
            border: 1px solid var(--border-color);
        }
        
        .winner-announcement {
            color: var(--success-color);
            font-size: 1.5rem;
            font-weight: 600;
            margin-top: 1rem;
        }
        
        /* Select box styling */
        .stSelectbox > div > div {
            background-color: var(--card-bg);
            color: var(--text-color);
            border-color: var(--border-color);
        }
        
        /* Metrics styling */
        .stMetric {
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            padding: 1rem;
            border-radius: 8px;
        }
        
        /* Info box styling */
        .stInfo {
            background-color: var(--card-bg);
            border-color: var(--border-color);
            color: var(--text-color);
        }
        
        /* Handle dark mode specifics */
        @media (prefers-color-scheme: dark) {
            .stMarkdown {
                color: var(--text-color);
            }
            
            .stMetric [data-testid="stMetricValue"] {
                color: var(--text-color);
            }
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div class="header-container">
            <h1 class="header-title">Cricket Match Prediction System</h1>
            <p style="text-align: center; color: var(--secondary-text);">
                Advanced analytics for cricket match outcome prediction
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model_artifacts = load_model()
    
    if model_artifacts:
        # Main prediction interface
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("### Match Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("Team 1", TEAMS)
        with col2:
            team2 = st.selectbox("Team 2", TEAMS, index=1)
        
        # Match details with venue
        col3, col4, col5 = st.columns(3)
        with col3:
            toss_winner = st.selectbox("Toss Winner", [team1, team2])
        with col4:
            toss_decision = st.selectbox(
                "Toss Decision",
                ["Batting First", "Fielding First"]
            )
        with col5:
            venue = st.selectbox("Venue", VENUES)
        
        # Convert toss decision
        toss_decision = 'bat' if 'Batting' in toss_decision else 'field'
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Prediction button
        if st.button("Generate Prediction", key="predict_button"):
            if team1 == team2:
                st.error("Please select different teams for prediction")
            else:
                try:
                    # Get feature data and probabilities
                    features, team1_prob, team2_prob = prepare_input(team1, team2, toss_winner, toss_decision, venue, model_artifacts)
                    
                    # Display prediction results
                    st.markdown('<div class="prediction-results">', unsafe_allow_html=True)
                    st.markdown("### Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(f"{team1} Win Probability", f"{team1_prob*100:.1f}%")
                    with col2:
                        st.metric(f"{team2} Win Probability", f"{team2_prob*100:.1f}%")
                    
                    # Winner announcement
                    winner = team1 if team1_prob > team2_prob else team2
                    win_prob = max(team1_prob, team2_prob) * 100
                    
                    st.markdown(f"""
                        <div class="winner-announcement">
                            Predicted Winner: {winner}<br>
                            <small style="font-size: 1rem; color: var(--secondary-text);">
                                Win Probability: {win_prob:.1f}%
                            </small>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Factors affecting prediction
                    st.markdown("### Key Factors Affecting Prediction")
                    
                    # Display venue advantage if applicable
                    if venue in HOME_VENUES:
                        home_team = HOME_VENUES[venue]
                        if home_team in [team1, team2]:
                            st.info(f"🏟️ Home Advantage: {home_team} has home advantage at {venue}")
                    
                    # Display toss advantage
                    st.info(f"🎲 Toss Advantage: {toss_winner} won the toss and chose to {toss_decision} first")
                    
                    # Display team ratings
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"📊 {team1} Rating: {TEAM_RATINGS.get(team1, 0.5)*100:.1f}%")
                    with col2:
                        st.info(f"📊 {team2} Rating: {TEAM_RATINGS.get(team2, 0.5)*100:.1f}%")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
    else:
        st.error("Unable to load the prediction model. Please check the model file.")

if __name__ == "__main__":
    main()