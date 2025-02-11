import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

def load_model():
    """Load model artifacts"""
    try:
        model_artifacts = joblib.load('../models/cricket_predictor_v3.pkl')
        return model_artifacts
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def prepare_input(team1, team2, toss_winner, toss_decision, model_artifacts):
    """Prepare input data with all required features"""
    input_data = {
        'season': str(datetime.now().year),
        'team1': team1,
        'team2': team2,
        'toss_winner': toss_winner,
        'toss_decision': toss_decision,
        'venue': 'neutral',
        'year': datetime.now().year,
        'month': datetime.now().month,
        'day_of_week': datetime.now().weekday(),
        'toss_winner_is_team1': 1 if toss_winner == team1 else 0,
        'toss_winner_batted_first': 1 if toss_decision == 'bat' else 0,
        'team1_matches': 100,
        'team2_matches': 100,
        'team1_win_rate': 0.5,
        'team2_win_rate': 0.5,
        'matches_at_venue': 10
    }
    
    df = pd.DataFrame([input_data])
    encoders = model_artifacts['encoders']
    categorical_cols = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'season']
    
    for col in categorical_cols:
        if col in encoders:
            try:
                df[col] = encoders[col].transform(df[col].astype(str))
            except:
                df[col] = encoders[col].transform([encoders[col].classes_[0]])
    
    feature_columns = model_artifacts['feature_columns']
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
            
    return df[feature_columns]

def main():
    # Page config
    st.set_page_config(
        page_title="Cricket Predictor",
        page_icon="🏏",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Enhanced CSS styling
    st.markdown("""
        <style>
        /* Overall page styling */
        .stApp {
            background: linear-gradient(180deg, #1a1a1a 0%, #0D0D0D 100%);
        }
        
        /* Header styling */
        h1 {
            color: #ffffff;
            text-align: center;
            font-size: 3rem !important;
            padding: 1.5rem 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        /* Card styling for the main content */
        .main-content {
            
        }
        
        /* Selectbox styling */
        div[data-testid="stSelectbox"] > div {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        /* Button styling */
        .stButton>button {
            width: 100%;
            padding: 0.75rem;
            font-size: 1.2rem;
            background: linear-gradient(45deg, #FF4B4B, #FF6B6B);
            color: white;
            border: none;
            border-radius: 8px;
            margin-top: 2rem;
            transition: all 0.3s ease;
            text-transform: uppercase;
            font-weight: bold;
            letter-spacing: 1px;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
        }
        
        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 2.5rem !important;
            color: #ffffff !important;
            text-align: center;
        }
        
        /* Success message styling */
        .stSuccess {
            background: rgba(0, 255, 0, 0.1);
            border: 1px solid rgba(0, 255, 0, 0.2);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            backdrop-filter: blur(5px);
        }
        
        /* Team flags */
        .team-flag {
            width: 30px;
            height: 20px;
            margin-right: 10px;
            vertical-align: middle;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # App Header
    st.markdown("""
        <h1>Cricket Match Predictor 🏏</h1>
        <div class='main-content'>
    """, unsafe_allow_html=True)
    
    # Load model
    model_artifacts = load_model()
    
    if model_artifacts:
        # Teams with flags (emoji flags as placeholders)
        teams = {
            "India": "🇮🇳 India",
            "Australia": "🇦🇺 Australia",
            "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 England",
            "South Africa": "🇿🇦 South Africa",
            "New Zealand": "🇳🇿 New Zealand",
            "Pakistan": "🇵🇰 Pakistan",
            "Sri Lanka": "🇱🇰 Sri Lanka",
            "Bangladesh": "🇧🇩 Bangladesh"
        }
        
        # Team Selection with enhanced layout
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Team 1")
            team1 = st.selectbox("", list(teams.values()), key="team1", label_visibility="collapsed")
            team1 = team1.split(" ", 1)[1]  # Remove emoji
            
        with col2:
            st.markdown("### Team 2")
            team2 = st.selectbox("", list(teams.values()), key="team2", 
                               index=1, label_visibility="collapsed")
            team2 = team2.split(" ", 1)[1]  # Remove emoji
        
        # Toss Details
        st.markdown("### Match Details")
        col3, col4 = st.columns(2)
        with col3:
            toss_winner = st.selectbox("Toss Winner", [team1, team2])
        with col4:
            choice = st.selectbox("Toss Decision", ["Batting First", "Fielding First"])
        
        toss_decision = 'bat' if 'Batting' in choice else 'field'
        
        # Prediction Button
        if st.button("PREDICT WINNER 🎯"):
            if team1 == team2:
                st.error("Please select different teams")
            else:
                try:
                    features = prepare_input(team1, team2, toss_winner, toss_decision, model_artifacts)
                    model = model_artifacts['model']
                    probs = model.predict_proba(features)[0]
                    
                    # Results with enhanced visualization
                    st.markdown("### Prediction Results")
                    col5, col6 = st.columns(2)
                    
                    with col5:
                        prob1 = probs[1] * 100
                        st.metric(f"{teams[team1].split(' ')[0]} {team1}", 
                                f"{prob1:.1f}%")
                        
                    with col6:
                        prob2 = probs[0] * 100
                        st.metric(f"{teams[team2].split(' ')[0]} {team2}", 
                                f"{prob2:.1f}%")
                    
                    # Winner announcement with animation
                    winner = team1 if probs[1] > probs[0] else team2
                    winner_emoji = teams[winner].split(" ")[0]
                    st.markdown(f"""
                        <div class='success-message' style='text-align: center; padding: 20px;'>
                            <h2 style='color: #00ff00; font-size: 1.5rem;'>
                                {winner_emoji} {winner} is predicted to win! 🏆
                            </h2>
                        </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error("Prediction failed. Please check your inputs.")
    
    # Close main content div
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()