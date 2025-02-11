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

## Screenshots

### Web Interface Overview

![Web Interface Overview](https://github.com/user-attachments/assets/b0c40f34-f1a6-4f78-abd9-326afc4520e5)

*Description: This screenshot displays the main page of the Streamlit web application, where users can input match details for prediction.*

### Prediction Results

![Prediction Results](path_to_screenshot_here)

*Description: After submitting the match details, this screenshot shows the predicted outcome including the probabilities for each team winning the match.*

### Model Accuracy and Metrics

![Model Metrics](path_to_screenshot_here)

*Description: This section of the application provides insights into the model's performance metrics and feature importance, giving users an understanding of how predictions are derived.*

### Interactive Features

![Interactive Features](path_to_screenshot_here)

*Description: This screenshot highlights additional interactive features of the application, such as options to modify or explore different data inputs and settings.*

---

**Note:** Replace `path_to_screenshot_here` with the actual paths to the images of your application. You may upload these images to a public web server or include them in your repository and link directly to them.

This section will enhance the README by providing visual context to your textual descriptions, making it easier for others to understand and appreciate the capabilities of your Cricket Match Prediction System.