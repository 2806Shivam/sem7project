from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

app = Flask(__name__)

# Load and preprocess data
def load_data():
    world_cup = pd.read_csv("ODI Rankings - World_cup_2023 (1).csv")
    results = pd.read_csv("Results (2015-2024) - results.csv")

    results.drop(columns=['Date', 'Margin', 'Ground'], axis=1, inplace=True, errors='ignore')

    world_cup_teams = ['England', 'South Africa', 'West Indies', 'Pakistan', 'New Zealand',
                       'Sri Lanka', 'Afganistan', 'Australia', 'Bangladesh', 'India']

    df_teams_1 = results[results['Team_1'].isin(world_cup_teams)]
    df_teams_2 = results[results['Team_2'].isin(world_cup_teams)]
    df_winners = results[results['Winner'].isin(world_cup_teams)]

    df_team = pd.concat((df_teams_1, df_teams_2, df_winners), axis=0)
    df_team['Winning'] = np.where(df_team['Winner'] == df_team['Team_1'], 1, 2)
    df_team.drop(columns=['Winner'], axis=1, inplace=True)

    # One-Hot Encoding
    df_team = pd.get_dummies(df_team, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'], dtype=int)

    # Split data into training and test sets
    x = df_team.drop(columns=['Winning'], axis=1)
    y = df_team['Winning']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=34)

    # Feature scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=34)
    x_train, y_train = smote.fit_resample(x_train, y_train)

    # Initialize RandomForestClassifier
    rf = RandomForestClassifier(random_state=34)
    rf.fit(x_train, y_train)

    # Calculate model accuracy
    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred) * 100

    return rf, scaler, x.columns.tolist(), accuracy

# Load model, scaler, feature names, and accuracy
model, scaler, feature_names, accuracy = load_data()

@app.route('/')
def index():
    return render_template('index.html', accuracy=f"{accuracy:.2f}")

@app.route('/predict', methods=['POST'])
def predict():
    team_1 = request.form['team_1']
    team_2 = request.form['team_2']

    # Create a DataFrame for the input match with one-hot encoded teams
    match_data = pd.DataFrame(columns=feature_names)

    # Set default values for all columns
    match_data.loc[0] = [0] * len(feature_names)

    # Set 1 for the selected teams
    if f'Team_1_{team_1}' in match_data.columns:
        match_data[f'Team_1_{team_1}'] = 1
    if f'Team_2_{team_2}' in match_data.columns:
        match_data[f'Team_2_{team_2}'] = 1

    # Scale the input match data
    match_data_scaled = scaler.transform(match_data)

    # Predict the winner (1 = Team_1 wins, 2 = Team_2 wins)
    prediction = model.predict(match_data_scaled)[0]

    predicted_winner = team_1 if prediction == 1 else team_2
    return render_template('index.html', accuracy=f"{accuracy:.2f}", predicted_winner=predicted_winner, team_1=team_1, team_2=team_2)
