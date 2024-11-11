import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load your data
world_cup = pd.read_csv("ODI Rankings - World_cup_2023 (1).csv")
results = pd.read_csv("Results (2015-2024) - results.csv")

# Drop unnecessary columns
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
smote = SMOTE()
x_train, y_train = smote.fit_resample(x_train, y_train)

# Initialize RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Function to predict winner for new matches
def predict_winner(team_1, team_2):
    # Get the columns used for training
    all_columns = list(x.columns)
    
    # Create a dataframe for the input match with one-hot encoded teams
    match_data = pd.DataFrame({
        'Team_1_' + team_1: [1],
        'Team_2_' + team_2: [1]
    })
    
    # Add missing columns (all other teams not involved in the match) and fill with 0
    for col in all_columns:
        if col not in match_data.columns:
            match_data[col] = 0

    # Reorder columns to match the original training data order
    match_data = match_data[all_columns]
    
    # Scale the input match data
    match_data_scaled = scaler.transform(match_data)
    
    # Predict the winner (1 = Team_1 wins, 2 = Team_2 wins)
    prediction = rf.predict(match_data_scaled)[0]
    
    if prediction == 1:
        return team_1
    else:
        return team_2

# Example: Predict the winner between India and Australia
team_1 = 'England'
team_2 = 'Pakistan'
predicted_winner = predict_winner(team_1, team_2)
print(f'Predicted winner between {team_1} and {team_2}: {predicted_winner}')

# You can add more matches and call the function as needed to predict other matchups
