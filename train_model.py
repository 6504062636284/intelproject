from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

# Load data
try:
    df = pd.read_csv('solution.csv')
except FileNotFoundError:
    print("Error: 'solution.csv' not found. Please ensure it’s in the current directory.")
    exit()

X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Test the new model
input_data = {'GrLivArea': 1500, 'BedroomAbvGr': 3, 'FullBath': 2, 'TotRmsAbvGrd': 1, 'Fireplaces': 0, 'OverallCond': 3}
for col in X.columns:
    if col not in input_data:
        input_data[col] = 0
input_df = pd.DataFrame([input_data], columns=X.columns)
input_scaled = scaler.transform(input_df)
print("Prediction for default input:", model.predict(input_scaled))

input_data2 = {'GrLivArea': 3000, 'BedroomAbvGr': 5, 'FullBath': 4, 'TotRmsAbvGrd': 2, 'Fireplaces': 1, 'OverallCond': 5}
for col in X.columns:
    if col not in input_data2:
        input_data2[col] = 0
input_df2 = pd.DataFrame([input_data2], columns=X.columns)
input_scaled2 = scaler.transform(input_df2)
print("Prediction for larger input:", model.predict(input_scaled2))