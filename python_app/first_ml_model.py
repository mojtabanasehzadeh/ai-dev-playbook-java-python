# first_ml_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data (as an engineering manager, you'll recognize this)
data = {
    'team_size': [3, 5, 8, 4, 6, 7, 3, 9, 5, 8],
    'complexity_score': [2, 7, 9, 3, 6, 8, 1, 10, 4, 7],
    'on_time': [1, 0, 0, 1, 0, 0, 1, 0, 1, 0]  # 1=on time, 0=delayed
}

df = pd.DataFrame(data)
X = df[['team_size', 'complexity_score']]
y = df['on_time']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model accuracy: {accuracy:.2f}")
new_project = pd.DataFrame({'team_size': [6], 'complexity_score': [5]})
print(f"Prediction for team_size=3, complexity=10: {model.predict(new_project)[0]}")
