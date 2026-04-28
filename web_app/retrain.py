import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("AI_Impact_Student_Life_2026.csv")
new_data = pd.read_csv("new_data.csv")

df = pd.concat([df, new_data], ignore_index=True)

X = df.drop("GPA_Post_AI", axis=1)
y = df["GPA_Post_AI"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, "gpa_model.pkl")

print("Model retrained successfully!")