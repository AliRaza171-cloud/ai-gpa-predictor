import joblib
import numpy as np
import pandas as pd


# here we load our model
model=joblib.load("gpa_model.pkl")
features=joblib.load("features.pkl")
# sample input
input_data = {
    "Age": 21,
    "Task_Frequency_Daily": 5,
    "GPA_Baseline": 3.2,
    "Time_Saved_Hours_Weekly": 10,
    "Career_Confidence_Score": 8,

    # example encoded features (IMPORTANT)
    "Major_Biology": 0,
    "Major_Business Administration": 0,
    "Major_Data Science": 1,
    "Major_Fine Arts": 0,
    "Major_Modern History": 0,
    "Major_Software Engineering": 0,

    "Primary_AI_Tool_ChatGPT-4o": 1,
    "Primary_AI_Tool_Claude 3.5": 0,
    "Primary_AI_Tool_Gemini Pro": 0,
    "Primary_AI_Tool_GitHub Copilot": 0,
    "Primary_AI_Tool_Perplexity": 0,

    "Main_Usage_Case_Brainstorming": 0,
    "Main_Usage_Case_Code Debugging": 1,
    "Main_Usage_Case_Essay Drafting": 0,
    "Main_Usage_Case_Exam Prep": 0,
    "Main_Usage_Case_Literature Review": 0,

    "AI_Ethics_Concern_High": 0,
    "AI_Ethics_Concern_Low": 1,
    "AI_Ethics_Concern_Medium": 0
}
# convert into datafram
df_input=pd.DataFrame([input_data])

df_input=df_input.reindex(columns=features,fill_value=0)

#prediction
prediction=model.predict(df_input)
print(f"prediction of GPA", prediction[0])
