import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error
import joblib

df = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\AI project\AI_Impact_Student_Life_2026.csv")

# print(df.head())
# print(df.info())
df=df.drop("Student_ID" ,axis=1)
# print(df.info())
df=pd.get_dummies(df,columns=[
    "Major",
    "Primary_AI_Tool",
    "Main_Usage_Case",
    "AI_Ethics_Concern"
])
# print(df.describe())
# print(df.isnull().sum())
# print(df.duplicated().sum())
# print(df[df.duplicated()])
# z=stats.zscore(df["Age"])
# outlier=df[(z > 3) | (z < -3)]
# print(outlier)
print(df.info())
print(df.head(5))
# outlier detection of all columns at once
numeric_col= df.select_dtypes(include=["int64","float64"]).columns
for col in numeric_col:
    Q1=df[col].quantile(0.25)
    Q3=df[col].quantile(0.75)
    IQR=Q3-Q1
    lower=Q1-1.5*IQR
    upper=Q3+1.5*IQR
    outlier=df[(df[col] < lower) | (df[col] > upper)]
    df[col] = df[col].clip(lower, upper)
    print(f"\nColumn:{col}")
    print("outlier count:",len(outlier))


# plt.figure(figsize=(10,6))
# sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
# plt.show()
y=df["GPA_Post_AI"]
X=df.drop("GPA_Post_AI",axis=1)
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("R2_score",r2_score(y_test,y_pred))
print("MAE",mean_absolute_error(y_test,y_pred))

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print(importance.sort_values(by="Importance", ascending=False))

joblib.dump(model,"gpa_model.pkl")
joblib.dump(X.columns, "features.pkl")