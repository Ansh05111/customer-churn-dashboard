import pandas as pd

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True) * 100)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

print(df["TotalCharges"].dtype)
print(df["TotalCharges"].isna().sum())