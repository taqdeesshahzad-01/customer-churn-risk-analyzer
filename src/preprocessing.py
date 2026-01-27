import pandas as pd

# Load raw data
df = pd.read_csv("data/raw/churn.csv")

# Inspect data (important for learning)
print(df.head())
print(df.info())

# Standardize column names
df.columns = df.columns.str.lower().str.replace(" ", "_")

# Drop customerID (not useful for ML)
if "customerid" in df.columns:
    df = df.drop(columns=["customerid"])

# Handle missing values
df = df.dropna()

# Encode target variable
df["churn"] = df["churn"].map({"Yes": 1, "No": 0})

# Save cleaned data
df.to_csv("data/processed/cleaned_churn.csv", index=False)

print("✅ Data preprocessing completed successfully!")
