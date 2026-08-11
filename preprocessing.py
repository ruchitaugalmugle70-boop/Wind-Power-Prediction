import pandas as pd

# Load dataset
df = pd.read_csv("T1.csv")

# Convert Date/Time into datetime
df["Date/Time"] = pd.to_datetime(df["Date/Time"], format="%d %m %Y %H:%M")

# Extract useful time features
df["Hour"] = df["Date/Time"].dt.hour
df["Day"] = df["Date/Time"].dt.day
df["Month"] = df["Date/Time"].dt.month
df["DayOfWeek"] = df["Date/Time"].dt.dayofweek

# Remove original Date/Time column
df = df.drop("Date/Time", axis=1)

print("Dataset after preprocessing:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nDataset Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())