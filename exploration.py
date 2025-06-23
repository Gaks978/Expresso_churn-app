import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("Expresso_churn_dataset.csv")
print("Head of the dataset:\n", df.head())
print("\nDataset Info:\n")
print(df.info())
print("\nStatistical Summary:\n")
print(df.describe())

# Generate profile report
profile = ProfileReport(df, title="Expresso Churn Data Report", explorative=True)
profile.to_file("expresso_churn_report.html")

# Check for missing values
print("\nMissing Values Per Column:\n")
print(df.isnull().sum())

# Fill numerical columns with mean
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Fill categorical columns with mode
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Handle outliers using IQR method
Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1

# Filter out outliers
df = df[~((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

print("\nShape after outlier removal:", df.shape)

# Encode categorical features using LabelEncoder
label_encoder = LabelEncoder()
cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col])

print("\nShape after label encoding categorical variables:", df.shape)


# Save cleaned dataset
df.to_csv("cleaned_expresso_data.csv", index=False)
print("\nCleaned data saved to 'cleaned_expresso_data.csv'")
