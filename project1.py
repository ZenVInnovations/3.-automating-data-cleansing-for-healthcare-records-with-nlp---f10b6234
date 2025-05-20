import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Load the data ---
print("Loading dataset...")
df = pd.read_csv("dataset.csv")
print(f"Initial shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# --- Step 2: Check for missing values and anomalies ---
print("\nChecking for missing values...")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

print("\nChecking unique values in GENDER and LUNG_CANCER columns (before cleaning):")
print(df['GENDER'].value_counts(dropna=False))
print(df['LUNG_CANCER'].value_counts(dropna=False))

print("\nChecking for invalid ages (outside 0–120):")
print(df[(df['AGE'] <= 0) | (df['AGE'] >= 120)])

# --- Step 3: Define NLP-based cleaning functions ---
def normalize_gender(gender):
    gender = str(gender).strip().lower()
    if gender in ['m', 'male']:
        return 'Male'
    elif gender in ['f', 'female']:
        return 'Female'
    return 'Other'

def normalize_cancer_status(status):
    status = str(status).strip().lower()
    if 'yes' in status:
        return 'Yes'
    elif 'no' in status:
        return 'No'
    return 'Unknown'

def clean_healthcare_data(df):
    df['GENDER'] = df['GENDER'].apply(normalize_gender)
    df['LUNG_CANCER'] = df['LUNG_CANCER'].apply(normalize_cancer_status)
    df = df.drop_duplicates()
    df = df[(df['AGE'] > 0) & (df['AGE'] < 120)]
    return df

# --- Step 4: Clean the dataset ---
print("\nCleaning dataset using NLP-based functions...")
cleaned_df = clean_healthcare_data(df)

# --- Step 5: Show results after cleaning ---
print("\nAfter cleaning:")
print(f"New shape: {cleaned_df.shape}")
print("\nUnique values in GENDER:")
print(cleaned_df['GENDER'].value_counts())
print("\nUnique values in LUNG_CANCER:")
print(cleaned_df['LUNG_CANCER'].value_counts())
print("\nChecking for invalid ages again:")
print(cleaned_df[(cleaned_df['AGE'] <= 0) | (cleaned_df['AGE'] >= 120)])

# --- Step 6: Save cleaned dataset ---
cleaned_df.to_csv("cleaned_dataset.csv", index=False)
print("\nCleaned dataset saved as cleaned_dataset.csv")

# --- Step 7: Plot graphs ---
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 6))

# Gender Distribution
plt.subplot(1, 2, 1)
sns.countplot(data=cleaned_df, x='GENDER', palette='Set2')
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")

# Lung Cancer Distribution
plt.subplot(1, 2, 2)
sns.countplot(data=cleaned_df, x='LUNG_CANCER', palette='Set1')
plt.title("Lung Cancer Status Distribution")
plt.xlabel("Lung Cancer")
plt.ylabel("Count")

plt.tight_layout()
plt.show()
