import os
import pandas as pd
from sklearn.model_selection import train_test_split
import requests

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the base directory
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "admission.csv")  # Relative path to raw data
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")  # Relative path to processed data
DATA_URL = "https://assets-datascientest.s3.eu-west-1.amazonaws.com/MLOPS/bentoml/admission.csv"

# Ensure the raw data folder exists
os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)

# Download the dataset if it doesn't exist
if not os.path.exists(RAW_DATA_PATH):
    print(f"Downloading dataset from {DATA_URL}...")
    response = requests.get(DATA_URL)
    if response.status_code == 200:
        with open(RAW_DATA_PATH, "wb") as file:
            file.write(response.content)
        print(f"Dataset downloaded successfully and saved to '{RAW_DATA_PATH}'.")
    else:
        raise Exception(f"Failed to download the dataset. HTTP Status Code: {response.status_code}")

# Load the dataset
data = pd.read_csv(RAW_DATA_PATH)

# Drop unnecessary columns (adjust based on your dataset)
data = data.drop(columns=["Serial No."], errors='ignore')

# Separate features (X) and target (y)
X = data.drop(columns=['Chance of Admit '])  # Replace with your target variable
y = data['Chance of Admit ']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure the processed data folder exists
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# Save the datasets
X_train.to_csv(os.path.join(PROCESSED_DATA_PATH, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(PROCESSED_DATA_PATH, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(PROCESSED_DATA_PATH, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(PROCESSED_DATA_PATH, "y_test.csv"), index=False)

print("Data preparation complete. Processed files saved in 'data/processed'.")