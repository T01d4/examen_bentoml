import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import bentoml

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the base directory
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")  # Relative path to processed data

# Load the processed datasets
X_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "X_train.csv"))
X_test = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "X_test.csv"))
y_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "y_train.csv"))
y_test = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "y_test.csv"))

# Initialize and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Save the model using BentoML
bentoml.sklearn.save_model("admission_model", model)
print("Model saved to BentoML Model Store.")