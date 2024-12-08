
# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import dump
import xgboost as xgb

# Step 1: Load the dataset
# Replace with your dataset path
dataset_path = 'C:\\Users\\Sahil\\Desktop\\Project\\custom_taxi_fare_dataset.csv'  # Update with your file location
df = pd.read_csv(dataset_path)

# Display dataset summary
print(f"Dataset Shape: {df.shape}")
print(df.head())

# Step 2: Data Preprocessing
# Filter relevant columns (adjust as per your dataset)
columns_to_use = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count", "fare_amount"]

# Drop missing values and filter invalid entries
df = df[columns_to_use]
df = df.dropna()
df = df[(df["fare_amount"] > 0) & (df["passenger_count"] > 0)]

# Features (X) and Target (y)
X = df[["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count"]]
y = df["fare_amount"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model using GPU
params = {
    "tree_method": "gpu_hist",  # Enables GPU acceleration
    "predictor": "gpu_predictor",
    "objective": "reg:squarederror",  # Regression task
    "eval_metric": "rmse",
    "learning_rate": 0.1,
    "max_depth": 6,
    "n_estimators": 100
}

print("Training model using GPU...")
model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train)

# Step 4: Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model trained successfully! Mean Squared Error: {mse:.2f}")

# Step 5: Save the Model
model_filename = "taxi_fare_model.pkl"
dump(model, model_filename)
print(f"Model saved as {model_filename}")
