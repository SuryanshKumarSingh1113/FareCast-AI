
# Import libraries for data processing and machine learning
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Step 1: Load the original dataset
df = pd.read_csv("data/ncr_ride_bookings.csv")

# Step 2: Keep only completed rides
# Cancelled and incomplete bookings do not have valid fare targets
df = df[
    df["Booking Status"].astype(str).str.strip().str.lower()
    == "completed"
].copy()

# Step 3: Remove rows missing essential values
# Fare, distance, vehicle type and locations are required
required_columns = [
    "Booking Value",
    "Ride Distance",
    "Vehicle Type",
    "Pickup Location",
    "Drop Location",
    "Date",
    "Time"
]

df = df.dropna(subset=required_columns).copy()

# Step 4: Remove non-positive fare and distance values
df = df[
    (df["Booking Value"] > 0) &
    (df["Ride Distance"] > 0)
].copy()

# Step 5: Convert date and time into useful numeric features
# The model can learn patterns by hour, weekday and month
df["Trip DateTime"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"].astype(str),
    errors="coerce"
)

# Remove rows where date or time could not be parsed
df = df.dropna(subset=["Trip DateTime"]).copy()

df["Hour"] = df["Trip DateTime"].dt.hour
df["Day of Week"] = df["Trip DateTime"].dt.dayofweek
df["Month"] = df["Trip DateTime"].dt.month

print("Usable records:", len(df))


# Step 6: Define model inputs and prediction target
# Only use information available when making a prediction
# Final prediction features available before a ride
features = [
    "Vehicle Type",
    "Ride Distance",
    "Pickup Location",
    "Drop Location",
    "Hour",
    "Day of Week",
    "Month"
]

X = df[features]
y = df["Booking Value"]


# Step 7: Separate categorical and numerical features
# Categorical features require encoding
categorical_features = [
    "Vehicle Type",
    "Pickup Location",
    "Drop Location"
]

# Numerical features require missing-value handling
numerical_features = [
    "Ride Distance",
    "Hour",
    "Day of Week",
    "Month"
]    



# Step 8: Build preprocessing pipelines
# Fill missing categorical values and one-hot encode categories
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Fill missing numerical values with their median
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Apply the appropriate transformation to each feature type
preprocessor = ColumnTransformer(transformers=[
    ("categorical", categorical_transformer, categorical_features),
    ("numerical", numerical_transformer, numerical_features)
])


# Step 9: Split data into training and test sets
# The test set is kept separate for final evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Baseline: predict the median training fare for every test ride
# This provides a simple benchmark for evaluating our ML models

baseline_prediction = np.full(
    shape=len(y_test),
    fill_value=y_train.median()
)

# Evaluate the baseline using the same test set
baseline_mae = mean_absolute_error(y_test, baseline_prediction)
baseline_rmse = np.sqrt(
    mean_squared_error(y_test, baseline_prediction)
)
baseline_r2 = r2_score(y_test, baseline_prediction)

print("\nMedian Baseline")
print(f"MAE: {baseline_mae:.2f}")
print(f"RMSE: {baseline_rmse:.2f}")
print(f"R2 Score: {baseline_r2:.4f}")

# Step 10: Define candidate regression models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
}


# Step 11: Train, evaluate and compare the models
best_model = None
best_rmse = float("inf")

for model_name, regressor in models.items():

    # Combine preprocessing and the regressor into one pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", regressor)
    ])

    # Learn patterns from the training set
    pipeline.fit(X_train, y_train)

    # Predict fares for unseen test records
    predictions = pipeline.predict(X_test)

    # Calculate standard regression evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"\n{model_name}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

    # Keep the model with the lowest test RMSE
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = pipeline


# Step 12: Save the selected model and preprocessing pipeline
# Joblib saves both together for consistent predictions later
joblib.dump(best_model, "farecast_model.joblib")

print("\nSelected model saved as farecast_model.joblib")
