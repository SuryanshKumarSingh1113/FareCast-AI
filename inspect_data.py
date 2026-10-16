
import pandas as pd

# Step 1: Load the original dataset from the data folder
df = pd.read_csv("data/ncr_ride_bookings.csv")

# Display the original dataset size (rows, columns)
print("Original dataset shape:", df.shape)


# Step 2: Check how many bookings belong to each status
# This helps us identify completed, cancelled, and incomplete rides
print("\nBooking Status Distribution:")
print(df["Booking Status"].value_counts(dropna=False))


# Step 3: Select columns that are important for fare prediction
# These columns may be useful as input features or the target variable
important_cols = [
    "Booking Status",
    "Vehicle Type",
    "Ride Distance",
    "Booking Value",
    "Avg VTAT",
    "Avg CTAT"
]

# Count missing values in the selected columns
print("\nMissing Values in Important Columns:")
print(df[important_cols].isnull().sum())


# Step 4: Keep only completed rides
# Cancelled and incomplete bookings are not suitable for this initial model
completed = df[
    df["Booking Status"].astype(str).str.strip().str.lower() == "completed"
].copy()

# Display the number of completed rides
print("\nNumber of Completed Rides:", len(completed))


# Step 5: Remove rows missing essential prediction information
# We need a known fare, ride distance, and vehicle type
valid = completed.dropna(
    subset=["Booking Value", "Ride Distance", "Vehicle Type"]
).copy()


# Step 6: Remove invalid fare and distance values
# Fare and distance must both be greater than zero
valid = valid[
    (valid["Booking Value"] > 0) &
    (valid["Ride Distance"] > 0)
].copy()

# Display the number of usable records remaining
print("Valid Rides for Initial Modeling:", len(valid))


# Step 7: Examine the fare distribution
# Summary statistics help us understand typical fares and unusual values
print("\nFare Summary:")
print(valid["Booking Value"].describe())


# Step 8: Examine the ride-distance distribution
# This helps identify unrealistic or extreme trip distances
print("\nRide Distance Summary:")
print(valid["Ride Distance"].describe())


# Step 9: Select relevant features for fare prediction
# These features must be available when a user requests a prediction
features = [
    "Vehicle Type",
    "Ride Distance"
]

# Define the target column that the model will predict
target = "Booking Value"

# Create a separate dataset containing only selected features and target
# The original dataset remains unchanged
model_df = valid[features + [target]].copy()

# Remove any remaining rows with missing values in selected columns
model_df = model_df.dropna(
    subset=features + [target]
).copy()

# Display the final dataset shape
print("\nFinal Modeling Dataset Shape:", model_df.shape)

# Display a preview of the selected data
print("\nSelected Features and Target:")
print(model_df.head())

# Save the cleaned modeling dataset separately
model_df.to_csv("data/cleaned_fare_data.csv", index=False)

print("\nCleaned dataset saved to data/cleaned_fare_data.csv")


# Step 10: Analyse the relationship between distance and fare
# This helps us understand whether longer rides generally cost more

print("\nAverage Fare by Vehicle Type:")
print(
    valid.groupby("Vehicle Type")["Booking Value"]
    .agg(["count", "mean", "median"])
    .sort_values("count", ascending=False)
)

# Step 11: Check the correlation between ride distance and fare
# Correlation ranges from -1 to +1
# Values near +1 indicate a strong positive linear relationship

correlation = valid["Ride Distance"].corr(valid["Booking Value"])

print("\nDistance-Fare Correlation:", round(correlation, 4))


# Step 12: Check unusual fare and distance values
# These summaries help identify extreme values for further investigation

print("\nRide Distance Quantiles:")
print(valid["Ride Distance"].quantile([0, 0.25, 0.50, 0.75, 0.95, 0.99, 1]))

print("\nBooking Value Quantiles:")
print(valid["Booking Value"].quantile([0, 0.25, 0.50, 0.75, 0.95, 0.99, 1]))


# Step 13: Check whether fares are repeated or nearly fixed
# This helps us understand how much useful variation exists in the target

print("\nMost Common Booking Values:")
print(valid["Booking Value"].value_counts().head(10))

# Step 14: Check fare variation for each vehicle type
# Standard deviation shows how much fares vary within each category

print("\nFare Variation by Vehicle Type:")
print(
    valid.groupby("Vehicle Type")["Booking Value"]
    .agg(["count", "mean", "std", "min", "max"])
    .round(2)
)

# Step 15: Check whether distance has a nonlinear relationship with fare
# Spearman correlation checks whether the relationship is monotonic,
# even when it is not linear

spearman_corr = valid["Ride Distance"].corr(
    valid["Booking Value"],
    method="spearman"
)

print("\nDistance-Fare Spearman Correlation:", round(spearman_corr, 4))



# Step 16: Check whether fare changes across distance groups
# Group rides into distance ranges and compare their average fares

valid["Distance Group"] = pd.cut(
    valid["Ride Distance"],
    bins=[0, 5, 10, 20, 30, 40, 50],
    include_lowest=True
)

print("\nAverage Fare by Distance Group:")
print(
    valid.groupby("Distance Group", observed=False)["Booking Value"]
    .agg(["count", "mean", "median"])
    .round(2)
)

# Step 17: Check whether booking status is related to missing fares
# This helps us understand why many booking values are missing

print("\nBooking Status vs Missing Booking Value:")
print(
    df.assign(Fare_Missing=df["Booking Value"].isnull())
    .groupby("Booking Status")["Fare_Missing"]
    .agg(["count", "sum", "mean"])
)



# Step 18: Check location diversity
# Location differences may help explain fare variation

print("\nUnique Pickup Locations:")
print(valid["Pickup Location"].nunique())

print("\nUnique Drop Locations:")
print(valid["Drop Location"].nunique())

# Display the most common pickup locations
print("\nTop 10 Pickup Locations:")
print(valid["Pickup Location"].value_counts().head(10))

# Display the most common drop locations
print("\nTop 10 Drop Locations:")
print(valid["Drop Location"].value_counts().head(10))



# Step 19: Check whether waiting-time features have enough data
# These features may help only if they are available before fare prediction

waiting_cols = ["Avg VTAT", "Avg CTAT"]

print("\nWaiting-Time Feature Availability:")

for col in waiting_cols:
    # Count non-missing values among completed rides
    available = valid[col].notna().sum()

    print(f"{col}: {available} available out of {len(valid)} completed rides")
