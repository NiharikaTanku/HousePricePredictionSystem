import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1️⃣ Load dataset
df = pd.read_csv('house_price_prediction_dataset (1).csv')

# 2️⃣ One-hot encoding
df = pd.get_dummies(df, columns=['furnishing'], drop_first=True)
df = pd.get_dummies(df, columns=['location'], drop_first=True)

# 3️⃣ Separate features and target
X = df.drop('price', axis=1)
y = df['price']

# 4️⃣ Split data (before scaling)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5️⃣ Scale only numerical columns
numerical_cols = ['area_sqft', 'bedrooms', 'bathrooms', 'age_years']

scaler = StandardScaler()

# Fit only on training data
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])

# Transform test data using same scaler
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# 6️⃣ Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Take user input
area = int(input("Enter area in sqft: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
ageyears = int(input("Enter age of the house in years: "))

furnishing = input("Enter furnishing (Semi-Furnished/Unfurnished/Other): ")
location = input("Enter location (Downtown/Suburb/Other): ")

# Create input dictionary with all features
input_data = {
    'area_sqft': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age_years': ageyears,
    'furnishing_Semi-Furnished': 1 if furnishing == "Semi-Furnished" else 0,
    'furnishing_Unfurnished': 1 if furnishing == "Unfurnished" else 0,
    'location_Downtown': 1 if location == "Downtown" else 0,
    'location_Suburb': 1 if location == "Suburb" else 0
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Scale numerical columns
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# Predict
predicted_price = model.predict(input_df)

print("Predicted House Price:", predicted_price[0])
# 7️⃣ Predict
y_pred = model.predict(X_test)

# 8️⃣ Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

