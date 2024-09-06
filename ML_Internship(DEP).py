import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#Task 1
#Price Prediction
# Load the dataset
DATA_PATH = 'C:/Users\Malik Danish Awan\Downloads\Bengaluru_House_Data.csv'
df = pd.read_csv(DATA_PATH)

# Display the first few rows of the dataset
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Handle missing values for bath, balcony, and society
df['bath'] = df['bath'].fillna(df['bath'].median())
df['balcony'] = df['balcony'].fillna(df['balcony'].median())
df['society'] = df['society'].fillna('Unknown')
df['size'] = df['size'].fillna('Unknown')

# Convert size to numeric (extract the number of bedrooms)
df['BHK'] = df['size'].apply(lambda x: int(x.split(' ')[0]) if x != 'Unknown' else 0)

# Convert total_sqft to numeric
def convert_sqft_to_num(x):
    try:
        if '-' in x:
            vals = list(map(float, x.split('-')))
            return np.mean(vals)
        return float(x)
    except:
        return np.nan

df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)

# Fill missing values in total_sqft
df['total_sqft'] = df['total_sqft'].fillna(df['total_sqft'].median())

# Drop the original size column
df.drop(['size'], axis=1, inplace=True)

# Drop any remaining rows with missing values
df.dropna(inplace=True)

#Feature Engineering
# Convert categorical columns to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['area_type', 'availability', 'location', 'society'], drop_first=True)

print(df.head())

# Save the preprocessed data for reference
df.to_csv('preprocessed_house_prices.csv', index=False)

#Model Training
# Define features and target variable
X = df.drop('price', axis=1)
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")

#Model Evaluation
# Initialize and train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest - MAE: {mae_rf}")
print(f"Random Forest - MSE: {mse_rf}")
print(f"Random Forest - RMSE: {rmse_rf}")
print(f"Random Forest - R2 Score: {r2_rf}")

#Price Prediction
# Function to predict house prices
def predict_house_price(model, data):
    """
    Predict the price of a house given the input features.

    Parameters:
    model: Trained model
    data: DataFrame containing the input features

    Returns:
    Predicted house price
    """
    prediction = model.predict(data)
    return prediction


# Example new data (make sure it matches the input features)
new_data = pd.DataFrame({
    'total_sqft': [1200],
    'bath': [2],
    'balcony': [1],
    'BHK': [3],
    # Add other necessary one-hot encoded columns with default values
    # For example, if there are columns like 'area_type_Super built-up  Area', 'availability_Ready To Move', etc.
    # Include default 0 or 1 values for these columns
})

# Ensure the new_data DataFrame has the same columns as the training data
new_data = pd.get_dummies(new_data).reindex(columns=X.columns, fill_value=0)

# Predict the price
predicted_price = predict_house_price(model, new_data)
print(f"Predicted House Price: {predicted_price[0]}")