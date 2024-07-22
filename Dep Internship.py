import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import BertTokenizer, BertForSequenceClassification
import torch



#Task 1
#Transaction Data
#Load Data
# df=pd.read_csv("C:/Users\Malik Danish Awan\Downloads\Daily Household Transactions.csv")
# print(df.head(50))
#
# #Covert Date column to Datetime
# df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
#
# # Check for any rows where the date conversion failed
# invalid_dates = df[df['Date'].isna()]
#
# if not invalid_dates.empty:
#     print(f"Found {len(invalid_dates)} rows with invalid date formats.")
#     print(invalid_dates)
#     # Handle invalid dates if necessary (e.g., drop these rows or fix the format)
#     df = df.dropna(subset=['Date'])
# else:
#     print("All dates successfully converted.")
#
# #Handle Missing Value
# df.dropna(inplace=True)
#
# # Remove duplicates
# df.drop_duplicates(inplace=True)
#
# # Filter for expenses only if necessary
# df = df[df['Income/Expense'] == 'Expense']
#
# # Summary statistics
# print(df.describe())
#
# # Distribution of amount
# sns.histplot(df['Amount'], kde=True)
# plt.show()
#
# # Transactions over time
# df.set_index('Date')['Amount'].resample('ME').sum().plot()
# plt.show()
#
# # Distribution across categories
# sns.countplot(data=df, x='Category')
# plt.show()
#
# #Feature Engineering
# # Aggregate data
# agg_data = df.groupby(['Category', 'Subcategory']).agg({
#     'Amount': ['sum', 'mean', 'count']
# }).reset_index()
#
# # Rename columns for easier access
# agg_data.columns = ['category', 'subcategory', 'total_amount', 'avg_amount', 'transaction_count']
# print(agg_data.head())
#
# # Scale the data
# scaler = StandardScaler()
# agg_data_scaled = scaler.fit_transform(agg_data[['total_amount', 'avg_amount', 'transaction_count']])
#
# # Determine the number of clusters
# inertia = []
# for n in range(1, 11):
#     kmeans = KMeans(n_clusters=n)
#     kmeans.fit(agg_data_scaled)
#     inertia.append(kmeans.inertia_)
#
# import matplotlib.pyplot as plt
#
# plt.plot(range(1, 11), inertia)
# plt.xlabel('Number of Clusters')
# plt.ylabel('Inertia')
# plt.show()
#
# # Fit K-means
# kmeans = KMeans(n_clusters=3)  # Example: 3 clusters
# agg_data['cluster'] = kmeans.fit_predict(agg_data_scaled)
# print(agg_data.head())
#
# # PCA for visualization
# pca = PCA(n_components=2)
# agg_data_pca = pca.fit_transform(agg_data_scaled)
#
# plt.scatter(agg_data_pca[:, 0], agg_data_pca[:, 1], c=agg_data['cluster'])
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.title('Customer Segments')
# plt.show()


#Task 2
#Price Prediction
# Load the dataset
# DATA_PATH = 'C:/Users\Malik Danish Awan\Downloads\Bengaluru_House_Data.csv'
# df = pd.read_csv(DATA_PATH)
#
# # Display the first few rows of the dataset
# print(df.head())
#
# # Check for missing values
# print(df.isnull().sum())
#
# # Handle missing values for bath, balcony, and society
# df['bath'] = df['bath'].fillna(df['bath'].median())
# df['balcony'] = df['balcony'].fillna(df['balcony'].median())
# df['society'] = df['society'].fillna('Unknown')
# df['size'] = df['size'].fillna('Unknown')
#
# # Convert size to numeric (extract the number of bedrooms)
# df['BHK'] = df['size'].apply(lambda x: int(x.split(' ')[0]) if x != 'Unknown' else 0)
#
# # Convert total_sqft to numeric
# def convert_sqft_to_num(x):
#     try:
#         if '-' in x:
#             vals = list(map(float, x.split('-')))
#             return np.mean(vals)
#         return float(x)
#     except:
#         return np.nan
#
# df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
#
# # Fill missing values in total_sqft
# df['total_sqft'] = df['total_sqft'].fillna(df['total_sqft'].median())
#
# # Drop the original size column
# df.drop(['size'], axis=1, inplace=True)
#
# # Drop any remaining rows with missing values
# df.dropna(inplace=True)
#
# #Feature Engineering
# # Convert categorical columns to numerical using one-hot encoding
# df = pd.get_dummies(df, columns=['area_type', 'availability', 'location', 'society'], drop_first=True)
#
# print(df.head())
#
# # Save the preprocessed data for reference
# df.to_csv('preprocessed_house_prices.csv', index=False)
#
# #Model Training
# # Define features and target variable
# X = df.drop('price', axis=1)
# y = df['price']
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Initialize and train the model
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# # Make predictions
# y_pred = model.predict(X_test)
#
# # Evaluate the model
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)
#
# print(f"MAE: {mae}")
# print(f"MSE: {mse}")
# print(f"RMSE: {rmse}")
# print(f"R2 Score: {r2}")
#
# #Model Evaluation
# # Initialize and train the model
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)
#
# # Make predictions
# y_pred_rf = rf_model.predict(X_test)
#
# # Evaluate the model
# mae_rf = mean_absolute_error(y_test, y_pred_rf)
# mse_rf = mean_squared_error(y_test, y_pred_rf)
# rmse_rf = np.sqrt(mse_rf)
# r2_rf = r2_score(y_test, y_pred_rf)
#
# print(f"Random Forest - MAE: {mae_rf}")
# print(f"Random Forest - MSE: {mse_rf}")
# print(f"Random Forest - RMSE: {rmse_rf}")
# print(f"Random Forest - R2 Score: {r2_rf}")
#
# #Price Prediction
# # Function to predict house prices
# def predict_house_price(model, data):
#     """
#     Predict the price of a house given the input features.
#
#     Parameters:
#     model: Trained model
#     data: DataFrame containing the input features
#
#     Returns:
#     Predicted house price
#     """
#     prediction = model.predict(data)
#     return prediction
#
#
# # Example new data (make sure it matches the input features)
# new_data = pd.DataFrame({
#     'total_sqft': [1200],
#     'bath': [2],
#     'balcony': [1],
#     'BHK': [3],
#     # Add other necessary one-hot encoded columns with default values
#     # For example, if there are columns like 'area_type_Super built-up  Area', 'availability_Ready To Move', etc.
#     # Include default 0 or 1 values for these columns
# })
#
# # Ensure the new_data DataFrame has the same columns as the training data
# new_data = pd.get_dummies(new_data).reindex(columns=X.columns, fill_value=0)
#
# # Predict the price
# predicted_price = predict_house_price(model, new_data)
# print(f"Predicted House Price: {predicted_price[0]}")


#Task 3
#Sentiment Analysis
# Load the dataset
DATA_PATH = 'Twitter_Data.csv'
df = pd.read_csv(DATA_PATH)

# Ensure 'clean_text' is all strings and fill missing values
df['clean_text'] = df['clean_text'].astype(str).fillna('')

# Text Processing Function
def text_processing(text):
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'#[\w-]+', '', text)
    text = re.sub(r'\d{2}[-/]\d{2}[-/]\d{4}', '', text)
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

df['processed_text'] = df['clean_text'].apply(text_processing)

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to get sentiment
def get_sentiment(text):
    sentiment_score = sia.polarity_scores(text)
    compound_score = sentiment_score['compound']
    if compound_score > 0.05:
        return "Positive"
    elif compound_score < -0.05:
        return "Negative"
    else:
        return "Neutral"

df['sentiment'] = df['processed_text'].apply(get_sentiment)

# Count the sentiments
sentiment_counts = df['sentiment'].value_counts()

# Plot the sentiment distribution
plt.figure(figsize=(8, 6))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'blue'])
plt.xlabel('Sentiment')
plt.ylabel('Counts')
plt.title('Sentiment Distribution')
plt.show()

# Example usage
sample_text = "Life is a beautiful journey, full of ups and downs, but with a positive mindset, you can turn every obstacle into a stepping stone for success."
processed_text = text_processing(sample_text)
sentiment = get_sentiment(processed_text)
print(f"Sentiment: {sentiment}")


















