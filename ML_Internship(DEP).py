import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#Task 1
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


#Task2
#Classifying Emails as Spam or Ham
# Load the dataset
df = pd.read_csv('C:/Users\Malik Danish Awan\Downloads\spam_or_not_spam.csv')

# Display the first few rows of the dataset
print(df.head())

# Ensure all email texts are strings and fill missing values
df['email'] = df['email'].astype(str).fillna('')

# Data Cleaning and Preprocessing
def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove email addresses, URLs, and special characters
    text = re.sub(r'\S+@\S+', '', text)  # Remove email addresses
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits

    # Tokenization and removing stopwords
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization (convert words to their root forms)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join the tokens back to string
    return ' '.join(tokens)


# Apply the clean_text function to the dataset
df['cleaned_emails'] = df['email'].apply(clean_text)

# Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_emails']).toarray()
y = df['label']

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model: Naive Bayes Classifier
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predictions
nb_predictions = nb_model.predict(X_test)

# Evaluation of the model
print("Naive Bayes Classifier:")
print("Accuracy:", accuracy_score(y_test, nb_predictions))
print("Classification Report:\n", classification_report(y_test, nb_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, nb_predictions))


# Function to classify new emails
def classify_new_email(email_text):
    # Preprocess the new email
    cleaned_email = clean_text(email_text)

    # Convert the email to TF-IDF features
    email_tfidf = vectorizer.transform([cleaned_email])

    # Naive Bayes prediction
    nb_prediction = nb_model.predict(email_tfidf)[0]

    if nb_prediction == 1:
        return "Spam"
    else:
        return "Not Spam"


# Test the function with new emails
new_email_1 = "Congratulations! You've won a $1000 gift card. Click here to claim your prize."
new_email_2 = "Dear user, your account has been updated. Please contact support if you did not authorize this change."
new_email_3 = "Meeting tomorrow at 10 AM. Please confirm your attendance."

print("\nClassification for new email 1:")
print(classify_new_email(new_email_1))

print("\nClassification for new email 2:")
print(classify_new_email(new_email_2))

print("\nClassification for new email 3:")
print(classify_new_email(new_email_3))