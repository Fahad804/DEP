import pandas as pd
import numpy as np
import re
import string
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
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
# df = pd.read_csv('C:/Users\Malik Danish Awan\Downloads\spam_or_not_spam.csv')
#
# # Display the first few rows of the dataset
# print(df.head())
#
# # Ensure all email texts are strings and fill missing values
# df['email'] = df['email'].astype(str).fillna('')
#
# # Data Cleaning and Preprocessing
# def clean_text(text):
#     # Convert to lowercase
#     text = text.lower()
#
#     # Remove email addresses, URLs, and special characters
#     text = re.sub(r'\S+@\S+', '', text)  # Remove email addresses
#     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
#     text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
#     text = re.sub(r'\d+', '', text)  # Remove digits
#
#     # Tokenization and removing stopwords
#     tokens = text.split()
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if word not in stop_words]
#
#     # Lemmatization (convert words to their root forms)
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(token) for token in tokens]
#
#     # Join the tokens back to string
#     return ' '.join(tokens)
#
#
# # Apply the clean_text function to the dataset
# df['cleaned_emails'] = df['email'].apply(clean_text)
#
# # Feature Extraction using TF-IDF
# vectorizer = TfidfVectorizer(max_features=5000)
# X = vectorizer.fit_transform(df['cleaned_emails']).toarray()
# y = df['label']
#
# # Split the data into training and test sets (80% train, 20% test)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Model: Naive Bayes Classifier
# nb_model = MultinomialNB()
# nb_model.fit(X_train, y_train)
#
# # Predictions
# nb_predictions = nb_model.predict(X_test)
#
# # Evaluation of the model
# print("Naive Bayes Classifier:")
# print("Accuracy:", accuracy_score(y_test, nb_predictions))
# print("Classification Report:\n", classification_report(y_test, nb_predictions))
# print("Confusion Matrix:\n", confusion_matrix(y_test, nb_predictions))
#
#
# # Function to classify new emails
# def classify_new_email(email_text):
#     # Preprocess the new email
#     cleaned_email = clean_text(email_text)
#
#     # Convert the email to TF-IDF features
#     email_tfidf = vectorizer.transform([cleaned_email])
#
#     # Naive Bayes prediction
#     nb_prediction = nb_model.predict(email_tfidf)[0]
#
#     if nb_prediction == 1:
#         return "Spam"
#     else:
#         return "Not Spam"
#
#
# # Test the function with new emails
# new_email_1 = "Congratulations! You've won a $1000 gift card. Click here to claim your prize."
# new_email_2 = "Dear user, your account has been updated. Please contact support if you did not authorize this change."
# new_email_3 = "Meeting tomorrow at 10 AM. Please confirm your attendance."
#
# print("\nClassification for new email 1:")
# print(classify_new_email(new_email_1))
#
# print("\nClassification for new email 2:")
# print(classify_new_email(new_email_2))
#
# print("\nClassification for new email 3:")
# print(classify_new_email(new_email_3))


#Task3
#Predicting Customer Churn
# Load the dataset
df = pd.read_csv('C:/Users\Malik Danish Awan\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv')

#Handle missing values in TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

#Convert categorical variables to numeric
categorical_cols = df.select_dtypes(include=['object']).columns

# Use LabelEncoder for binary categories
le = LabelEncoder()

# Apply LabelEncoder to binary categorical columns
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']

for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# Handle multi-class categorical columns with dummy variables
df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'], drop_first=True)

# Convert remaining categorical columns with multiple categories
# Replace 'No internet service' with 'No' for binary encoding
for col in ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
    df[col] = df[col].replace({'No internet service': 'No'})
    df[col] = le.fit_transform(df[col])

#Select features and target
X = df.drop(columns=['customerID', 'Churn'])
y = df['Churn']

#Check for any remaining non-numeric values or NaN in features
print(X.dtypes)  # Check data types
print(X.isnull().sum())  # Check for NaN values

#Convert any remaining non-numeric columns to numeric, if needed
X = X.apply(pd.to_numeric, errors='coerce')

#Check for NaN values again and fill them
X.fillna(X.mean(), inplace=True)

#Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

#Make predictions
y_pred = model.predict(X_test_scaled)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")

# New customer data (example input)
new_customer = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 12,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'Yes',
    'StreamingTV': 'No',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 70.35,
    'TotalCharges': 840.5
}


# Convert the new data into the format used for the training data
def preprocess_new_data(customer_data):
    # Convert categorical fields into binary or one-hot encoding as in training
    df_new = pd.DataFrame([customer_data])

    # Encode binary categorical features
    label_enc = LabelEncoder()
    df_new['gender'] = label_enc.fit_transform(df_new['gender'])
    df_new['Partner'] = label_enc.fit_transform(df_new['Partner'])
    df_new['Dependents'] = label_enc.fit_transform(df_new['Dependents'])
    df_new['PhoneService'] = label_enc.fit_transform(df_new['PhoneService'])
    df_new['PaperlessBilling'] = label_enc.fit_transform(df_new['PaperlessBilling'])

    # One-hot encode other categorical features
    df_new = pd.get_dummies(df_new,
                            columns=['InternetService', 'Contract', 'PaymentMethod', 'MultipleLines', 'OnlineSecurity',
                                     'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                     'StreamingMovies'])

    # Ensure the new data has the same columns as the training data
    missing_cols = set(X.columns) - set(df_new.columns)
    for col in missing_cols:
        df_new[col] = 0
    df_new = df_new[X.columns]

    # Scale the data
    df_new_scaled = scaler.transform(df_new)

    return df_new_scaled


# Preprocess new customer data
new_customer_processed = preprocess_new_data(new_customer)

# Predict churn
churn_prediction = model.predict(new_customer_processed)

# Output the result
if churn_prediction == 1:
    print("The customer is likely to churn.")
else:
    print("The customer is unlikely to churn.")