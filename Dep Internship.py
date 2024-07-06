import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


#Transaction Data
#Load Data
df=pd.read_csv("C:/Users\Malik Danish Awan\Downloads\Daily Household Transactions.csv")
print(df.head(50))

#Covert Date column to Datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')

# Check for any rows where the date conversion failed
invalid_dates = df[df['Date'].isna()]

if not invalid_dates.empty:
    print(f"Found {len(invalid_dates)} rows with invalid date formats.")
    print(invalid_dates)
    # Handle invalid dates if necessary (e.g., drop these rows or fix the format)
    df = df.dropna(subset=['Date'])
else:
    print("All dates successfully converted.")

#Handle Missing Value
df.dropna(inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Filter for expenses only if necessary
df = df[df['Income/Expense'] == 'Expense']

# Summary statistics
print(df.describe())

# Distribution of amount
sns.histplot(df['Amount'], kde=True)
plt.show()

# Transactions over time
df.set_index('Date')['Amount'].resample('ME').sum().plot()
plt.show()

# Distribution across categories
sns.countplot(data=df, x='Category')
plt.show()

#Feature Engineering
# Aggregate data
agg_data = df.groupby(['Category', 'Subcategory']).agg({
    'Amount': ['sum', 'mean', 'count']
}).reset_index()

# Rename columns for easier access
agg_data.columns = ['category', 'subcategory', 'total_amount', 'avg_amount', 'transaction_count']
print(agg_data.head())

# Scale the data
scaler = StandardScaler()
agg_data_scaled = scaler.fit_transform(agg_data[['total_amount', 'avg_amount', 'transaction_count']])

# Determine the number of clusters
inertia = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(agg_data_scaled)
    inertia.append(kmeans.inertia_)

import matplotlib.pyplot as plt

plt.plot(range(1, 11), inertia)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Fit K-means
kmeans = KMeans(n_clusters=3)  # Example: 3 clusters
agg_data['cluster'] = kmeans.fit_predict(agg_data_scaled)
print(agg_data.head())

# PCA for visualization
pca = PCA(n_components=2)
agg_data_pca = pca.fit_transform(agg_data_scaled)

plt.scatter(agg_data_pca[:, 0], agg_data_pca[:, 1], c=agg_data['cluster'])
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Customer Segments')
plt.show()



