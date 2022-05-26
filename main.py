import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# import data
customer = pd.read_csv("/Users/sean/Documents/project data/CUSTOMER.csv")
orderhistory = pd.read_csv("/Users/sean/Documents/project data/ORDER_HISTORY.csv")
product = pd.read_csv("/Users/sean/Documents/project data/PRODUCT.csv")

# exploratory Data Analysis
# # customer age - top 20
# plt.figure(figsize=(10,6))
# sns.countplot(data=customer, x="age", order=customer['age'].value_counts().iloc[:20].index)
# plt.xlabel('Age', fontsize=12)
# plt.ylabel('Total User', fontsize=12)
# plt.title('Number of customers from each age', fontsize=15)
# plt.show()

# # order history product - top 20
# plt.figure(figsize=(10,6))
# sns.countplot(data=orderhistory, x="barcode", order=orderhistory['barcode'].value_counts().iloc[:20].index)
# plt.xlabel('Product', fontsize=12)
# plt.ylabel('Total Purchase', fontsize=12)
# plt.title('Number of each product purchased', fontsize=15)
# plt.show()

# # order history country - top 20
# plt.figure(figsize=(10,6))
# sns.countplot(data=orderhistory, x="country", order=orderhistory['country'].value_counts().iloc[:5].index)
# plt.xlabel('Delivery Country', fontsize=12)
# plt.ylabel('Total Purchase', fontsize=12)
# plt.title('Number of products purchased from each country', fontsize=15)
# plt.show()


# encode the values from string to int
customer['region'] = LabelEncoder().fit_transform(customer['region'])
customer['age'] = LabelEncoder().fit_transform(customer['age'])

sc = StandardScaler()
X_train = sc.fit_transform(customer)

orderhistory['locale'] = LabelEncoder().fit_transform(orderhistory['locale'])
orderhistory['deliverysuburb'] = LabelEncoder().fit_transform(orderhistory['deliverysuburb'])
orderhistory['deliverycity'] = LabelEncoder().fit_transform(orderhistory['deliverycity'])
orderhistory['country'] = LabelEncoder().fit_transform(orderhistory['country'])


sc = StandardScaler()
X_train = sc.fit_transform(customer)

pca = PCA(n_components=3)
X_train = pca.fit_transform(X_train)
print(pca.explained_variance_ratio_)

# pca = PCA(n_components=3)
# fit = pca.fit(customer)
# print("Explained Variance: %s") % fit.explained_variance_ratio_
# print(fit.components_)

# print(customer.info())

# scaler = MinMaxScaler(feature_range=(0,10))
# print(scaler.fit_transform(customer))

# scaler = LabelEncoder()
# d = scaler.fit_transform(customer_filled)
# scaled_customer = pd.DataFrame(d, columns=names)
# print(scaled_customer.head())


# array = customer.values
# X = array[:,0:4]
#
# print(X)
# # # feature extraction
# pca = PCA(n_components=3)
# fit = pca.fit(X)
#
# print("Explained Variance: %s" % fit.explained_variance_ratio_)
# print(fit.components_)