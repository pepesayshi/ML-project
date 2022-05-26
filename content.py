import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re

product = pd.read_csv("/Users/sean/Documents/project data/PRODUCT.csv")

# combine all the useful context into a new column called combined
product["combined"] = product[['description', 'extdescription', 'department', 'prodgroup', 'category', 'colour']].agg(lambda x: ','.join(x.values.astype(str)), axis=1)

# product department - top 10
# plt.figure(figsize=(10,6))
# sns.countplot(data=product, x="department", order=product['department'].value_counts().iloc[:10].index)
# plt.xlabel('Department', fontsize=12)
# plt.ylabel('Total Products', fontsize=12)
# plt.title('Number of products from each department', fontsize=15)
# plt.show()

# Word Count Distribution for the combined bag of words
# product['word_count'] = product['combined'].apply(lambda x: len(str(x).split())) # Plotting the word count
# product['word_count'].plot(
#     kind='hist',
#     bins = 50,
#     figsize = (12,8),title='Word Count Distribution for product descriptions')
# plt.show()

# # Bigram distribution for the product description
# #Converting text descriptions into vectors using TF-IDF using Bigram
# tf = TfidfVectorizer(ngram_range=(2, 2), stop_words='english', lowercase = False)
# tfidf_matrix = tf.fit_transform(product['combined'].values.astype('U'))
# total_words = tfidf_matrix.sum(axis=0)
# #Finding the word frequency
# freq = [(word, total_words[0, idx]) for word, idx in tf.vocabulary_.items()]
# freq =sorted(freq, key = lambda x: x[1], reverse=True)
# #converting into dataframe
# bigram = pd.DataFrame(freq)
# bigram.rename(columns = {0:'bigram', 1: 'count'}, inplace = True)
# #Taking first 20 records
# bigram = bigram.head(20)
#
# #Plotting the bigram distribution
# bigram.plot(x ='bigram', y='count', kind = 'bar', title = "Bigram disribution for the top 20 words in the product description", figsize = (13,10), )
# plt.show()

# Function for removing NonAscii characters
def _removeNonAscii(s):
    return "".join(i for i in s if  ord(i)<128)

# Function for converting into lower case
def make_lower_case(text):
    return text.lower()

# Function for removing stop words
def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text

# Function for removing punctuation
def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text

# Function for removing the html tags
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

# Applying all the functions in description and storing as a cleaned_desc
product['cleaned_combined'] = product['combined'].apply(_removeNonAscii)
product['cleaned_combined'] = product.cleaned_desc.apply(func = make_lower_case)
product['cleaned_combined'] = product.cleaned_desc.apply(func = remove_stop_words)
product['cleaned_combined'] = product.cleaned_desc.apply(func=remove_punctuation)
product['cleaned_combined'] = product.cleaned_desc.apply(func=remove_html)

# We are going to build two recommendation engines using the book titles and descriptions.
#
# Convert each book title and description into vectors using TF-IDF and bigram. See here for more details on TF-IDF
# We are building two recommendation engines, one with a book title and another one with a book description. The model recommends a similar book based on title and description.
# Calculate the similarity between all the books using cosine similarity.
# Define a function that takes the book title and genre as input and returns the top five similar recommended books based on the title and description.