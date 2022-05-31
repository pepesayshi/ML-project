import numpy as np
import pandas as pd
import seaborn as sns
from pasta.augment import inline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
import string
import random
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt


product = pd.read_csv("/Users/sean/Documents/project data/PRODUCT_SMALLER.csv")

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
# Converting text descriptions into vectors using TF-IDF using Bigram
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
product['cleaned_desc'] = product['combined'].apply(_removeNonAscii)
product['cleaned_desc'] = product['combined'].apply(func=make_lower_case)
product['cleaned_desc'] = product['combined'].apply(func=remove_stop_words)
product['cleaned_desc'] = product['combined'].apply(func=remove_punctuation)
product['cleaned_desc'] = product['combined'].apply(func=remove_html)

# We are going to build two recommendation engines using the book titles and descriptions.
#
# Convert each book title and description into vectors using TF-IDF and bigram. See here for more details on TF-IDF
# We are building two recommendation engines, one with a book title and another one with a book description. The model recommends a similar book based on title and description.
# Calculate the similarity between all the books using cosine similarity.
# Define a function that takes the book title and genre as input and returns the top five similar recommended books based on the title and description.

# Function for recommending books based on Book title. It takes book title and genre as an input.def recommend(title, genre):

# def recommend(title, colour):
#     # Converting the book description into vectors and used bigram
#     tf = TfidfVectorizer(analyzer='word', ngram_range=(2, 2), min_df=1, stop_words='english')
#     tfidf_matrix = tf.fit_transform(product['cleaned_desc'])
#
#     # Calculating the similarity measures based on Cosine Similarity
#     sg = cosine_similarity(tfidf_matrix, tfidf_matrix)
#
#     movie_index = product[product['description'] == title].index.values[0]
#     similar_movies = list(enumerate(sg[movie_index]))
#     sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
#
#     def get_title_from_index(index):
#         return product[product.index == index]["description"].values[0]
#
#     i = 0
#     for movies in sorted_similar_movies:
#         print(get_title_from_index(movies[0]))
#         response = requests.get(product[product.index == movies[0]]["url"].values[0])
#         img = Image.open(BytesIO(response.content))
#         plt.figure()
#         print(plt.imshow(img))
#         plt.show()
#         i = i + 1;
#         if i > 15:
#             break
#
# recommend("Oversized Button Front Blazer", "BLACK")

def recommend(title, colour = None):
    global rec
    # Matching the colour with the dataset and reset the index
    data = product.loc[product['colour'] == colour]
    data.reset_index(level=0, inplace=True)

    # Convert the index into series
    indices = pd.Series(data.index, index=data['description'])

    # Converting the book description into vectors and used bigram
    tf = TfidfVectorizer(analyzer='word', ngram_range=(2, 2), min_df=1, stop_words='english')
    tfidf_matrix = tf.fit_transform(data['cleaned_desc'])

    # Calculating the similarity measures based on Cosine Similarity
    sg = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get the index corresponding to original_title

    idx = indices[title]  # Get the pairwsie similarity scores
    sig = list(enumerate(sg[idx]))  # Sort the books
    sig = sorted(sig, key=lambda x: x[1], reverse=True)  # Scores of the 5 most similar books
    sig = sig[1:6]  # Book indicies
    movie_indices = [i[0] for i in sig]

    # Top 5 book recommendation
    rec = data[['description', 'url']].iloc[movie_indices]

    print('Selected Product is: ' + title)

    count = 1;
    for i in rec['description']:
        print('Recommended Top 5 products are: Number ' + str(count) + ' - ' + i)
        count = count + 1;

    image = product.loc[product['description'] == title]['url'].iloc[0]
    response = requests.get(image)
    img = Image.open(BytesIO(response.content))
    plt.figure()
    print(plt.imshow(img))
    plt.show()

    for i in rec['url']:
        response = requests.get(i)
        img = Image.open(BytesIO(response.content))
        plt.figure()
        print(plt.imshow(img))
        plt.show()

# recommend("Twist Reversible Knit Dress", "BLACK")
# recommend("Oversized Button Front Blazer", "BLACK")
# recommend("Ribbed Halter Knit Top", "WHITE")
# recommend("Pearl Ribbed Ring 3 Pack", "GOLD")
recommend("High Waist Faux Leather Short", "BLACK")
