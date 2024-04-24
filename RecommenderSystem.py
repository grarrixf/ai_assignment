#Book Recommender System Using Clustering | Collaborative Based
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

# load the dataset
books = pd.read_csv('AmanzonBooks.csv', sep=',', encoding='latin-1')

# print unique values in the genre column
print("Unique genres:", books['genre'].unique())

# drop rows with missing values in the genre column
books.dropna(subset=['genre'], inplace=True)

#print("Size of the dataset after dropping rows with missing genres:", books.shape)

# keep only relevant columns
books = books[['rank', 'bookTitle', 'bookPrice', 'rating', 'genre']]
# rename
books.rename(columns={'rank': 'no',
                      'bookTitle': 'title',
                      'bookPrice': 'price',
                      'rating': 'rate'
                     }, inplace=True)

# convert price to numeric
books['price'] = pd.to_numeric(books['price'], errors='coerce')

# convert rating to numeric
books['rate'] = pd.to_numeric(books['rate'], errors='coerce')

# pivot table to show the item feature matrix
book_pivot = pd.pivot_table(books, index='title', columns='genre', values=['price', 'rate'])

# drop multi-level column index
book_pivot.columns = book_pivot.columns.droplevel(0)

# fill missing values with zeros
book_pivot.fillna(0, inplace=True)

# convert pivot table to a sparse matrix
book_sparse = csr_matrix(book_pivot.values)

# initialize the Nearest Neighbors model (K-Means)
model = NearestNeighbors(algorithm='brute')

# fit the model to the sparse matrix
model.fit(book_sparse)

# set the index of books DataFrame to title
books.set_index('title', inplace=True)

def recommend_books(book_features, n=5):
    # convert genre to lowercase
    book_features['genre'] = book_features['genre'].str.lower()

    # array
    recommended_books = []

    # check if the provided genre match any genre in the dataset ignoring case
    for genre in book_features['genre']:
        genre_matches = book_pivot.columns.str.lower() == genre
        # if found filter books by the genre
        if genre_matches.any():
            genre_books = book_pivot.loc[:, genre_matches]
            genre_books = genre_books[genre_books.sum(axis=1) > 0]
        else:  # if no match find the closest one using fuzzy string matching
            closest_genre, score = process.extractOne(genre, books['genre'], scorer=process.WRatio)
            if score < 70:  # minimum similarity score
                print(f"No books found for the specific genre '{genre}'.")
                continue
            genre_books = book_pivot.loc[:, closest_genre]
            genre_books = genre_books[genre_books.sum(axis=1) > 0]

        # find nearest neighbors
        if len(genre_books) > 0:
            distances, indices = model.kneighbors(book_pivot.loc[genre_books.index].values, n_neighbors=n+1)
            # add recommended books
            for idx in indices[0][1:]:  # start from 1 to skip the first book
                recommended_books.append(book_pivot.index[idx])
        else:
            print(f"No books found for the specific genre '{genre}'.")

    # sort recommended books by rate (highest to lowest) and then by price (lowest to highest)
    if recommended_books:
        recommended_books = sorted(recommended_books, key=lambda x: (books.loc[x, 'rate'], -books.loc[x, 'price']), reverse=True)
        print()
        print("Recommendations:")
        for i, book in enumerate(recommended_books[:n]):
            print(f"{i+1}: {book}")
    else:
        print("No recommendations found.")

# test the recommendation function with a sample book
sample_book = pd.DataFrame({'price': [7], 'rate': [4.5], 'genre': ['Childrens']})
recommend_books(sample_book)
