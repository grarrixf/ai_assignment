import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

# Load the dataset
books = pd.read_csv('AmanzonBooks.csv', sep=',', encoding='latin-1')
books.dropna(subset=['genre'], inplace=True)

# Keep only relevant columns and rename them
books = books[['rank', 'bookTitle', 'bookPrice', 'rating', 'genre']]
books.rename(columns={'rank': 'no', 'bookTitle': 'title', 'bookPrice': 'price', 'rating': 'rate'}, inplace=True)

# Convert price and rating to numeric
books['price'] = pd.to_numeric(books['price'], errors='coerce')
books['rate'] = pd.to_numeric(books['rate'], errors='coerce')

# Pivot table to show the item feature matrix
book_pivot = pd.pivot_table(books, index='title', columns='genre', values=['price', 'rate'])
book_pivot.columns = book_pivot.columns.droplevel(0)
book_pivot.fillna(0, inplace=True)

# Convert pivot table to a sparse matrix
book_sparse = csr_matrix(book_pivot.values)

# Initialize the Nearest Neighbors model (K-Means)
model = NearestNeighbors(algorithm='brute')
model.fit(book_sparse)

# Set the index of books DataFrame to title
books.set_index('title', inplace=True)

# Function to recommend books
def recommend_books(book_features, n=5):
    book_features['genre'] = book_features['genre'].str.lower()
    recommended_books = []

    for genre in book_features['genre']:
        genre_matches = book_pivot.columns.str.lower() == genre
        if genre_matches.any():
            genre_books = book_pivot.loc[:, genre_matches]
            genre_books = genre_books[genre_books.sum(axis=1) > 0]
        else:
            closest_genre, score = process.extractOne(genre, books['genre'], scorer=process.WRatio)
            if score < 70:
                st.warning(f"No books found for the specific genre '{genre}'.")
                continue
            genre_books = book_pivot.loc[:, closest_genre]
            genre_books = genre_books[genre_books.sum(axis=1) > 0]

        if len(genre_books) > 0:
            distances, indices = model.kneighbors(book_pivot.loc[genre_books.index].values, n_neighbors=n+1)
            for idx in indices[0][1:]:
                recommended_books.append(book_pivot.index[idx])
        else:
            st.warning(f"No books found for the specific genre '{genre}'.")

    if recommended_books:
        recommended_books = sorted(recommended_books, key=lambda x: (books.loc[x, 'rate'], -books.loc[x, 'price']), reverse=True)
        st.write("\nRecommendations:")
        for i, book in enumerate(recommended_books[:n]):
            st.write(f"{i+1}: {book}")
    else:
        st.warning("No recommendations found.")

# Streamlit UI
st.title("Book Recommendation System")
st.sidebar.title("Input Book Details")

# Sidebar inputs
genre = st.sidebar.text_input("Enter Genre:", "Childrens")
price = st.sidebar.number_input("Enter Price:", min_value=0.0, max_value=100.0)
rating = st.sidebar.slider("Select Rating:", min_value=0.0, max_value=5.0, value=4.5, step=0.1)

if st.sidebar.button("Get Recommendations"):
    sample_book = pd.DataFrame({'price': [price], 'rate': [rating], 'genre': [genre]})
    recommend_books(sample_book)

st.sidebar.write("Created by Your Name")
