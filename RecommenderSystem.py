import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

# Load the dataset
@st.cache_data
def load_data():
    books = pd.read_csv('AmanzonBooks.csv', sep=',', encoding='latin-1')
    books.dropna(subset=['genre'], inplace=True)
    books = books[['bookTitle', 'bookPrice', 'rating', 'genre']]
    books.rename(columns={'bookTitle': 'title', 'bookPrice': 'price', 'rating': 'rate'}, inplace=True)
    books['price'] = pd.to_numeric(books['price'], errors='coerce')
    books['rate'] = pd.to_numeric(books['rate'], errors='coerce')
    return books

books = load_data()

# Perform KMeans clustering
def perform_clustering(data):
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    kmeans.fit(data[['price', 'rate']])
    return kmeans

kmeans_model = perform_clustering(books)

# Display books by genre
def display_books_by_genre(genre):
    genre_books = books[books['genre'] == genre]
    for _, row in genre_books.iterrows():
        if st.button(f"Add '{row['title']}' to Cart"):
            st.session_state.cart.append(row['title'])
    st.write(genre_books[['title', 'price', 'rate']])

# Initialize shopping cart
if 'cart' not in st.session_state:
    st.session_state.cart = []

st.title('Books by Genre')

# Display genre buttons
genres = books['genre'].unique()
selected_genre = st.sidebar.selectbox('Select a Genre:', genres)

# Display books for the selected genre
if selected_genre:
    st.header(selected_genre)
    display_books_by_genre(selected_genre)

# Show shopping cart
st.sidebar.title('Shopping Cart')
st.sidebar.subheader('Your Cart')
for item in st.session_state.cart:
    st.sidebar.write(item)

# Clear the cart
if st.sidebar.button('Clear Cart'):
    st.session_state.cart = []

# Recommend books based on selected books
if st.sidebar.button('Get Recommendations'):
    selected_books_df = books[books['title'].isin(st.session_state.cart)]
    if not selected_books_df.empty:
        # Get the most frequent genre among the selected books
        most_frequent_genre = selected_books_df['genre'].mode().iat[0]
        # Filter books of the most frequent genre
        genre_filtered_books = books[books['genre'] == most_frequent_genre]
        # Perform KMeans clustering on the filtered books
        kmeans_model = perform_clustering(genre_filtered_books)
        # Predict clusters for the selected books
        recommended_books_indices = kmeans_model.predict(selected_books_df[['price', 'rate']])
        # Get the recommended books from the same genre as the majority of selected books
        recommended_books = genre_filtered_books.iloc[recommended_books_indices]
        st.write("Recommended Books:")
        st.write(recommended_books[['title', 'genre']])
    else:
        st.write("No books selected.")
