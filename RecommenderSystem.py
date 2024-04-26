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
    # Pivot table to create item feature matrix
    pivot_table = pd.pivot_table(data, index='title', columns='genre', values=['price', 'rate'])
    pivot_table.columns = pivot_table.columns.droplevel(0)
    pivot_table.fillna(0, inplace=True)
    
    # Convert pivot table to a sparse matrix
    from scipy.sparse import csr_matrix
    sparse_matrix = csr_matrix(pivot_table.values)
    
    # Initialize KMeans model
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    
    # Fit the model to the sparse matrix
    kmeans.fit(sparse_matrix)
    
    return kmeans

kmeans_model = perform_clustering(books)

# Display books by genre
genre_list = books['genre'].unique()

st.title('Books by Genre')

for genre in genre_list:
    st.header(genre)
    genre_books = books[books['genre'] == genre]
    st.write(genre_books[['title', 'price', 'rate']])

# Initialize shopping cart
if 'cart' not in st.session_state:
    st.session_state.cart = []

# Add selected books to the cart
st.sidebar.title('Shopping Cart')

for genre in genre_list:
    selected_books = st.sidebar.selectbox(f'Add {genre} books to cart:', books[books['genre'] == genre]['title'])
    if selected_books not in st.session_state.cart:
        st.session_state.cart.append(selected_books)

# Show shopping cart
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
        recommended_books_indices = kmeans_model.predict(selected_books_df[['price', 'rate']])
        recommended_books = pd.DataFrame(columns=books.columns)
        for index in recommended_books_indices:
            recommended_books = pd.concat([recommended_books, books[books.index == index]])
        st.write("Recommended Books:")
        st.write(recommended_books[['title', 'genre']])
    else:
        st.write("No books selected.")
