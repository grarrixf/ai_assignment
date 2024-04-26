from sklearn.cluster import KMeans
import pandas as pd
import streamlit as st

# Load the dataset
books = pd.read_csv('AmanzonBooks.csv', sep=',', encoding='latin-1')

# Drop rows with missing values in the genre column
books.dropna(subset=['genre'], inplace=True)

# Keep only relevant columns and rename
books = books[['rank', 'bookTitle', 'bookPrice', 'rating', 'genre']]
books.rename(columns={'rank': 'no', 'bookTitle': 'title', 'bookPrice': 'price', 'rating': 'rate'}, inplace=True)

# Convert price and rating to numeric
books['price'] = pd.to_numeric(books['price'], errors='coerce')
books['rate'] = pd.to_numeric(books['rate'], errors='coerce')

# Initialize cart if it doesn't exist
if 'cart' not in st.session_state:
    st.session_state.cart = []

# Perform KMeans clustering
def perform_clustering(data):
    # Get the number of unique genres
    n_genres = len(data['genre'].unique())
    # Adjust the number of clusters based on the number of unique genres
    n_clusters = min(10, n_genres)
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(data[['price', 'rate']])
    return kmeans

# Sidebar to select books
st.sidebar.header('Add Books to Cart')
st.sidebar.subheader('Available Books')

# Display genres as buttons
selected_genre = st.sidebar.radio("Select Genre", books['genre'].unique())

# Filter books based on selected genre
genre_filtered_books = books[books['genre'] == selected_genre]

# Display cart contents
st.sidebar.subheader('Cart')
for idx, item in enumerate(st.session_state.cart):
    if st.sidebar.button(f"Remove: {item}", key=f"remove_{idx}"):
        st.session_state.cart.remove(item)
    else:
        st.sidebar.write(item)

# Recommendation layout
st.write("# Book Recommendations")

# Get recommendations based on selected books
if st.button('Get Recommendations'):
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
        recommended_books = genre_filtered_books.iloc[recommended_books_indices].drop_duplicates(subset='title')
        st.write("## Recommended Books")
        for index, row in recommended_books.iterrows():
            add_button = st.button(f"Add to Cart: {row['title']}")
            if add_button:
                st.session_state.cart.append(row['title'])
            st.write(f"**Title:** {row['title']}")
            st.write(f"**Genre:** {row['genre']}")
            st.write('---')
    else:
        st.write("No books selected.")

# Display available books
st.write("## Available Books")
if not genre_filtered_books.empty:
    # Display available books as a table with checkboxes
    for index, row in genre_filtered_books.iterrows():
        add_to_cart = st.checkbox(f"Add to Cart: {row['title']}", key=f"add_{index}")
        if add_to_cart:
            st.session_state.cart.append(row['title'])
    st.write(genre_filtered_books)
else:
    st.write("No books available in this genre.")
