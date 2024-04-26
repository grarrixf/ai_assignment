import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

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

# Sidebar to select books
st.sidebar.header('Available Books')

# Display genres as buttons
selected_genre = st.sidebar.radio("Select Genre", books['genre'].unique())

# Filter books based on selected genre
genre_filtered_books = books[books['genre'] == selected_genre]

# Initialize cart
if 'cart' not in st.session_state:
    st.session_state.cart = []

# Display cart contents
st.sidebar.subheader('Cart')
items_to_remove = []
for idx, item in enumerate(st.session_state.cart):
    remove_button = st.sidebar.button(f"Remove: {item}", key=f"remove_{idx}")
    if remove_button:
        items_to_remove.append(item)

# Remove items from cart
for item in items_to_remove:
    st.session_state.cart.remove(item)

# Recommendation layout
st.write("# Book Recommendations")

# Get recommendations based on selected books
if st.button('Get Recommendations'):
    selected_books_df = books[books['title'].isin(st.session_state.cart)]
    if not selected_books_df.empty:
        # Perform KMeans clustering on the selected books
        kmeans_model = KMeans(n_clusters=min(10, len(selected_books_df)), random_state=42)
        kmeans_model.fit(selected_books_df[['price', 'rate']])
        # Predict clusters for all books
        all_books_clusters = kmeans_model.predict(books[['price', 'rate']])
        # Get cluster for selected books
        selected_books_clusters = kmeans_model.predict(selected_books_df[['price', 'rate']])
        # Filter books from the same cluster as selected books
        recommended_books_indices = [idx for idx, cluster in enumerate(all_books_clusters) if cluster in selected_books_clusters]
        recommended_books = books.iloc[recommended_books_indices].drop_duplicates(subset='title')
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
        add_to_cart = st.checkbox(f'Add to Cart: {row["title"]}')
        if add_to_cart:
            st.session_state.cart.append(row['title'])
else:
    st.write("No books available in this genre.")
