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
        # Calculate the percentage of each genre in the cart
        genre_counts = selected_books_df['genre'].value_counts(normalize=True)
        recommended_books = pd.DataFrame(columns=books.columns)
        for genre, percentage in genre_counts.items():
            # Filter books from the selected genre
            genre_books = books[books['genre'] == genre]
            # Calculate the number of recommended books for this genre
            num_recommended_books = max(int(len(selected_books_df) * percentage), 1)
            # Perform KMeans clustering on the genre books
            if num_recommended_books > 0:
                kmeans_model = KMeans(n_clusters=min(10, len(genre_books)), random_state=42)
                kmeans_model.fit(genre_books[['price', 'rate']])
                # Predict clusters for all books
                all_books_clusters = kmeans_model.predict(books[['price', 'rate']])
                # Get cluster for selected books
                selected_books_clusters = kmeans_model.predict(selected_books_df[['price', 'rate']])
                # Filter books from the same cluster as selected books
                recommended_books_indices = [idx for idx, cluster in enumerate(all_books_clusters) if cluster in selected_books_clusters]
                # Ensure recommended_books_indices is not empty and within the bounds of the DataFrame's index
                if recommended_books_indices:
                    recommended_books_indices = recommended_books_indices[:min(len(recommended_books_indices), num_recommended_books)]
                    recommended_books_indices = [idx for idx in recommended_books_indices if idx < len(genre_books)]
                    if recommended_books_indices:
                        genre_recommended_books = genre_books.iloc[recommended_books_indices].drop_duplicates(subset='title', keep='first')
                        # Limit the number of recommended books for this genre
                        genre_recommended_books = genre_recommended_books.head(num_recommended_books)
                        recommended_books = pd.concat([recommended_books, genre_recommended_books])
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
