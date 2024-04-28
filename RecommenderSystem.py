import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

# Load the dataset
books = pd.read_csv('AmanzonBooks.csv')

# Drop rows with missing values in the genre column
books.dropna(subset=['genre'], inplace=True)

# Keep only relevant columns and rename
books = books[['rank', 'bookTitle', 'bookPrice', 'rating', 'genre']]
books.rename(columns={'rank': 'no', 'bookTitle': 'title', 'bookPrice': 'price', 'rating': 'rate'}, inplace=True)

# Convert price and rating to numeric
books['price'] = pd.to_numeric(books['price'], errors='coerce')
books['rate'] = pd.to_numeric(books['rate'], errors='coerce')

# Sidebar to select books
st.sidebar.header('Books Recommender System')

# Display genres as buttons
selected_genre = st.sidebar.radio("Select Genre", books['genre'].unique())

# Filter books based on selected genre
genre_filtered_books = books[books['genre'] == selected_genre]

# Initialize cart as a dictionary to store book title and quantity
if 'cart' not in st.session_state:
    st.session_state.cart = {}

# Display available books with scrollbar
st.write("# Available Books")
st.write('---')

if not genre_filtered_books.empty:
    with st.container(height=300):  # Set container height to display scrollbar
        for index, row in genre_filtered_books.iterrows():
            add_to_cart = st.button(f'Add to Cart: {row["title"]}', key=f"button_{index}")
            if add_to_cart:
                print(f"Adding {row['title']} to cart...")
                title = row['title']
                if title in st.session_state.cart:
                    print(f"Item {title} already in cart")
                    st.session_state.cart[title]['quantity'] += 1
                else:
                    print(f"Adding new item {title} to cart")
                    st.session_state.cart[title] = {'price': row['price'], 'quantity': 1}
else:
    st.write("No books available in this genre.")

# Recommendation layout with scrollbar
st.write("# Book Recommendations")
st.write('---')

# Get recommendations based on selected books
if st.button('Get Recommendations'):
    selected_books_df = books[books['title'].isin(st.session_state.cart.keys())]
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
                        # Exclude books already in the cart
                        genre_recommended_books = genre_recommended_books[~genre_recommended_books['title'].isin(st.session_state.cart)]
                        # Limit the number of recommended books for this genre
                        genre_recommended_books = genre_recommended_books.head(num_recommended_books)
                        recommended_books = pd.concat([recommended_books, genre_recommended_books])
        with st.container(height=300):  # Set container height to display scrollbar
            for index, row in recommended_books.iterrows():
                st.write(f"**Title:** {row['title']}")
                st.write(f"**Genre:** {row['genre']}")
                st.write('---')
    else:
        st.write("No books selected.")

# Cart display
st.write("# Cart")
st.write('---')

total_price = 0
if st.session_state.cart:
    for title, book_info in st.session_state.cart.items():
        st.write(f"**Title:** {title}")
        st.write(f"**Price:** ${book_info['price']:.2f}")
        col1, col2, col3 = st.columns(3)
        with col1:
            if col1.button("+"):
                st.session_state.cart[title]['quantity'] += 1
        with col2:
            st.write(f"**Quantity:** {book_info['quantity']}")
        with col3:
            if col3.button("-"):
                if st.session_state.cart[title]['quantity'] > 1:
                    st.session_state.cart[title]['quantity'] -= 1
        st.write('---')
        total_price += book_info['price'] * book_info['quantity']
else:
    st.write("Your cart is empty.")

# Display total price
st.write(f"**Total Price:** ${total_price:.2f}")
