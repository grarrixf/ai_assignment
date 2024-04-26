import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

# Load the dataset
@st.cache_data  # Use st.cache_data instead of st.cache
def load_data():
    books = pd.read_csv('AmanzonBooks.csv', sep=',', encoding='latin-1')
    books.dropna(subset=['genre'], inplace=True)
    books = books[['bookTitle', 'bookPrice', 'rating', 'genre']]
    books.rename(columns={'bookTitle': 'title', 'bookPrice': 'price', 'rating': 'rate'}, inplace=True)
    books['price'] = pd.to_numeric(books['price'], errors='coerce')
    books['rate'] = pd.to_numeric(books['rate'], errors='coerce')
    return books

books = load_data()

# Sidebar - Book Selection
st.sidebar.title('Select Books')
selected_books = st.sidebar.multiselect('Select books:', books['title'])

# Filter books based on selected titles
selected_books_df = books[books['title'].isin(selected_books)]

# Display selected books
if not selected_books_df.empty:
    st.subheader('Selected Books:')
    st.write(selected_books_df)

# Sidebar - K-Means clustering
st.sidebar.title('K-Means Clustering')
num_clusters = st.sidebar.slider('Select number of clusters:', min_value=2, max_value=10, value=5)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
book_features = books[['price', 'rate']].values
kmeans.fit(book_features)
books['cluster'] = kmeans.labels_

# Display clusters
st.header('Clusters')
st.write(books[['title', 'cluster']])

# Define function to recommend books based on selected books
def recommend_books(selected_books_df, num_recommendations=5):
    if selected_books_df.empty:
        st.write("Please select some books first.")
        return None
    
    selected_genres = selected_books_df['genre'].tolist()
    genre_filtered_books = books[books['genre'].isin(selected_genres)]
    
    if genre_filtered_books.empty:
        st.write("No matching books found in the dataset.")
        return None
    
    # Calculate mean rating for selected books
    mean_rate = selected_books_df['rate'].mean()
    
    # Get the most frequent genre among selected books
    most_frequent_genre = selected_books_df['genre'].mode().iat[0]
    
    # Filter books by most frequent genre
    genre_filtered_books = genre_filtered_books[genre_filtered_books['genre'] == most_frequent_genre]
    
    # Filter books based on price
    max_price = selected_books_df['price'].max()
    genre_filtered_books = genre_filtered_books[genre_filtered_books['price'] <= max_price]
    
    # Exclude selected books from recommendations
    recommended_books = genre_filtered_books[~genre_filtered_books['title'].isin(selected_books_df['title'])]
    
    # Sort recommended books by rating and price
    recommended_books = recommended_books.sort_values(by=['rate', 'price'], ascending=[False, True]).head(num_recommendations)
    
    return recommended_books

# Call the recommend_books function and display recommendations
recommended_books = recommend_books(selected_books_df)
if recommended_books is not None and not recommended_books.empty:
    st.subheader('Recommended Books:')
    st.write(recommended_books)

# Add selected books to the cart
if not selected_books_df.empty:
    st.sidebar.subheader('Shopping Cart')
    for index, row in selected_books_df.iterrows():
        st.sidebar.write(row['title'], row['genre'], row['price'], row['rate'])

# Add recommended books to the cart
if recommended_books is not None and not recommended_books.empty:
    st.sidebar.subheader('Shopping Cart')
    for index, row in recommended_books.iterrows():
        st.sidebar.write(row['title'], row['genre'], row['price'], row['rate'])
