import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

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

# Initialize cart
if 'cart' not in st.session_state:
    st.session_state.cart = []

# Initialize recommended books DataFrame
recommended_books = pd.DataFrame(columns=books.columns)

# Display available books with scrollbar
st.write("# Available Books")
st.write('---')

if not genre_filtered_books.empty:
    with st.container(height=300):  # Set container height to display scrollbar
        for index, row in genre_filtered_books.iterrows():
            add_to_cart = st.checkbox(f'Add to Cart: {row["title"]}', key=f"checkbox_{index}")
            if add_to_cart:
                # Check if the book is already in the cart
                book_index = next((i for i, item in enumerate(st.session_state.cart) if item['title'] == row['title']), None)
                if book_index is not None:
                    st.session_state.cart[book_index]['quantity'] += 1  # Increment quantity if already in cart
                else:
                    st.session_state.cart.append({'title': row['title'], 'quantity': 1, 'genre': row['genre']})  # Add to cart with quantity 1 and genre
else:
    st.write("No books available in this genre.")

# Recommendation layout with scrollbar
st.write("# Book Recommendations")
st.write('---')

# Get recommendations based on selected books
if st.button('Get Recommendations'):
    selected_books_df = books[books['title'].isin([item['title'] for item in st.session_state.cart])]
    if not selected_books_df.empty:
        # Calculate total quantity of books in cart
        total_quantity = sum(item['quantity'] for item in st.session_state.cart)
        
        # Update genre counts based on cart
        cart_genre_counts = selected_books_df['genre'].value_counts(normalize=True)
        for genre, percentage in cart_genre_counts.items():
            # Filter books from the selected genre
            genre_books = books[books['genre'] == genre]
            # Calculate the number of recommended books for this genre
            num_recommended_books = max(int(total_quantity * percentage), 1)
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
                        genre_recommended_books = genre_recommended_books[~genre_recommended_books['title'].isin([item['title'] for item in st.session_state.cart])]
                        # Limit the number of recommended books for this genre
                        genre_recommended_books = genre_recommended_books.head(num_recommended_books)
                        # Add to recommended_books DataFrame
                        recommended_books = pd.concat([recommended_books, genre_recommended_books])
        
        # Initialize recommended_books if it's empty
        if recommended_books.empty:
            recommended_books = pd.DataFrame(columns=books.columns)
        
        # Calculate percentage of each recommended book in the cart
        recommended_books['percentage'] = recommended_books['title'].apply(lambda x: st.session_state.cart[next((i for i, item in enumerate(st.session_state.cart) if item['title'] == x), None)]['quantity'] / total_quantity * 100 if next((i for i, item in enumerate(st.session_state.cart) if item['title'] == x), None) is not None else 0)
        
        # Sort recommended books by percentage
        recommended_books = recommended_books.sort_values(by='percentage', ascending=False)
        
        with st.container(height=300):  # Set container height to display scrollbar
            for index, row in recommended_books.iterrows():
                st.write(f"**Title:** {row['title']}")
                st.write(f"**Genre:** {row['genre']}")
                st.write('---')
    else:
        st.write("No books selected.")

# Cart layout
st.write("# Cart")
st.write('---')

total_price = 0
if st.session_state.cart:
    st.write('## Items in Cart')
    with st.container(height=300):  # Set container height to display scrollbar
        for idx, item in enumerate(st.session_state.cart):
            col1, col2, col3 = st.columns([1, 10, 1])
            with col1:
                if st.button(f"# +", key=f"add_{idx}"):
                    st.session_state.cart[idx]['quantity'] += 1
                    # Recalculate percentage when quantity is increased
                    total_quantity = sum(item['quantity'] for item in st.session_state.cart)
                    recommended_books['percentage'] = recommended_books['title'].apply(lambda x: st.session_state.cart[next((i for i, item in enumerate(st.session_state.cart) if item['title'] == x), None)]['quantity'] / total_quantity * 100 if next((i for i, item in enumerate(st.session_state.cart) if item['title'] == x), None) is not None else 0)
                    recommended_books = recommended_books.sort_values(by='percentage', ascending=False)
                    # Update X and y
                    book = books[books['title'] == item['title']]
                    X = pd.concat([X, book[['price', 'rate']]])
                    y = pd.concat([y, pd.Series([item['genre']] * item['quantity'])])
            with col2:
                st.write(f"**Title:** {item['title']}")
                st.write(f"**Quantity:** {item['quantity']}")
            with col3:
                if st.button(f"# -", key=f"remove_{idx}"):
                    if st.session_state.cart[idx]['quantity'] > 1:
                        st.session_state.cart[idx]['quantity'] -= 1
                        # Recalculate percentage when quantity is decreased
                        total_quantity = sum(item['quantity'] for item in st.session_state.cart)
                        recommended_books['percentage'] = recommended_books['title'].apply(lambda x: st.session_state.cart[next((i for i, item in enumerate(st.session_state.cart) if item['title'] == x), None)]['quantity'] / total_quantity * 100 if next((i for i, item in enumerate(st.session_state.cart) if item['title'] == x), None) is not None else 0)
                        recommended_books = recommended_books.sort_values(by='percentage', ascending=False)
                        # Update X and y
                        book = books[books['title'] == item['title']]
                        X = pd.concat([X, book[['price', 'rate']]])
                        y = pd.concat([y, pd.Series([item['genre']] * item['quantity'])])
                    else:
                        del st.session_state.cart[idx]  # Remove the item if quantity becomes zero
                        # Remove corresponding entries from X and y
                        X = X[~((X['price'] == book['price'].iloc[0]) & (X['rate'] == book['rate'].iloc[0]))]
                        y = y[~((y == item['genre']) & (y.index == X.index))]
            total_price += item['quantity'] * books.loc[books['title'] == item['title'], 'price'].iloc[0]
else:
    st.write("Your cart is empty.")

# Checkout button
if st.session_state.cart:
    if st.button("Checkout"):
        st.session_state.cart = []  # Clear the cart upon checkout

# Display total price
st.write('---')
st.write(f"**Total Price:** USD {total_price:.2f}")

# Classification report and accuracy score
st.write('---')
st.write("## Classification Report and Accuracy Score")

# Features and target variable
X = books[['price', 'rate']]
y = books['genre']

# Adding items from the cart to the dataset
for item in st.session_state.cart:
    book = books[books['title'] == item['title']]
    X = pd.concat([X, book[['price', 'rate']]])
    y = pd.concat([y, pd.Series([item['genre']] * item['quantity'])])

# Check if y contains only valid genres
valid_genres = books['genre'].unique()
y = y[y.isin(valid_genres)]  # Keep only the genres that exist in the dataset

# Check the shapes of X and y
st.write("Shape of X:", X.shape)
st.write("Shape of y:", y.shape)

# Splitting the dataset
if not X.empty and not y.empty:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)

    # Classification report
    classification_rep = classification_report(y_test, y_pred)
    st.write("### Classification Report")
    st.write(classification_rep)

    # Accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    st.write("### Accuracy Score")
    st.write(f"Accuracy: {accuracy:.2f}")
else:
    st.write("Cannot perform classification. Please add items to your cart.") 
