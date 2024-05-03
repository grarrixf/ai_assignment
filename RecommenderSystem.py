import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Function to update the classifier and metrics
def update_classifier_and_metrics():
    global X, y, clf
    X = books[['price', 'rate']]
    y = books['genre']

    for item in st.session_state.cart:
        book = books[books['title'] == item['title']]
        X = pd.concat([X, pd.DataFrame({'price': [book['price'].values[0]], 'rate': [book['rate'].values[0]]})])
        y = pd.concat([y, pd.Series([item['genre']])])

    valid_genres = books['genre'].unique()
    y = y[y.isin(valid_genres)]

    if not X.empty and not y.empty and X.shape[0] == y.shape[0]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        classification_rep = classification_report(y_test, y_pred)
        st.write("### Updated Classification Report")
        st.write(classification_rep)

        accuracy = accuracy_score(y_test, y_pred)
        st.write("### Updated Accuracy Score")
        st.write(f"Accuracy: {accuracy:.2f}")

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
                book_index = next((i for i, item in enumerate(st.session_state.cart) if item['title'] == row['title']), None)
                if book_index is not None:
                    st.session_state.cart[book_index]['quantity'] += 1
                else:
                    st.session_state.cart.append({'title': row['title'], 'quantity': 1, 'genre': row['genre']})
else:
    st.write("No books available in this genre.")

# Recommendation layout with scrollbar
st.write("# Book Recommendations")
st.write('---')

# Get recommendations based on selected books
if st.button('Get Recommendations'):
    selected_books_df = books[books['title'].isin([item['title'] for item in st.session_state.cart])]
    if not selected_books_df.empty:
        total_quantity = sum(item['quantity'] for item in st.session_state.cart)
        
        cart_genre_counts = selected_books_df['genre'].value_counts(normalize=True)
        for genre, percentage in cart_genre_counts.items():
            genre_books = books[books['genre'] == genre]
            num_recommended_books = max(int(total_quantity * percentage), 1)
            if num_recommended_books > 0:
                kmeans_model = KMeans(n_clusters=min(10, len(genre_books)), random_state=42)
                kmeans_model.fit(genre_books[['price', 'rate']])
                all_books_clusters = kmeans_model.predict(books[['price', 'rate']])
                selected_books_clusters = kmeans_model.predict(selected_books_df[['price', 'rate']])
                recommended_books_indices = [idx for idx, cluster in enumerate(all_books_clusters) if cluster in selected_books_clusters]
                if recommended_books_indices:
                    recommended_books_indices = recommended_books_indices[:min(len(recommended_books_indices), num_recommended_books)]
                    recommended_books_indices = [idx for idx in recommended_books_indices if idx < len(genre_books)]
                    if recommended_books_indices:
                        genre_recommended_books = genre_books.iloc[recommended_books_indices].drop_duplicates(subset='title', keep='first')
                        genre_recommended_books = genre_recommended_books[~genre_recommended_books['title'].isin([item['title'] for item in st.session_state.cart])]
                        genre_recommended_books = genre_recommended_books.head(num_recommended_books)
                        recommended_books = pd.concat([recommended_books, genre_recommended_books])
        
        if recommended_books.empty:
            recommended_books = pd.DataFrame(columns=books.columns)
        
        recommended_books['percentage'] = recommended_books['title'].apply(lambda x: st.session_state.cart[next((i for i, item in enumerate(st.session_state.cart) if item['title'] == x), None)]['quantity'] / total_quantity * 100 if next((i for i, item in enumerate(st.session_state.cart) if item['title'] == x), None) is not None else 0)
        
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
                    update_classifier_and_metrics()  # Update classifier when quantity is increased
            with col2:
                st.write(f"**Title:** {item['title']}")
                st.write(f"**Quantity:** {item['quantity']}")
            with col3:
                if st.button(f"# -", key=f"remove_{idx}"):
                    if st.session_state.cart[idx]['quantity'] > 1:
                        st.session_state.cart[idx]['quantity'] -= 1
                        update_classifier_and_metrics()  # Update classifier when quantity is decreased
                    else:
                        del st.session_state.cart[idx]  # Remove the item if quantity becomes zero
                        update_classifier_and_metrics()  # Update classifier when item is removed
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

# Function to update the classifier and metrics with correct quantities
def update_classifier_and_metrics():
    global X, y, clf
    X = books[['price', 'rate']]
    y = books['genre']

    for item in st.session_state.cart:
        book = books[books['title'] == item['title']]
        for _ in range(item['quantity']):  # Consider the quantity of each book in the cart
            X = pd.concat([X, pd.DataFrame({'price': [book['price'].values[0]], 'rate': [book['rate'].values[0]]})])
            y = pd.concat([y, pd.Series([item['genre']])])

    valid_genres = books['genre'].unique()
    y = y[y.isin(valid_genres)]

    if not X.empty and not y.empty and X.shape[0] == y.shape[0]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        classification_rep = classification_report(y_test, y_pred)
        st.write("### Updated Classification Report")
        st.write(classification_rep)

update_classifier_and_metrics()  # Initially update classifier and metrics
