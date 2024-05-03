from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.cluster import KMeans
import pandas as pd
import streamlit as st

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

# classifier and metric
def update_classifier_and_metrics():
    global x, y, clf
    x = books[['price', 'rate']]
    y = books['genre']

    # update x and y following the item in the cart
    for item in st.session_state.cart:
        book = books[books['title'] == item['title']]
        for _ in range(item['quantity']): 
            x = pd.concat([x, pd.DataFrame({'price': [book['price'].values[0]], 'rate': [book['rate'].values[0]]})])
            y = pd.concat([y, pd.Series([item['genre']])])

    # filter out invalid genre
    valid_genres = books['genre'].unique()
    y = y[y.isin(valid_genres)]

    # no enough data to train model
    if not x.empty and not y.empty and x.shape[0] == y.shape[0]:
        # use class weights to handle imbalanced classes
        clf = RandomForestClassifier(random_state=42, class_weight='balanced')
        # predict classes using cross-validation
        y_pred = cross_val_predict(clf, x, y, cv=5, fit_params={'sample_weight': calculate_sample_weights(y)})
        
        classification_rep = classification_report(y, y_pred)
        st.write("### Classification Report")
        st.write(classification_rep)

        accuracy = accuracy_score(y, y_pred)
        st.write("### Accuracy Score")
        st.write(f"Accuracy: {accuracy:.2f}")

# Function to calculate sample weights based on class weights
def calculate_sample_weights(y):
    class_counts = y.value_counts()
    class_weights = {genre: len(y) / (class_counts[genre] * len(class_counts)) for genre in class_counts.index}
    return y.map(class_weights)

# Function to get recommended books
def get_recommended_books(selected_books_df, genre_books, num_recommended_books):
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
            return genre_recommended_books.head(num_recommended_books)
    return pd.DataFrame(columns=books.columns)

# Sidebar to select books
st.sidebar.header('Books Recommender System')

# Display genres as buttons
selected_genre = st.sidebar.radio("Select Genre", books['genre'].unique())

# Filter books based on selected genre
genre_filtered_books = books[books['genre'] == selected_genre]

# Initialize cart
if 'cart' not in st.session_state:
    st.session_state.cart = []

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
        recommended_books = pd.DataFrame() 
        cart_genre_counts = selected_books_df['genre'].value_counts(normalize=True)
        for genre, percentage in cart_genre_counts.items():
            genre_books = books[books['genre'] == genre]
            num_recommended_books = max(int(total_quantity * percentage), 1)
            if num_recommended_books > 0:
                genre_recommended_books = get_recommended_books(selected_books_df, genre_books, num_recommended_books)
                if not genre_recommended_books.empty:
                    recommended_books = pd.concat([recommended_books, genre_recommended_books], ignore_index=True)
        
        if recommended_books.empty:
            st.write("No recommendations.")
        
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

update_classifier_and_metrics()
