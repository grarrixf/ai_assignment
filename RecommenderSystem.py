from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import reportort, accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.cluster import KMeans
import pandas as pd
import streamlit as st

# load the dataset
books = pd.read_csv('AmanzonBooks.csv')

# drop rows with missing values in the genre column
books.dropna(subset=['genre'], inplace=True)

# keep only relevant columns and rename
books = books[['rank', 'bookTitle', 'bookPrice', 'rating', 'genre']]
books.rename(columns={'rank': 'no', 'bookTitle': 'title', 'bookPrice': 'price', 'rating': 'rate'}, inplace=True)

# convert price and rating to numeric
books['price'] = pd.to_numeric(books['price'], errors='coerce')
books['rate'] = pd.to_numeric(books['rate'], errors='coerce')

# classifier and metric
def cam():
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
    validGenres = books['genre'].unique()
    y = y[y.isin(validGenres)]

    # no enough data to train model
    if not x.empty and not y.empty and x.shape[0] == y.shape[0]:
        # use class weights to handle imbalanced classes
        clf = RandomForestClassifier(random_state=42, class_weight='balanced')
        # predict classes using cross-validation
        yPred = cross_val_predict(clf, x, y, cv=5, fit_params={'sample_weight': cWeights(y)})
        
        report = reportort(y, yPred)
        st.write("### Classification Report")
        st.write(report)

        accuracy = accuracy_score(y, yPred)
        st.write("### Accuracy Score")
        st.write(f"Accuracy: {accuracy:.2f}")

# function to calculate sample weights based on class weights
def cWeights(y):
    cCounts = y.value_counts()
    cWeights = {genre: len(y) / (cCounts[genre] * len(cCounts)) for genre in cCounts.index}
    return y.map(cWeights)

# function to get recommended books
def getRbook(selectedBook, genreBook, numRbook):
    kmeans = KMeans(n_clusters=min(10, len(genreBook)), random_state=42)
    kmeans.fit(genreBook[['price', 'rate']])
    booksClusters = kmeans.predict(books[['price', 'rate']])
    sBooksClusters = kmeans.predict(selectedBook[['price', 'rate']])
    position = [idx for idx, cluster in enumerate(booksClusters) if cluster in sBooksClusters]
    if position:
        position = position[:min(len(position), numRbook)]
        position = [idx for idx in position if idx < len(genreBook)]
        if position:
            genreRbook = genreBook.iloc[position].drop_duplicates(subset='title', keep='first')
            genreRbook = genreRbook[~genreRbook['title'].isin([item['title'] for item in st.session_state.cart])]
            return genreRbook.head(numRbook)
    return pd.DataFrame(columns=books.columns)

# select books
st.sidebar.header('Books Recommender System')
selectedGenre = st.sidebar.radio("Select Genre", books['genre'].unique())

# filter books based on selected genre
genreFbook = books[books['genre'] == selectedGenre]

# cart
if 'cart' not in st.session_state:
    st.session_state.cart = []

# display available books
st.write("# Available Books")
st.write('---')

if not genreFbook.empty:
    with st.container(height=300): 
        for index, row in genreFbook.iterrows():
            add_to_cart = st.checkbox(f'Add to Cart: {row["title"]}', key=f"checkbox_{index}")
            if add_to_cart:
                book_index = next((i for i, item in enumerate(st.session_state.cart) if item['title'] == row['title']), None)
                if book_index is not None:
                    st.session_state.cart[book_index]['quantity'] += 1
                else:
                    st.session_state.cart.append({'title': row['title'], 'quantity': 1, 'genre': row['genre']})
else:
    st.write("No books available in this genre.")

# recommendation layout
st.write("# Book Recommendations")
st.write('---')

# get recommendations based on selected books
if st.button('Get Recommendations'):
    selectedBook = books[books['title'].isin([item['title'] for item in st.session_state.cart])]
    if not selectedBook.empty:
        tQuantity = sum(item['quantity'] for item in st.session_state.cart)
        rBook = pd.DataFrame() 
        cartGenreC = selectedBook['genre'].value_counts(normalize=True)
        for genre, percentage in cartGenreC.items():
            genreBook = books[books['genre'] == genre]
            numRbook = max(int(tQuantity * percentage), 1)
            if numRbook > 0:
                genreRbook = getRbook(selectedBook, genreBook, numRbook)
                if not genreRbook.empty:
                    rBook = pd.concat([rBook, genreRbook], ignore_index=True)
        
        if rBook.empty:
            st.write("No recommendations.")
        
        with st.container(height=300):
            for index, row in rBook.iterrows():
                st.write(f"**Title:** {row['title']}")
                st.write(f"**Genre:** {row['genre']}")
                st.write('---')
    else:
        st.write("No books selected.")

# cart layout
st.write("# Cart")
st.write('---')

tPrice = 0
if st.session_state.cart:
    st.write('## Items in Cart')
    with st.container(height=300): 
        for idx, item in enumerate(st.session_state.cart):
            col1, col2, col3 = st.columns([1, 10, 1])
            with col1:
                if st.button(f"# +", key=f"add_{idx}"):
                    st.session_state.cart[idx]['quantity'] += 1
                    cam()
            with col2:
                st.write(f"**Title:** {item['title']}")
                st.write(f"**Quantity:** {item['quantity']}")
            with col3:
                if st.button(f"# -", key=f"remove_{idx}"):
                    if st.session_state.cart[idx]['quantity'] > 1:
                        st.session_state.cart[idx]['quantity'] -= 1
                    else:
                        del st.session_state.cart[idx]
            tPrice += item['quantity'] * books.loc[books['title'] == item['title'], 'price'].iloc[0]
else:
    st.write("Your cart is empty.")

# checkout button
if st.session_state.cart:
    if st.button("Checkout"):
        st.session_state.cart = []  # clear cart

# display total price
st.write('---')
st.write(f"**Total Price:** USD {tPrice:.2f}")

cam()
