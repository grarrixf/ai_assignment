import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

# load the dataset
books = pd.read_csv('AmanzonBooks.csv')

# drop rows with missing values
books.dropna(subset=['genre'], inplace=True)

# keep only relevant column
books = books[['rank', 'bookTitle', 'bookPrice', 'rating', 'genre']]
# rename
books.rename(columns={'rank': 'no', 'bookTitle': 'title', 'bookPrice': 'price', 'rating': 'rate'}, inplace=True)

# convert price and rating to numeric values
books['price'] = pd.to_numeric(books['price'], errors='coerce')
books['rate'] = pd.to_numeric(books['rate'], errors='coerce')

# sidebar to select books
st.sidebar.header('Books Recommender System')
# display genres for selection
selectedGenre = st.sidebar.radio("Select Genre", books['genre'].unique())
# filter books based on selected genre to display
genreFilter = books[books['genre'] == selectedGenre]

# initialize cart
if 'cart' not in st.session_state:
    st.session_state.cart = []

# initialize recommended books data frame
recommendedBooks = pd.DataFrame(columns=books.columns)

# available books section
st.write("# Available Books")
st.write('---')

if not genreFilter.empty:
    with st.container(height=300):  # set container height
        for index, row in genreFilter.iterrows():
            addToCart = st.checkbox(f'Add to Cart: {row["title"]}', key=f"checkbox_{index}")
            if addToCart:
                # check if the book is already in the cart
                bookPosition = next((i for i, item in enumerate(st.session_state.cart) if item['title'] == row['title']), None)
                if bookPosition is not None:
                    st.session_state.cart[bookPosition]['quantity'] += 1  # increase quantity if book already in cart
                else:
                    st.session_state.cart.append({'title': row['title'], 'quantity': 1})  # add to cart with quantity 1
else:
    st.write("No books available in this genre.")

# recommendation section
st.write("# Book Recommendations")
st.write('---')

# get recommendations based on selected books
if st.button('Get Recommendations'):
    selectedBook = books[books['title'].isin([item['title'] for item in st.session_state.cart])]
    
    if not selectedBook.empty:  
        # calculate total quantity of books in cart
        tQuantity = sum(item['quantity'] for item in st.session_state.cart)
        
        # update genre counts based on cart
        cartGenreCounts = selectedBook['genre'].value_counts(normalize=True)
        
        for genre, percentage in cartGenreCounts.items():
            # filter books from the selected genre
            genreBooks = books[books['genre'] == genre]
            
            # calculate the number of recommended book for this genre
            numRecommendedBook = max(int(tQuantity * percentage), 1)
            
            # perform KMeans on the genre book
            if numRecommendedBook > 0:
                kmeans = KMeans(allBooksClusters =min(10, len(genreBooks)), randomState=42)
                kmeans.fit(genreBooks[['price', 'rate']])
                
                # predict clusters for all books
                allBooksClusters = kmeans.predict(books[['price', 'rate']])
                
                # get a cluster for selected books
                selectedBooksClusters = kmeans.predict(selectedBook[['price', 'rate']])
                
                # filter books from the same cluster as selected books
                recommendedBookPosition = [idx for idx, cluster in enumerate(allBooksClusters) if cluster in selectedBooksClusters]
                
                # ensure recommendedBookPosition is not empty and within the range of the data frame index
                if recommendedBookPosition:
                    recommendedBookPosition = recommendedBookPosition[:min(len(recommendedBookPosition), numRecommendedBook)]
                    recommendedBookPosition = [idx for idx in recommendedBookPosition if idx < len(genreBooks)]
                    if recommendedBookPosition:
                        genreRecommendedBook = genreBooks.iloc[recommendedBookPosition].drop_duplicates(subset='title', keep='first')
                        
                        # exclude books already in the cart
                        genreRecommendedBook = genreRecommendedBook[~genreRecommendedBook['title'].isin([item['title'] for item in st.session_state.cart])]
                        
                        # limit the number of recommended book for this genre
                        genreRecommendedBook = genreRecommendedBook.head(numRecommendedBook)
                        
                        # add to recommendedBooks data frame
                        recommendedBooks = pd.concat([recommendedBooks, genreRecommendedBook])
        
        # initialize recommendedBooks if emtpy
        if recommendedBooks.empty:
            recommendedBooks = pd.DataFrame(columns=books.columns)
        
        # calculate percentage of each recommended book in the cart
        recommendedBooks['percentage'] = recommendedBooks['title'].apply(lambda x: st.session_state.cart[next((i for i, item in enumerate(st.session_state.cart) 
                                                                                                               if item['title'] == x), None)]
                                                                         ['quantity'] / tQuantity * 100 
                                                                         if next((i for i, item in enumerate(st.session_state.cart) 
                                                                                  if item['title'] == x), None) is not None else 0)
        
        # sort recommended book by percentage
        recommendedBooks = recommendedBooks.sort_values(by='percentage', ascending=False)
        
        with st.container(height=300):
            for index, row in recommendedBooks.iterrows():
                st.write(f"**Title:** {row['title']}")
                st.write(f"**Genre:** {row['genre']}")
                st.write('---')
    else:
        st.write("No books selected.")

# cart section
st.write("# Cart")
st.write('---')

# default price value
tPrice = 0

if st.session_state.cart:
    
    st.write('## Items in Cart')
    
    with st.container(height=300):
        
        for idx, item in enumerate(st.session_state.cart):
            
            col1, col2, col3 = st.columns([1, 10, 1])
            
            with col1:
                
                if st.button(f"# +", key=f"add_{idx}"):
                    
                    st.session_state.cart[idx]['quantity'] += 1
                    
                    # recalculate percentage when quantity is increased
                    tQuantity = sum(item['quantity'] for item in st.session_state.cart)
                    recommendedBooks['percentage'] = recommendedBooks['title'].apply(lambda x: st.session_state.cart[next((i for i, item in enumerate(st.session_state.cart) 
                                                                                                                           if item['title'] == x), None)]
                                                                                     ['quantity'] / tQuantity * 100 if next((i for i, item in enumerate(st.session_state.cart) 
                                                                                                                                  if item['title'] == x), None) is not None else 0)
                    recommendedBooks = recommendedBooks.sort_values(by='percentage', ascending=False)
            
            with col2:
                
                st.write(f"**Title:** {item['title']}")
                
                st.write(f"**Quantity:** {item['quantity']}")
                
            with col3:
                
                if st.button(f"# -", key=f"remove_{idx}"):
                    
                    if st.session_state.cart[idx]['quantity'] > 1:
                        
                        st.session_state.cart[idx]['quantity'] -= 1
                        
                        # recalculate percentage when quantity is decreased
                        tQuantity = sum(item['quantity'] for item in st.session_state.cart)
                        
                        recommendedBooks['percentage'] = recommendedBooks['title'].apply(lambda x: st.session_state.cart[next((i for i, item in enumerate(st.session_state.cart) 
                                                                                                                               if item['title'] == x), None)]
                                                                                         ['quantity'] / tQuantity * 100 
                                                                                         if next((i for i, item in enumerate(st.session_state.cart) 
                                                                                                                                      if item['title'] == x), None) is not None else 0)
                        
                        recommendedBooks = recommendedBooks.sort_values(by='percentage', ascending=False)
                    
                    else:
                        del st.session_state.cart[idx]  # remove the item if quantity is zero
                        
            tPrice += item['quantity'] * books.loc[books['title'] == item['title'], 'price'].iloc[0]
else:
    st.write("Cart is empty.")

# checkout button
if st.session_state.cart:
    if st.button("Checkout"):
        st.session_state.cart = []  # clear the cart if checkout

# display total price
st.write('---')
st.write(f"**Total Price:** USD {tPrice:.2f}")
