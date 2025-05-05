import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load data
movies = pd.read_csv("Movie-Recommendation-System/imdb_5000_movies.csv")
credits = pd.read_csv("Movie-Recommendation-System/imdb_5000_credits.csv")
movies = movies.merge(credits, on='title')

# Preprocess
def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def extract_cast(obj):
    return [i['name'] for i in ast.literal_eval(obj)[:3]]

def extract_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

movies['cast'] = movies['cast'].apply(extract_cast)
movies['crew'] = movies['crew'].apply(extract_director)
movies['overview'] = movies['overview'].fillna('')
movies['tags'] = movies['overview'] + " " + \
                 movies['genres'].apply(lambda x: " ".join(x)) + " " + \
                 movies['keywords'].apply(lambda x: " ".join(x)) + " " + \
                 movies['cast'].apply(lambda x: " ".join(x)) + " " + \
                 movies['crew'].apply(lambda x: " ".join(x))

# Vectorize
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    movie = movie.lower()
    if movie not in movies['title'].str.lower().values:
        return ["Movie not found"]
    index = movies[movies['title'].str.lower() == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movie_list]

# Streamlit UI
st.title("Movie Recommendation System")
movie_input = st.text_input("Enter a movie name:")

if st.button("Recommend"):
    recommendations = recommend(movie_input)
    for i in recommendations:
        st.write(i)
