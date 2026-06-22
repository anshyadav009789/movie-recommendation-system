import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommendation System")
st.write("Select a movie and get similar movie recommendations.")


try:
    movies = pd.read_csv("movies.csv")
except FileNotFoundError:
    st.error("movies.csv file not found!")
    st.stop()


movies = movies[['title', 'genres']]

# Create Genre Vectors
cv = CountVectorizer()
vectors = cv.fit_transform(movies['genres']).toarray()


def recommend(movie_name):
    if movie_name not in movies['title'].values:
        return []

    movie_index = movies[movies['title'] == movie_name].index[0]

    similarities = cosine_similarity(
        vectors[movie_index].reshape(1, -1),
        vectors
    )[0]

    recommendations = sorted(
        list(enumerate(similarities)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    return [movies.iloc[i[0]].title for i in recommendations]


selected_movie = st.selectbox(
    "Choose a Movie",
    movies['title'].values
)


if st.button("Recommend Movies"):
    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies")

    for i, movie in enumerate(recommendations, start=1):
        st.write(f"{i}. {movie}")

st.markdown("---")
st.caption("Built with Streamlit")
