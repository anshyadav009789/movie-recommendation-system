import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")

movies = movies[['title', 'genres']]

cv = CountVectorizer()
vectors = cv.fit_transform(movies['genres']).toarray()

def recommend(movie_name):
    if movie_name not in movies['title'].values:
        print("\nMovie not found!")
        return

    movie_index = movies[movies['title'] == movie_name].index[0]

    movie_vector = vectors[movie_index]

    similarities = cosine_similarity(
        movie_vector.reshape(1, -1),
        vectors
    )[0]

    recommendations = sorted(
        list(enumerate(similarities)),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    print("\nRecommended Movies:\n")

    for movie in recommendations:
        print(movies.iloc[movie[0]].title)


while True:
    movie_name = input("\nEnter Movie Name (or 'exit' to quit): ")

    if movie_name.lower() == "exit":
        print("Goodbye!")
        break

    recommend(movie_name)