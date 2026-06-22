# Movie Recommendation System

A Content-Based Movie Recommendation System built using Python, Pandas, Scikit-Learn, and Cosine Similarity.

##Project Overview

This project recommends similar movies based on their genres. The recommendation engine uses Natural Language Processing (NLP) techniques and Cosine Similarity to find movies with similar genre patterns.

## 🚀 Features

* Search for a movie title
* Get top 5 similar movie recommendations
* Built using Machine Learning concepts

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* CountVectorizer
* Cosine Similarity
* Streamlit 

## 📂 Project Structure

movie-recommendation-system/

├── app.py

├── movies.csv

├── requirements.txt

├── README.md

└── .gitignore


## 🧠 How It Works

1. Load movie dataset.
2. Extract movie genres.
3. Convert genres into numerical vectors using CountVectorizer.
4. Calculate similarity using Cosine Similarity.
5. Recommend top 5 most similar movies.

## 📊 Example

Input:

Toy Story (1995)

Output:

* Toy Story 2 (1999)
* Antz (1998)
* Monsters, Inc. (2001)
* Emperor's New Groove, The (2000)
* Adventures of Rocky and Bullwinkle, The (2000)

