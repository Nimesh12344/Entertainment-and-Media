
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    reviews = pd.read_csv("IMDB Dataset.csv")
    return movies, ratings, reviews

movies, ratings, reviews = load_data()

movies.drop_duplicates(inplace=True)
ratings.drop_duplicates(inplace=True)
reviews.drop_duplicates(inplace=True)

movies['genres'] = movies['genres'].fillna('')
reviews['review'] = reviews['review'].fillna('')

if 'timestamp' in ratings.columns:
    ratings = ratings.drop(columns=['timestamp'])

data = pd.merge(ratings, movies, on='movieId')

@st.cache_resource
def build_recommendation():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    return cosine_sim, indices

cosine_sim, indices = build_recommendation()

def recommend(movie_title):
    if movie_title not in indices:
        return ["Movie not found"]
    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

@st.cache_resource
def train_sentiment():
    reviews['sentiment'] = reviews['sentiment'].map({'positive': 1, 'negative': 0})
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(reviews['review'])
    y = reviews['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, vectorizer, accuracy

model, vectorizer, accuracy = train_sentiment()

def predict_sentiment(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return "Positive 😊" if prediction[0] == 1 else "Negative 😞"

st.title("🎬 Entertainment & Media Analytics System")
st.write("Movie Recommendation + Sentiment Analysis")

st.subheader("🎥 Movie Recommendation")
movie_name = st.selectbox("Select a Movie", movies['title'].sort_values().unique())
if st.button("Recommend"):
    recommendations = recommend(movie_name)
    st.write("### Recommended Movies:")
    for movie in recommendations:
        st.write("•", movie)

st.subheader("💬 Sentiment Analysis")
st.write("Model Accuracy:", round(accuracy * 100, 2), "%")
user_review = st.text_area("Enter your movie review:")
if st.button("Analyze"):
    if user_review.strip() == "":
        st.warning("Please enter a review.")
    else:
        result = predict_sentiment(user_review)
        st.success(f"Predicted Sentiment: {result}")

st.subheader("📊 Ratings Distribution")
fig, ax = plt.subplots()
ax.hist(data['rating'], bins=10)
ax.set_title("Ratings Distribution")
ax.set_xlabel("Rating")
ax.set_ylabel("Frequency")
st.pyplot(fig)
