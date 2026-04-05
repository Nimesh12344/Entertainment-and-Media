import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
reviews = pd.read_csv("IMDB Dataset.csv")


print(movies.isnull().sum())
print(ratings.isnull().sum())
print(reviews.isnull().sum())

# Fill missing genres
movies['genres'] = movies['genres'].fillna('')

# Remove duplicates
movies.drop_duplicates(inplace=True)
ratings.drop_duplicates(inplace=True)
reviews.drop_duplicates(inplace=True)

data = pd.merge(ratings, movies, on='movieId')

# Remove timestamp (not needed)
data.drop(['timestamp'], axis=1, inplace=True)

print(data.head())  

print(ratings.describe())
print("Mean Rating:", ratings['rating'].mean())
print("Standard Deviation:", ratings['rating'].std())
print("Skewness:", ratings['rating'].skew())

plt.figure(figsize=(8,5))
sns.histplot(data['rating'], bins=10)
plt.title("Ratings Distribution")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))

top_movies = data.groupby('title')['rating'].count().sort_values(ascending=False).head(10)

top_movies.plot(kind='bar')

plt.title("Top 10 Most Rated Movies")
plt.xlabel("Movie Title")
plt.ylabel("Number of Ratings")

plt.xticks(rotation=45, ha='right')   # rotate properly
plt.tight_layout()                    # adjust spacing

plt.show()

genres = movies['genres'].str.split('|').explode()

plt.figure(figsize=(10,6))
sns.countplot(y=genres, order=genres.value_counts().index)
plt.title("Genre Distribution")
plt.tight_layout()
plt.show()

movie_stats = data.groupby('title')['rating'].agg(['mean','count'])
movie_stats.columns = ['avg_rating','rating_count']

print("Correlation Matrix:")
print(movie_stats.corr())

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(movies['genres'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommend(movie_title):
    if movie_title not in indices:
        return "Movie not found"

    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]

    return movies['title'].iloc[movie_indices]

# Example
print(recommend("Toy Story (1995)"))

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X = vectorizer.fit_transform(reviews['review'])
y = reviews['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Sentiment Model Accuracy:", accuracy_score(y_test, pred))

def predict_sentiment(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return "Positive" if prediction[0]==1 else "Negative"

print(predict_sentiment("This movie was amazing and fantastic!"))