# Machine-Learning-Project-
from flask import Flask, request, jsonify, make_response
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
app = Flask(__name__)

# Load movies
movies = pd.read_csv("movies.csv")
movies["processed_features"] = movies["genres"].fillna("")

# Vectorization
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["processed_features"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
movie_indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

# Helper to add CORS headers
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route("/")
def home():
    response = make_response("🎬 Welcome to the Movie Recommendation API!")
    return add_cors(response)

@app.route("/recommend", methods=["GET"])
def recommend():
    movie_name = request.args.get("movie")
    if not movie_name:
        return add_cors(jsonify({"error": "No movie name provided"})), 400

    if movie_name not in movie_indices:
        return add_cors(jsonify({"error": "Movie not found"})), 404

    idx = movie_indices[movie_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_movies = [movies["title"].iloc[i[0]] for i in sim_scores[1:6]]
    return add_cors(jsonify(top_movies))

if __name__ == "__main__":
    app.run(debug=True)
