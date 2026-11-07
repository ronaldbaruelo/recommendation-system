import ast

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    credits = credits.rename(columns={"movie_id": "id"})
    movies = movies.merge(credits[["id", "cast", "crew"]], on="id", how="left")
    return movies


def get_director(x):
    for i in x or []:
        if i.get("job") == "Director":
            return i.get("name")
    return np.nan


def get_list(x):
    if isinstance(x, list):
        names = [i.get("name") for i in x if isinstance(i, dict) and i.get("name")]
        if len(names) > 3:
            names = names[:3]
        return names
    return []


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(str(i).replace(" ", "")) for i in x]
    if isinstance(x, str):
        return str.lower(x.replace(" ", ""))
    return ""


def build_recommendation_model(df):
    feature_cols = ["cast", "crew", "keywords", "genres"]
    for feature in feature_cols:
        df[feature] = df[feature].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    df["director"] = df["crew"].apply(get_director)
    for feature in ["cast", "keywords", "genres"]:
        df[feature] = df[feature].apply(get_list)

    for feature in ["cast", "keywords", "director", "genres"]:
        df[feature] = df[feature].apply(clean_data)

    df["soup"] = df.apply(
        lambda x: " ".join(x["keywords"]) + " " + " ".join(x["cast"]) + " " + x["director"] + " " + " ".join(x["genres"]),
        axis=1,
    )

    count = CountVectorizer(stop_words="english")
    count_matrix = count.fit_transform(df["soup"])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
    return df, cosine_sim, indices


def get_recommendations(title, df, cosine_sim, indices, limit=10):
    if title not in indices:
        return pd.DataFrame()

    idx = indices[title]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)
    sim_scores = [score for score in sim_scores if score[0] != idx][:limit]
    movie_indices = [i[0] for i in sim_scores]

    recommendations = df.iloc[movie_indices][["title", "genre_names", "vote_average", "vote_count", "overview"]].copy()
    recommendations = recommendations.rename(columns={"genre_names": "genres"})
    return recommendations.reset_index(drop=True)


def weighted_rating(x, m, C):
    v = x["vote_count"]
    R = x["vote_average"]
    return (v / (v + m) * R) + (m / (m + v) * C)


@st.cache_data
def get_top_rated_movies(df):
    C = df["vote_average"].mean()
    m = df["vote_count"].quantile(0.9)
    q_movies = df.copy().loc[df["vote_count"] >= m].copy()
    q_movies["score"] = q_movies.apply(lambda row: weighted_rating(row, m, C), axis=1)
    q_movies = q_movies.sort_values("score", ascending=False)
    return q_movies[["title", "vote_average", "vote_count", "score"]].head(10)


def main():
    st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")

    movies = load_data()

    if "genres" in movies.columns:
        movies["genre_names"] = movies["genres"].apply(
            lambda x: ", ".join([i.get("name") for i in ast.literal_eval(x)] if isinstance(x, str) else [])
        )
    else:
        movies["genre_names"] = ""

    movies["overview"] = movies["overview"].fillna("")

    df, cosine_sim, indices = build_recommendation_model(movies)

    st.title("🎬 Movie Recommendation System")
    st.caption("Content-based movie recommendations from The Movie Database (TMDB) dataset")

    movie_list = sorted(df["title"].dropna().unique().tolist())
    selected_movie = st.selectbox("Choose a movie", movie_list)

    if selected_movie:
        related = get_recommendations(selected_movie, df, cosine_sim, indices, limit=10)

        st.subheader(f"Recommended movies for: {selected_movie}")
        if related.empty:
            st.warning("No recommendations found for this title.")
        else:
            st.dataframe(
                related[["title", "genres", "vote_average", "vote_count"]],
                hide_index=True,
                use_container_width=True,
            )

            st.subheader("Top picks")
            for _, row in related.head(5).iterrows():
                with st.container():
                    st.markdown(f"### {row['title']}")
                    st.write(f"Genres: {row['genres']}")
                    st.write(f"Rating: {row['vote_average']} | Votes: {row['vote_count']}")
                    st.write(row.get("overview", "")[:220] + ("..." if len(row.get("overview", "")) > 220 else ""))

    st.markdown("---")
    st.subheader("Top Rated Movies")
    top_rated = get_top_rated_movies(df)
    st.dataframe(top_rated, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
