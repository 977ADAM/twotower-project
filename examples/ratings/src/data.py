from __future__ import annotations

import pandas as pd
from rich.console import Console
from src.config import Config

from twotower import normalize_interactions, split_interactions

console = Console()


def load_movies(movies_path: str) -> pd.DataFrame:
    movies = pd.read_csv(movies_path)
    genres_split = movies["genres"].str.split("|")
    movies = movies.copy()
    movies["genre_1"] = genres_split.str[0].fillna("__unk__")
    movies["genre_2"] = genres_split.str[1].fillna("__unk__")
    movies["genre_3"] = genres_split.str[2].fillna("__unk__")
    return movies[["movieId", "genre_1", "genre_2", "genre_3"]]


def load_interactions(config: Config) -> pd.DataFrame:
    ratings = pd.read_csv(config.ratings_path)
    ratings["event_date"] = pd.to_datetime(ratings["timestamp"], unit="s").dt.normalize()

    sampled_users = (
        ratings["userId"]
        .drop_duplicates()
        .sample(n=config.sample_users, random_state=config.seed)
    )
    ratings = ratings[ratings["userId"].isin(sampled_users)].copy()
    console.print(f"Sampled {config.sample_users} users → {len(ratings):,} interactions")

    return normalize_interactions(
        ratings,
        query_col="userId",
        candidate_col="movieId",
        clicks_col="rating",
        positive_threshold=config.positive_threshold,
    )


def load_training_frames(
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interactions = load_interactions(config)
    movies_df = load_movies(config.movies_path)

    known_movie_ids = set(interactions["movieId"].unique())
    movies_df = movies_df[movies_df["movieId"].isin(known_movie_ids)].reset_index(drop=True)
    console.print(f"Movies in sample: {len(movies_df):,}")

    pos_ratio = interactions["label"].mean()
    console.print(f"Positive ratio: {pos_ratio:.2%}  (threshold ≥ {config.positive_threshold})")

    train_df, valid_df, test_df = split_interactions(
        interactions,
        validation_ratio=config.validation_ratio,
        test_ratio=config.test_ratio,
    )
    console.print(f"Train: {len(train_df):,}  Valid: {len(valid_df):,}  Test: {len(test_df):,}")
    return movies_df, train_df, valid_df, test_df
