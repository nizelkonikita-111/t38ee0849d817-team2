"""
Feature engineering script with advanced affinity and text processing.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from . import config, constants

class FeatureGenerator:
    """
    Generates features based on a historical training set to prevent leakage.
    """
    def __init__(self, train_df: pd.DataFrame, book_genres_df: pd.DataFrame):
        self.train_df = train_df
        self.book_genres_df = book_genres_df
        self.global_mean = train_df[config.TARGET].mean()
        
        # Pre-compute User-Genre history
        # Map books in train to their genres
        self.train_with_genres = train_df.merge(
            book_genres_df, on=constants.COL_BOOK_ID, how="inner"
        )

    def _smoothed_target_encoding(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Applies smoothed target encoding."""
        # Calculate stats on TRAIN history
        stats = self.train_df.groupby(col)[config.TARGET].agg(["sum", "count"])
        
        # Smoothed Mean = (sum + global * alpha) / (count + alpha)
        stats["smoothed_mean"] = (stats["sum"] + (self.global_mean * config.TE_ALPHA)) / \
                                 (stats["count"] + config.TE_ALPHA)
        
        # Map to the target DF
        mapper = stats["smoothed_mean"].to_dict()
        df[f"{col}_target_enc"] = df[col].map(mapper).fillna(self.global_mean)
        return df

    def add_affinity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds 'User-Genre Affinity':
        1. How many times has this user read this genre?
        2. What % of user's total history is this genre?
        """
        # We need to know the Genre of the books in 'df' (the candidates)
        # We assume df has COL_BOOK_ID
        df_w_genre = df.merge(self.book_genres_df, on=constants.COL_BOOK_ID, how="left")
        
        # 1. User History Counts per Genre (Computed on TRAIN history)
        user_genre_counts = self.train_with_genres.groupby(
            [constants.COL_USER_ID, constants.COL_GENRE_ID]
        ).size().reset_index(name="user_genre_history_count")
        
        # 2. Total User History Count
        user_total_counts = self.train_df.groupby(constants.COL_USER_ID).size().to_dict()
        
        # Merge history counts into current candidates
        # Note: A book can have multiple genres. We take the MAX affinity across all genres of the book.
        merged = df_w_genre.merge(
            user_genre_counts, 
            on=[constants.COL_USER_ID, constants.COL_GENRE_ID], 
            how="left"
        )
        merged["user_genre_history_count"] = merged["user_genre_history_count"].fillna(0)
        
        # Normalize by user activity (Ratio)
        merged["user_total_activity"] = merged[constants.COL_USER_ID].map(user_total_counts).fillna(1) # avoid div/0
        merged["user_genre_ratio"] = merged["user_genre_history_count"] / merged["user_total_activity"]
        
        # Aggregate back to (User, Book) level (taking max affinity if book has multiple genres)
        affinity_agg = merged.groupby([constants.COL_USER_ID, constants.COL_BOOK_ID]).agg({
            "user_genre_history_count": "max",
            "user_genre_ratio": "max"
        }).reset_index()
        
        df = df.merge(affinity_agg, on=[constants.COL_USER_ID, constants.COL_BOOK_ID], how="left")
        return df

    def add_interaction_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds standard aggregations."""
        
        # User Stats
        user_stats = self.train_df.groupby(constants.COL_USER_ID)[config.TARGET].agg(["count", "mean"])
        user_stats.columns = ["user_hist_count", "user_hist_mean_relevance"]
        df = df.merge(user_stats, on=constants.COL_USER_ID, how="left")
        
        # Book Stats
        book_stats = self.train_df.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(["count", "mean"])
        book_stats.columns = ["book_hist_count", "book_hist_mean_relevance"]
        df = df.merge(book_stats, on=constants.COL_BOOK_ID, how="left")
        
        # Fill NaNs (Cold Start)
        df["user_hist_count"] = df["user_hist_count"].fillna(0)
        df["user_hist_mean_relevance"] = df["user_hist_mean_relevance"].fillna(self.global_mean)
        df["book_hist_count"] = df["book_hist_count"].fillna(0)
        df["book_hist_mean_relevance"] = df["book_hist_mean_relevance"].fillna(self.global_mean)
        
        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main pipeline to apply all features to df."""
        df = self.add_interaction_stats(df)
        df = self._smoothed_target_encoding(df, constants.COL_AUTHOR_ID)
        df = self._smoothed_target_encoding(df, constants.COL_PUBLISHER)
        df = self.add_affinity_features(df)
        return df


def generate_text_features(df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates TF-IDF + SVD features. 
    This should be run ONCE on the unique books and merged.
    """
    print(f"Generating TF-IDF (Top {config.TFIDF_MAX_FEATURES}) + SVD ({config.TFIDF_SVD_COMPONENTS})...")
    
    # Prepare corpus
    descriptions_df[constants.COL_DESCRIPTION] = descriptions_df[constants.COL_DESCRIPTION].fillna("")
    corpus = descriptions_df[constants.COL_DESCRIPTION]
    
    # TF-IDF
    tfidf = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        stop_words='english',
        dtype=np.float32
    )
    tfidf_matrix = tfidf.fit_transform(corpus)
    
    # SVD (Dimensionality Reduction)
    svd = TruncatedSVD(n_components=config.TFIDF_SVD_COMPONENTS, random_state=config.RANDOM_STATE)
    svd_matrix = svd.fit_transform(tfidf_matrix)
    
    # Create DataFrame
    cols = [f"desc_svd_{i}" for i in range(config.TFIDF_SVD_COMPONENTS)]
    svd_df = pd.DataFrame(svd_matrix, columns=cols)
    svd_df[constants.COL_BOOK_ID] = descriptions_df[constants.COL_BOOK_ID].values
    
    # Merge
    df = df.merge(svd_df, on=constants.COL_BOOK_ID, how="left")
    
    # Fill missing (for books without descriptions)
    for c in cols:
        df[c] = df[c].fillna(0.0)
        
    return df