"""
Inference script to generate predictions using Model Ensemble.
"""

import json
import numpy as np
import lightgbm as lgb
import pandas as pd
from pathlib import Path

from . import config, constants
from .data_processing import expand_candidates
from .features import FeatureGenerator # Assuming the fixed FeatureGenerator is used

def predict() -> None:
    # 1. Load Data
    print("Loading raw data...")
    targets_df = pd.read_csv(config.RAW_DATA_DIR / constants.TARGETS_FILENAME, dtype={constants.COL_USER_ID: "int32"})
    candidates_df = pd.read_csv(config.RAW_DATA_DIR / constants.CANDIDATES_FILENAME, dtype={constants.COL_USER_ID: "int32"})
    
    # Expand candidates
    candidates_expanded = expand_candidates(candidates_df)
    
    # Load base processed features (for metadata + SVD)
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    featured_df = pd.read_parquet(processed_path, engine="pyarrow")
    
    # 2. Extract full train history for aggregates
    full_train_history = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()
    
    # 3. Prepare Metadata for Candidates
    print("Merging metadata and text features...")
    # Get Metadata & SVD features from processed_df (unique by book_id)
    book_feats_cols = [
        constants.COL_BOOK_ID, constants.COL_AUTHOR_ID, constants.COL_PUBLICATION_YEAR,
        constants.COL_LANGUAGE, constants.COL_AVG_RATING, constants.COL_PUBLISHER
    ] + [c for c in featured_df.columns if "desc_svd" in c]
    
    book_feats = featured_df[book_feats_cols].drop_duplicates(subset=[constants.COL_BOOK_ID]).copy()
    
    # Merge Book Features into Candidates
    test_df = candidates_expanded.merge(book_feats, on=constants.COL_BOOK_ID, how="left")
    
    # Merge User Features
    user_data = pd.read_csv(config.RAW_DATA_DIR / constants.USER_DATA_FILENAME, dtype={constants.COL_USER_ID: "int32"})
    test_df = test_df.merge(user_data, on=constants.COL_USER_ID, how="left")
    
    # 4. Generate Interaction Features using FULL Train History
    print("Generating interaction features based on full history...")
    book_genres_df = pd.read_csv(config.RAW_DATA_DIR / constants.BOOK_GENRES_FILENAME)
    feat_gen = FeatureGenerator(full_train_history, book_genres_df)
    test_final = feat_gen.process(test_df.copy())
    
    # 5. Align Columns and Predict
    with open(config.MODEL_DIR / "features_list.json", "r") as f:
        features = json.load(f)
        
    # Handle missing columns that were only created in CV folds (e.g., if a new item appears)
    for f in features:
        if f not in test_final.columns:
            test_final[f] = 0.0
            
    # Handle Categoricals (Ensure type match)
    for col in config.CAT_FEATURES:
        if col in test_final.columns:
             test_final[col] = test_final[col].astype("category")

    X_test = test_final[features]
    
    # Ensemble Prediction
    print("Predicting...")
    ensemble_preds = np.zeros((len(X_test), 3)) 
    model_count = 0
    
    for fold in range(config.N_FOLDS):
        model_path = config.MODEL_DIR / config.MODEL_FILENAME_PATTERN.format(fold=fold)
        if model_path.exists():
            bst = lgb.Booster(model_file=str(model_path))
            ensemble_preds += bst.predict(X_test)
            model_count += 1
            
    if model_count == 0:
        raise RuntimeError("No models found! Cannot make predictions.")
            
    ensemble_preds /= model_count
    
    # Ranking Score: 2 * P(Read) + 1 * P(Planned)
    scores = ensemble_preds[:, 1] * 1.0 + ensemble_preds[:, 2] * 2.0
    test_final["score"] = scores
    
    # 6. Generate Submission
    print("Ranking candidates and generating submission...")
    submission_rows = []
    
    # Sort by User and Score descending
    candidates_final = test_final.sort_values([constants.COL_USER_ID, "score"], ascending=[True, False]).copy()
    
    # CRITICAL FIX: Ensure uniqueness of (user_id, book_id) pairs after scoring. 
    # This prevents inflated lists if feature engineering introduced duplicates.
    candidates_final.drop_duplicates(
        subset=[constants.COL_USER_ID, constants.COL_BOOK_ID], 
        keep='first', 
        inplace=True
    )
    
    grouped = candidates_final.groupby(constants.COL_USER_ID)
    
    for user_id in targets_df[constants.COL_USER_ID]:
        if user_id in grouped.groups:
            user_cands = grouped.get_group(user_id)
            
            # Top-K: Cap at MAX_RANKING_LENGTH (which is 20)
            top_k = user_cands.head(constants.MAX_RANKING_LENGTH)
            
            book_str = ",".join(top_k[constants.COL_BOOK_ID].astype(str))
        else:
            book_str = ""
            
        submission_rows.append({
            constants.COL_USER_ID: user_id,
            constants.COL_BOOK_ID_LIST: book_str
        })
        
    sub_df = pd.DataFrame(submission_rows)
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.SUBMISSION_DIR / constants.SUBMISSION_FILENAME
    sub_df.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")

if __name__ == "__main__":
    predict()