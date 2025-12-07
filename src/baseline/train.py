"""
Training script with strict temporal feature generation.
"""

import json
import gc
import lightgbm as lgb
import numpy as np
import pandas as pd

# FIX: Import config module
from . import config, constants
from .features import FeatureGenerator

def train() -> None:
    # 1. Load Preprocessed Data (Base features: ID, Meta, SVD)
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME
    print(f"Loading data from {processed_path}...")
    base_df = pd.read_parquet(processed_path)
    
    # Load separate Genre helper
    book_genres_df = pd.read_csv(config.RAW_DATA_DIR / constants.BOOK_GENRES_FILENAME)

    # Filter for Train Source
    full_train = base_df[base_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()
    full_train[constants.COL_TIMESTAMP] = pd.to_datetime(full_train[constants.COL_TIMESTAMP])
    full_train = full_train.sort_values(constants.COL_TIMESTAMP).reset_index(drop=True)

    # 2. Expanding Window Split
    n_samples = len(full_train)
    fold_size = int(n_samples * 0.2) # 20% for validation in each fold
    start_idx = int(n_samples * 0.4) # Start training with 40% data
    
    model_dir = config.MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    feature_names = None

    for fold in range(config.N_FOLDS):
        print(f"\n{'='*10} Fold {fold+1} {'='*10}")
        
        # Dynamic Split
        train_end = start_idx + (fold * fold_size)
        val_end = train_end + fold_size
        if val_end > n_samples: break
            
        train_split = full_train.iloc[:train_end].copy()
        val_split = full_train.iloc[train_end:val_end].copy()
        
        print(f"Train: {len(train_split):,} | Val: {len(val_split):,}")
        
        # 3. Dynamic Feature Generation (Prevents Leakage)
        print("Generating interaction features...")
        feat_gen = FeatureGenerator(train_split, book_genres_df)
        
        # Apply to Train (Learn from itself)
        X_train_processed = feat_gen.process(train_split.copy())
        # Apply to Val (Learn from Train history)
        X_val_processed = feat_gen.process(val_split.copy())
        
        # 4. Feature Selection
        exclude = [
            constants.COL_SOURCE, config.TARGET, constants.COL_PREDICTION, 
            constants.COL_TIMESTAMP, constants.COL_USER_ID, constants.COL_BOOK_ID,
            constants.COL_HAS_READ, "book_id_list",
            constants.COL_TITLE, constants.COL_AUTHOR_NAME # Exclude string columns
        ]
        
        features = [c for c in X_train_processed.columns if c not in exclude]
        
        # Identify Categorical Features
        cat_feats = [c for c in features if c in config.CAT_FEATURES]
        
        # Save feature list on first fold
        if feature_names is None:
            feature_names = features
            with open(model_dir / "features_list.json", "w") as f:
                json.dump(features, f)

        # 5. Train LightGBM
        print(f"Training on {len(features)} features...")
        
        dtrain = lgb.Dataset(
            X_train_processed[features], 
            label=X_train_processed[config.TARGET],
            categorical_feature=cat_feats
        )
        dval = lgb.Dataset(
            X_val_processed[features], 
            label=X_val_processed[config.TARGET],
            categorical_feature=cat_feats,
            reference=dtrain
        )
        
        model = lgb.train(
            config.LGB_PARAMS,
            dtrain,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(100), 
                lgb.log_evaluation(50)
            ]
        )
        
        # Save Model
        model_name = config.MODEL_FILENAME_PATTERN.format(fold=fold)
        model.save_model(str(model_dir / model_name))
        
        # Cleanup to save memory
        del X_train_processed, X_val_processed, dtrain, dval, model
        gc.collect()

    print("\nTraining Complete.")

if __name__ == "__main__":
    train()