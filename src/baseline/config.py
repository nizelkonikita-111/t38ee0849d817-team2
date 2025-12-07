"""
Configuration file for the NTO ML competition (Stage 2).
"""

from pathlib import Path
from . import constants

# --- DIRECTORIES ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

# --- PARAMETERS ---
N_FOLDS = 3
RANDOM_STATE = 42
TARGET = constants.COL_RELEVANCE

# --- TRAINING CONFIG ---
MODEL_FILENAME_PATTERN = "lgb_fold_{fold}.txt" # FIX: Re-added this constant

# --- TEXT FEATURES ---
TFIDF_MAX_FEATURES = 1000  
TFIDF_SVD_COMPONENTS = 32  
USE_BERT = False           

# --- FEATURE ENGINEERING ---
TE_ALPHA = 10 

# --- MODEL PARAMETERS ---
LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "n_estimators": 5000,
    "learning_rate": 0.015,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l1": 2.0,
    "lambda_l2": 5.0,
    "num_leaves": 63,
    "max_depth": 12,
    "min_child_samples": 50,
    "verbose": -1,
    "n_jobs": -1,
    "seed": RANDOM_STATE,
    "class_weight": "balanced",
    "force_row_wise": True
}

LGB_FIT_PARAMS = {
    "eval_metric": ["multi_logloss", "multi_error"],
    "callbacks": [],
}

CAT_FEATURES = [
    constants.COL_GENDER,
    constants.COL_LANGUAGE,
    constants.COL_PUBLISHER,
]