"""
Project-wide constants.
"""

# --- FILENAMES ---
TRAIN_FILENAME = "train.csv"
TARGETS_FILENAME = "targets.csv"
CANDIDATES_FILENAME = "candidates.csv"
USER_DATA_FILENAME = "users.csv"
BOOK_DATA_FILENAME = "books.csv"
BOOK_GENRES_FILENAME = "book_genres.csv"
GENRES_FILENAME = "genres.csv"
BOOK_DESCRIPTIONS_FILENAME = "book_descriptions.csv"
SUBMISSION_FILENAME = "submission.csv"
PROCESSED_DATA_FILENAME = "processed_features.parquet"

# --- COLUMN NAMES ---
COL_USER_ID = "user_id"
COL_BOOK_ID = "book_id"
COL_TARGET = "has_read"
COL_RELEVANCE = "relevance"  # 0, 1, 2
COL_SOURCE = "source"
COL_PREDICTION = "rating_predict"
COL_HAS_READ = "has_read"
COL_TIMESTAMP = "timestamp"
COL_BOOK_ID_LIST = "book_id_list"
COL_DESCRIPTION = "description"

# --- RAW METADATA COLUMNS ---
COL_TITLE = "title"           
COL_AUTHOR_NAME = "author_name"
COL_GENDER = "gender"
COL_AGE = "age"
COL_AUTHOR_ID = "author_id"
COL_PUBLICATION_YEAR = "publication_year"
COL_LANGUAGE = "language"
COL_PUBLISHER = "publisher"
COL_AVG_RATING = "avg_rating"
COL_GENRE_ID = "genre_id"

# --- VALUES ---
VAL_SOURCE_TRAIN = "train"
VAL_SOURCE_TEST = "test"
MISSING_CAT_VALUE = "-1"
MISSING_NUM_VALUE = -1
MAX_RANKING_LENGTH = 20