from . import config, constants
from .data_processing import load_and_merge_data
from .features import generate_text_features

def prepare_data():
    merged_df, _, _, _, descriptions_df = load_and_merge_data()
    
    # Generate STATIC features (Text SVD) here. 
    # Dynamic features (Aggregates) are done in train/predict.
    merged_df = generate_text_features(merged_df, descriptions_df)
    
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(
        config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME,
        index=False
    )
    print("Data preparation complete.")

if __name__ == "__main__":
    prepare_data()