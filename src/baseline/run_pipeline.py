"""
Main entry point to run the full end-to-end pipeline.
Sequentially executes data preparation, training, prediction, and validation.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.baseline import config
from src.baseline.prepare_data import prepare_data
from src.baseline.train import train
from src.baseline.predict import predict
from src.baseline.validate import validate

def run_pipeline():
    total_start_time = time.time()
    
    print("="*80)
    print("🚀 STARTING FULL PIPELINE EXECUTION")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Model: LightGBM (Multiclass + Ranking)")
    print(f"  - Folds: {config.N_FOLDS}")
    print(f"  - Text Features: TF-IDF + SVD (Components: {config.TFIDF_SVD_COMPONENTS})")
    print(f"  - BERT Enabled: {getattr(config, 'USE_BERT', False)}")
    print("="*80)

    # --- STEP 1: Data Preparation ---
    print("\n[STEP 1/4] Running Data Preparation...")
    step_start = time.time()
    try:
        prepare_data()
        print(f"✅ Data Preparation completed in {time.time() - step_start:.2f}s")
    except Exception as e:
        print(f"❌ Data Preparation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- STEP 2: Model Training ---
    print("\n[STEP 2/4] Running Model Training (CV)...")
    step_start = time.time()
    try:
        train()
        print(f"✅ Training completed in {time.time() - step_start:.2f}s")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- STEP 3: Inference ---
    print("\n[STEP 3/4] Running Inference and Submission Generation...")
    step_start = time.time()
    try:
        predict()
        print(f"✅ Prediction completed in {time.time() - step_start:.2f}s")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- STEP 4: Validation ---
    print("\n[STEP 4/4] Validating Submission Format...")
    step_start = time.time()
    try:
        validate()
        print(f"✅ Validation completed in {time.time() - step_start:.2f}s")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)

    total_duration = time.time() - total_start_time
    print("\n" + "="*80)
    print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"⏱️  Total execution time: {total_duration / 60:.1f} minutes")
    print(f"📂 Submission file: {config.SUBMISSION_DIR / 'submission.csv'}")
    print("="*80)

if __name__ == "__main__":
    run_pipeline()