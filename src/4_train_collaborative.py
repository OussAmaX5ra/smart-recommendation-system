"""
Step 4: Train Collaborative Filtering Model
Uses Matrix Factorization (SVD) for collaborative filtering
Library: Surprise (scikit-surprise)
"""

import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
from collections import defaultdict
import config
from utils import print_section, save_pickle, load_pickle


def load_training_data():
    """Load training data"""
    print_section("Loading Training Data")

    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    print(f"✓ Loaded train data: {train_df.shape}")

    return train_df


def prepare_cf_data(df):
    """
    Prepare data for collaborative filtering
    Create implicit ratings based on purchase frequency
    """
    print_section("Preparing CF Data")

    # Create implicit ratings: count how many times each user bought each product
    cf_data = df.groupby(['user_id', 'product_id']).size().reset_index(name='rating')

    # Normalize ratings to 1-5 scale
    max_rating = cf_data['rating'].max()
    cf_data['rating'] = 1 + (cf_data['rating'] - 1) / (max_rating - 1) * 4

    print(f"CF Data shape: {cf_data.shape}")
    print(f"Rating statistics:")
    print(cf_data['rating'].describe())

    return cf_data


def train_svd_model(cf_data):
    """
    Train SVD model using Surprise library
    """
    print_section("Training SVD Model")

    # Define the format
    reader = Reader(rating_scale=(1, 5))

    # Load the data
    data = Dataset.load_from_df(
        cf_data[['user_id', 'product_id', 'rating']],
        reader
    )

    # Build full trainset
    trainset = data.build_full_trainset()

    print(f"Trainset statistics:")
    print(f"  Users: {trainset.n_users}")
    print(f"  Items: {trainset.n_items}")
    print(f"  Ratings: {trainset.n_ratings}")
    print(f"  Sparsity: {1 - (trainset.n_ratings / (trainset.n_users * trainset.n_items)):.4f}")

    # Initialize SVD model
    model = SVD(
        n_factors=config.CF_N_FACTORS,
        n_epochs=config.CF_N_EPOCHS,
        lr_all=config.CF_LR_ALL,
        reg_all=config.CF_REG_ALL,
        random_state=config.RANDOM_SEED,
        verbose=True
    )

    print("\nTraining SVD model...")
    model.fit(trainset)
    print("✓ Training completed")

    return model, trainset


def evaluate_model(model, cf_data):
    """
    Evaluate model using cross-validation
    """
    print_section("Evaluating CF Model")

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(
        cf_data[['user_id', 'product_id', 'rating']],
        reader
    )

    # Perform cross-validation
    print("Performing 5-fold cross-validation...")
    cv_results = cross_validate(
        model,
        data,
        measures=['RMSE', 'MAE'],
        cv=5,
        verbose=True
    )

    print(f"\nCross-validation results:")
    print(f"  RMSE: {cv_results['test_rmse'].mean():.4f} (+/- {cv_results['test_rmse'].std():.4f})")
    print(f"  MAE: {cv_results['test_mae'].mean():.4f} (+/- {cv_results['test_mae'].std():.4f})")

    return cv_results


def get_top_n_recommendations(model, trainset, n=20):
    """
    Generate top-N recommendations for all users
    Returns a dictionary: user_id -> list of (product_id, rating) tuples
    """
    print_section("Generating Recommendations")

    # Get all users and items
    all_users = [trainset.to_raw_uid(i) for i in range(trainset.n_users)]
    all_items = [trainset.to_raw_iid(i) for i in range(trainset.n_items)]

    # Dictionary to store recommendations
    top_n = defaultdict(list)

    print(f"Generating recommendations for {len(all_users)} users...")

    for i, user in enumerate(all_users):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(all_users)} users")

        # Get predictions for all items for this user
        predictions = []
        for item in all_items:
            pred = model.predict(user, item)
            predictions.append((item, pred.est))

        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Store top N
        top_n[user] = predictions[:n]

    print(f"✓ Generated recommendations for {len(top_n)} users")

    return dict(top_n)


def save_cf_model(model, trainset, top_n_recs):
    """
    Save trained CF model and recommendations
    """
    print_section("Saving CF Model")

    cf_artifacts = {
        'model': model,
        'trainset': trainset,
        'top_n_recommendations': top_n_recs
    }

    save_pickle(cf_artifacts, config.CF_MODEL_PATH)
    print(f"✓ Saved CF model to: {config.CF_MODEL_PATH}")


def main():
    """Main CF training pipeline"""
    print("\n" + "=" * 60)
    print("  STEP 4: TRAIN COLLABORATIVE FILTERING MODEL")
    print("=" * 60)

    # Load data
    train_df = load_training_data()

    # Prepare CF data
    cf_data = prepare_cf_data(train_df)

    # Train model
    model, trainset = train_svd_model(cf_data)

    # Evaluate model
    cv_results = evaluate_model(model, cf_data)

    # Generate recommendations
    top_n_recs = get_top_n_recommendations(model, trainset, n=config.RECOMMENDATION_CANDIDATES)

    # Save model
    save_cf_model(model, trainset, top_n_recs)

    print("\n" + "=" * 60)
    print("  STEP 4 COMPLETED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()