"""
Step 6: Hybrid Recommender
Combines Collaborative Filtering and Deep Learning predictions
to generate final recommendations
"""

import pandas as pd
import numpy as np
from tensorflow import keras
import config
from utils import print_section, load_pickle, save_pickle, get_user_purchased_products


def load_models():
    """Load trained CF and DL models"""
    print_section("Loading Models")

    # Load CF model
    cf_artifacts = load_pickle(config.CF_MODEL_PATH)
    cf_model = cf_artifacts['model']
    cf_trainset = cf_artifacts['trainset']

    print(f"✓ Loaded CF model")

    # Load DL model
    dl_model = keras.models.load_model(config.DL_MODEL_PATH)
    print(f"✓ Loaded DL model")

    # Load DL predictions
    dl_predictions_path = config.DL_MODEL_PATH.replace('.h5', '_predictions.pkl')
    dl_predictions = load_pickle(dl_predictions_path)
    print(f"✓ Loaded DL predictions")

    # Load encoders
    user_encoder = load_pickle(config.USER_ENCODER_PATH)
    product_encoder = load_pickle(config.PRODUCT_ENCODER_PATH)
    print(f"✓ Loaded encoders")

    return cf_model, cf_trainset, dl_model, dl_predictions, user_encoder, product_encoder


def load_data():
    """Load train data to get user purchase history"""
    print_section("Loading Data")

    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(config.VAL_DATA_PATH)

    print(f"✓ Train data: {train_df.shape}")
    print(f"✓ Validation data: {val_df.shape}")

    return train_df, val_df


def get_cf_recommendations(cf_model, user_id, all_products, n=200):
    """
    Get CF recommendations for a user
    Returns list of (product_id, score) tuples
    """
    predictions = []

    for product_id in all_products:
        try:
            pred = cf_model.predict(user_id, product_id)
            predictions.append((product_id, pred.est))
        except:
            # Product not in training set
            predictions.append((product_id, 0.0))

    # Sort by score
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions[:n]


def get_dl_recommendations(dl_predictions, user_encoded, product_encoder, n=200):
    """
    Get DL recommendations for a user
    Returns list of (product_id, score) tuples
    """
    if user_encoded not in dl_predictions:
        return []

    # Get predictions for this user
    user_preds = dl_predictions[user_encoded][:n]

    # Convert encoded product IDs back to original IDs
    recommendations = []
    for product_enc, score in user_preds:
        try:
            product_id = product_encoder.inverse_transform([int(product_enc)])[0]
            recommendations.append((product_id, score))
        except:
            continue

    return recommendations


def normalize_scores(recommendations):
    """Normalize recommendation scores to [0, 1] range"""
    if not recommendations:
        return recommendations

    scores = [score for _, score in recommendations]
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [(pid, 0.5) for pid, _ in recommendations]

    normalized = [
        (pid, (score - min_score) / (max_score - min_score))
        for pid, score in recommendations
    ]

    return normalized


def combine_recommendations(cf_recs, dl_recs, cf_weight=0.5, dl_weight=0.5):
    """
    Combine CF and DL recommendations using weighted average
    """
    # Normalize scores
    cf_recs_norm = normalize_scores(cf_recs)
    dl_recs_norm = normalize_scores(dl_recs)

    # Create dictionaries for easy lookup
    cf_dict = dict(cf_recs_norm)
    dl_dict = dict(dl_recs_norm)

    # Get all products
    all_products = set(cf_dict.keys()) | set(dl_dict.keys())

    # Combine scores
    combined = {}
    for product_id in all_products:
        cf_score = cf_dict.get(product_id, 0.0)
        dl_score = dl_dict.get(product_id, 0.0)

        # Weighted average
        combined_score = cf_weight * cf_score + dl_weight * dl_score
        combined[product_id] = combined_score

    # Sort by combined score
    recommendations = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    return recommendations


def filter_purchased_products(recommendations, purchased_products):
    """Remove products that user has already purchased"""
    filtered = [
        (product_id, score)
        for product_id, score in recommendations
        if product_id not in purchased_products
    ]
    return filtered


def generate_hybrid_recommendations(
        cf_model,
        cf_trainset,
        dl_predictions,
        user_encoder,
        product_encoder,
        train_df,
        users,
        top_k=20
):
    """
    Generate hybrid recommendations for a list of users
    """
    print_section("Generating Hybrid Recommendations")

    # Get all products
    all_products = train_df['product_id'].unique()

    # Get user purchase history
    user_purchased = {}
    for user_id in users:
        user_purchased[user_id] = get_user_purchased_products(train_df, user_id)

    recommendations = {}

    print(f"Generating recommendations for {len(users)} users...")

    for idx, user_id in enumerate(users):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(users)} users")

        try:
            # Get user encoded ID
            user_encoded = user_encoder.transform([user_id])[0]

            # Get CF recommendations
            cf_recs = get_cf_recommendations(cf_model, user_id, all_products, n=config.RECOMMENDATION_CANDIDATES)

            # Get DL recommendations
            dl_recs = get_dl_recommendations(dl_predictions, user_encoded, product_encoder,
                                             n=config.RECOMMENDATION_CANDIDATES)

            # Combine recommendations
            combined_recs = combine_recommendations(
                cf_recs,
                dl_recs,
                cf_weight=config.CF_WEIGHT,
                dl_weight=config.DL_WEIGHT
            )

            # Filter out already purchased products
            filtered_recs = filter_purchased_products(combined_recs, user_purchased[user_id])

            # Take top K
            top_recs = [product_id for product_id, _ in filtered_recs[:top_k]]

            recommendations[user_id] = top_recs

        except Exception as e:
            print(f"  Warning: Error for user {user_id}: {e}")
            recommendations[user_id] = []

    print(f"✓ Generated recommendations for {len(recommendations)} users")

    return recommendations


def save_recommendations(recommendations):
    """Save recommendations"""
    print_section("Saving Recommendations")

    # Save as pickle for later use
    recommendations_pkl_path = config.RECOMMENDATIONS_PATH.replace('.csv', '.pkl')
    save_pickle(recommendations, recommendations_pkl_path)
    print(f"✓ Saved recommendations (pickle): {recommendations_pkl_path}")

    # Also save in a readable format
    recs_list = []
    for user_id, products in recommendations.items():
        for rank, product_id in enumerate(products, 1):
            recs_list.append({
                'user_id': user_id,
                'product_id': product_id,
                'rank': rank
            })

    recs_df = pd.DataFrame(recs_list)
    recs_df.to_csv(config.RECOMMENDATIONS_PATH, index=False)
    print(f"✓ Saved recommendations (CSV): {config.RECOMMENDATIONS_PATH}")
    print(f"  Total recommendations: {len(recs_df)}")


def main():
    """Main hybrid recommendation pipeline"""
    print("\n" + "=" * 60)
    print("  STEP 6: HYBRID RECOMMENDER")
    print("=" * 60)

    # Load models
    cf_model, cf_trainset, dl_model, dl_predictions, user_encoder, product_encoder = load_models()

    # Load data
    train_df, val_df = load_data()

    # Get users to generate recommendations for (validation users)
    validation_users = val_df['user_id'].unique()

    print(f"\nGenerating recommendations for {len(validation_users)} validation users")

    # Generate hybrid recommendations
    recommendations = generate_hybrid_recommendations(
        cf_model=cf_model,
        cf_trainset=cf_trainset,
        dl_predictions=dl_predictions,
        user_encoder=user_encoder,
        product_encoder=product_encoder,
        train_df=train_df,
        users=validation_users,
        top_k=config.TOP_K
    )

    # Save recommendations
    save_recommendations(recommendations)

    # Print sample recommendations
    print_section("Sample Recommendations")
    sample_users = list(recommendations.keys())[:5]
    for user_id in sample_users:
        recs = recommendations[user_id]
        print(f"\nUser {user_id}:")
        print(f"  Top 5 products: {recs[:5]}")

    print("\n" + "=" * 60)
    print("  STEP 6 COMPLETED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()