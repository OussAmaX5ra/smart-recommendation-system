"""
Step 8: Inference
Generate recommendations for test users
Input: user_test.csv (contains user_ids)
Output: recommendations.csv (user_id, product_id)
"""

import pandas as pd
import numpy as np
from tensorflow import keras
import config
from utils import print_section, load_pickle, get_user_purchased_products


def load_test_users():
    """Load test user IDs"""
    print_section("Loading Test Users")

    test_users_df = pd.read_csv(config.USER_TEST_PATH)

    # Assuming the CSV has a column named 'user_id'
    if 'user_id' in test_users_df.columns:
        test_users = test_users_df['user_id'].unique()
    else:
        # If no column name, assume first column is user_id
        test_users = test_users_df.iloc[:, 0].unique()

    print(f"✓ Loaded {len(test_users)} test users")

    return test_users


def load_models_and_data():
    """Load all necessary models and data"""
    print_section("Loading Models and Data")

    # Load CF model
    cf_artifacts = load_pickle(config.CF_MODEL_PATH)
    cf_model = cf_artifacts['model']
    cf_trainset = cf_artifacts['trainset']
    print(f"✓ Loaded CF model")

    # Load DL model
    dl_model = keras.models.load_model(config.DL_MODEL_PATH)
    print(f"✓ Loaded DL model")

    # Load DL predictions (if available)
    try:
        dl_predictions_path = config.DL_MODEL_PATH.replace('.h5', '_predictions.pkl')
        dl_predictions = load_pickle(dl_predictions_path)
        print(f"✓ Loaded DL predictions")
    except:
        dl_predictions = None
        print(f"⚠ DL predictions not found, will generate on-the-fly")

    # Load encoders
    user_encoder = load_pickle(config.USER_ENCODER_PATH)
    product_encoder = load_pickle(config.PRODUCT_ENCODER_PATH)
    print(f"✓ Loaded encoders")

    # Load training data (to get user purchase history and all products)
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    print(f"✓ Loaded training data: {train_df.shape}")

    return cf_model, cf_trainset, dl_model, dl_predictions, user_encoder, product_encoder, train_df


def normalize_scores(scores):
    """Normalize scores to [0, 1] range"""
    if len(scores) == 0:
        return scores

    values = list(scores.values())
    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        return {k: 0.5 for k in scores.keys()}

    normalized = {
        k: (v - min_val) / (max_val - min_val)
        for k, v in scores.items()
    }

    return normalized


def get_cf_scores(cf_model, user_id, all_products):
    """Get CF scores for all products for a user"""
    scores = {}

    for product_id in all_products:
        try:
            pred = cf_model.predict(user_id, product_id)
            scores[product_id] = pred.est
        except:
            scores[product_id] = 0.0

    return scores


def get_dl_scores(dl_model, user_encoded, product_encoder, all_products_encoded):
    """Get DL scores for all products for a user"""
    # Create input arrays
    user_array = np.full(len(all_products_encoded), user_encoded)
    product_array = np.array(all_products_encoded)

    # Get predictions
    predictions = dl_model.predict(
        [user_array, product_array],
        batch_size=1024,
        verbose=0
    ).flatten()

    # Create dictionary mapping product_id to score
    scores = {}
    for product_enc, score in zip(all_products_encoded, predictions):
        try:
            product_id = product_encoder.inverse_transform([int(product_enc)])[0]
            scores[product_id] = float(score)
        except:
            continue

    return scores


def generate_recommendations_for_user(
        user_id,
        cf_model,
        dl_model,
        user_encoder,
        product_encoder,
        all_products,
        all_products_encoded,
        purchased_products,
        cf_weight=0.5,
        dl_weight=0.5,
        top_k=20
):
    """Generate recommendations for a single user"""

    # Check if user is known
    try:
        user_encoded = user_encoder.transform([user_id])[0]
        user_known = True
    except:
        user_known = False
        user_encoded = None

    # Get CF scores
    cf_scores = get_cf_scores(cf_model, user_id, all_products)
    cf_scores = normalize_scores(cf_scores)

    # Get DL scores
    if user_known and dl_model is not None:
        dl_scores = get_dl_scores(dl_model, user_encoded, product_encoder, all_products_encoded)
        dl_scores = normalize_scores(dl_scores)
    else:
        dl_scores = {p: 0.0 for p in all_products}

    # Combine scores
    combined_scores = {}
    for product_id in all_products:
        cf_score = cf_scores.get(product_id, 0.0)
        dl_score = dl_scores.get(product_id, 0.0)

        combined_scores[product_id] = cf_weight * cf_score + dl_weight * dl_score

    # Filter out purchased products
    filtered_scores = {
        product_id: score
        for product_id, score in combined_scores.items()
        if product_id not in purchased_products
    }

    # Sort by score and get top K
    sorted_products = sorted(
        filtered_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_products = [product_id for product_id, _ in sorted_products[:top_k]]

    return top_products


def generate_recommendations_for_all_users(
        test_users,
        cf_model,
        cf_trainset,
        dl_model,
        dl_predictions,
        user_encoder,
        product_encoder,
        train_df
):
    """Generate recommendations for all test users"""
    print_section("Generating Recommendations for Test Users")

    # Get all products
    all_products = train_df['product_id'].unique()

    # Get encoded products (for DL model)
    all_products_encoded = []
    for product_id in all_products:
        try:
            product_enc = product_encoder.transform([product_id])[0]
            all_products_encoded.append(product_enc)
        except:
            continue

    # Get user purchase history from training data
    print("Building user purchase history...")
    user_purchased = {}
    for user_id in test_users:
        user_purchased[user_id] = get_user_purchased_products(train_df, user_id)

    # Generate recommendations
    print(f"Generating recommendations for {len(test_users)} users...")

    recommendations = {}

    for idx, user_id in enumerate(test_users):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(test_users)} users")

        try:
            recs = generate_recommendations_for_user(
                user_id=user_id,
                cf_model=cf_model,
                dl_model=dl_model,
                user_encoder=user_encoder,
                product_encoder=product_encoder,
                all_products=all_products,
                all_products_encoded=all_products_encoded,
                purchased_products=user_purchased[user_id],
                cf_weight=config.CF_WEIGHT,
                dl_weight=config.DL_WEIGHT,
                top_k=config.TOP_K
            )

            recommendations[user_id] = recs

        except Exception as e:
            print(f"  Warning: Error for user {user_id}: {e}")
            # Use most popular products as fallback
            popular_products = train_df['product_id'].value_counts().head(config.TOP_K).index.tolist()
            # Filter out purchased products
            fallback_recs = [p for p in popular_products if p not in user_purchased[user_id]][:config.TOP_K]
            recommendations[user_id] = fallback_recs

    print(f"✓ Generated recommendations for {len(recommendations)} users")

    return recommendations


def save_recommendations_csv(recommendations):
    """
    Save recommendations in the required format: user_id, product_id
    Each user will have TOP_K rows (one for each recommended product)
    """
    print_section("Saving Recommendations to CSV")

    # Create list of (user_id, product_id) pairs
    recommendation_pairs = []

    for user_id, products in recommendations.items():
        for product_id in products:
            recommendation_pairs.append({
                'user_id': user_id,
                'product_id': product_id
            })

    # Create DataFrame
    recommendations_df = pd.DataFrame(recommendation_pairs)

    # Save to CSV
    recommendations_df.to_csv(config.RECOMMENDATIONS_PATH, index=False)

    print(f"✓ Saved recommendations to: {config.RECOMMENDATIONS_PATH}")
    print(f"  Total rows: {len(recommendations_df)}")
    print(f"  Unique users: {recommendations_df['user_id'].nunique()}")
    print(f"  Unique products: {recommendations_df['product_id'].nunique()}")

    # Display sample
    print("\nSample recommendations:")
    print(recommendations_df.head(20))

    return recommendations_df


def generate_statistics(recommendations_df, test_users):
    """Generate statistics about the recommendations"""
    print_section("Recommendation Statistics")

    # Recommendations per user
    recs_per_user = recommendations_df.groupby('user_id').size()

    print(f"Recommendations per user:")
    print(f"  Mean: {recs_per_user.mean():.2f}")
    print(f"  Median: {recs_per_user.median():.2f}")
    print(f"  Min: {recs_per_user.min()}")
    print(f"  Max: {recs_per_user.max()}")

    # Coverage
    users_with_recs = recommendations_df['user_id'].nunique()
    coverage = users_with_recs / len(test_users)

    print(f"\nCoverage:")
    print(f"  Test users: {len(test_users)}")
    print(f"  Users with recommendations: {users_with_recs}")
    print(f"  Coverage: {coverage:.2%}")

    # Product diversity
    unique_products = recommendations_df['product_id'].nunique()
    print(f"\nProduct diversity:")
    print(f"  Unique products recommended: {unique_products}")

    # Most frequently recommended products
    print(f"\nTop 10 most frequently recommended products:")
    top_products = recommendations_df['product_id'].value_counts().head(10)
    for product_id, count in top_products.items():
        print(f"  Product {product_id}: {count} times")


def main():
    """Main inference pipeline"""
    print("\n" + "=" * 60)
    print("  STEP 8: INFERENCE - GENERATE RECOMMENDATIONS")
    print("=" * 60)

    # Load test users
    test_users = load_test_users()

    # Load models and data
    cf_model, cf_trainset, dl_model, dl_predictions, user_encoder, product_encoder, train_df = load_models_and_data()

    # Generate recommendations
    recommendations = generate_recommendations_for_all_users(
        test_users=test_users,
        cf_model=cf_model,
        cf_trainset=cf_trainset,
        dl_model=dl_model,
        dl_predictions=dl_predictions,
        user_encoder=user_encoder,
        product_encoder=product_encoder,
        train_df=train_df
    )

    # Save recommendations to CSV
    recommendations_df = save_recommendations_csv(recommendations)

    # Generate statistics
    generate_statistics(recommendations_df, test_users)

    print("\n" + "=" * 60)
    print("  STEP 8 COMPLETED SUCCESSFULLY!")
    print("  Output saved to:", config.RECOMMENDATIONS_PATH)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()