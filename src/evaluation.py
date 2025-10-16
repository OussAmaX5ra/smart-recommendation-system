"""
Step 7: Evaluation
Evaluate the hybrid recommendation system using MAP@20
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import config
from utils import print_section, load_pickle, save_json, calculate_map_at_k


def load_validation_data():
    """Load validation data"""
    print_section("Loading Validation Data")

    val_df = pd.read_csv(config.VAL_DATA_PATH)
    print(f"✓ Validation data: {val_df.shape}")
    print(f"  Users: {val_df['user_id'].nunique()}")
    print(f"  Products: {val_df['product_id'].nunique()}")

    return val_df


def load_recommendations():
    """Load generated recommendations"""
    print_section("Loading Recommendations")

    recommendations_pkl_path = config.RECOMMENDATIONS_PATH.replace('.csv', '.pkl')
    recommendations = load_pickle(recommendations_pkl_path)

    print(f"✓ Loaded recommendations for {len(recommendations)} users")

    return recommendations


def get_ground_truth(val_df):
    """
    Extract ground truth: products actually purchased by users in validation set
    """
    print_section("Extracting Ground Truth")

    ground_truth = defaultdict(list)

    for user_id, group in val_df.groupby('user_id'):
        # Get products purchased in validation set
        products = group['product_id'].unique().tolist()
        ground_truth[user_id] = products

    print(f"Ground truth for {len(ground_truth)} users")

    # Statistics
    num_products_per_user = [len(products) for products in ground_truth.values()]
    print(f"  Average products per user: {np.mean(num_products_per_user):.2f}")
    print(f"  Min products per user: {np.min(num_products_per_user)}")
    print(f"  Max products per user: {np.max(num_products_per_user)}")

    return dict(ground_truth)


def calculate_precision_at_k(actual, predicted, k=20):
    """Calculate Precision@K"""
    if not actual or not predicted:
        return 0.0

    predicted_k = predicted[:k]
    num_relevant = len(set(predicted_k) & set(actual))

    return num_relevant / k


def calculate_recall_at_k(actual, predicted, k=20):
    """Calculate Recall@K"""
    if not actual or not predicted:
        return 0.0

    predicted_k = predicted[:k]
    num_relevant = len(set(predicted_k) & set(actual))

    return num_relevant / len(actual)


def calculate_ndcg_at_k(actual, predicted, k=20):
    """Calculate Normalized Discounted Cumulative Gain@K"""
    if not actual or not predicted:
        return 0.0

    predicted_k = predicted[:k]

    # DCG
    dcg = 0.0
    for i, item in enumerate(predicted_k):
        if item in actual:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because positions start at 1, and log2(1)=0

    # IDCG (ideal DCG)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_hit_rate_at_k(actual, predicted, k=20):
    """Calculate Hit Rate@K (whether at least one relevant item is in top K)"""
    if not actual or not predicted:
        return 0.0

    predicted_k = predicted[:k]
    num_hits = len(set(predicted_k) & set(actual))

    return 1.0 if num_hits > 0 else 0.0


def evaluate_recommendations(recommendations, ground_truth, k=20):
    """
    Evaluate recommendations using multiple metrics
    """
    print_section(f"Evaluating Recommendations (K={k})")

    # Align users
    common_users = set(recommendations.keys()) & set(ground_truth.keys())
    print(f"Evaluating on {len(common_users)} common users")

    if len(common_users) == 0:
        print("⚠ Warning: No common users between recommendations and ground truth")
        return {}

    # Prepare data for MAP calculation
    actual_list = []
    predicted_list = []

    # Calculate per-user metrics
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    hit_rates = []

    for user_id in common_users:
        actual = ground_truth[user_id]
        predicted = recommendations[user_id]

        if not predicted:  # Skip users with no recommendations
            continue

        actual_list.append(actual)
        predicted_list.append(predicted)

        # Calculate metrics
        precision_scores.append(calculate_precision_at_k(actual, predicted, k))
        recall_scores.append(calculate_recall_at_k(actual, predicted, k))
        ndcg_scores.append(calculate_ndcg_at_k(actual, predicted, k))
        hit_rates.append(calculate_hit_rate_at_k(actual, predicted, k))

    # Calculate MAP@K
    map_score = calculate_map_at_k(actual_list, predicted_list, k)

    # Calculate average metrics
    metrics = {
        f'MAP@{k}': map_score,
        f'Precision@{k}': np.mean(precision_scores),
        f'Recall@{k}': np.mean(recall_scores),
        f'NDCG@{k}': np.mean(ndcg_scores),
        f'Hit_Rate@{k}': np.mean(hit_rates),
        'num_users_evaluated': len(common_users)
    }

    # Print results
    print("\nEvaluation Results:")
    print("=" * 40)
    for metric_name, value in metrics.items():
        if metric_name != 'num_users_evaluated':
            print(f"  {metric_name}: {value:.4f}")
        else:
            print(f"  {metric_name}: {value}")
    print("=" * 40)

    return metrics


def analyze_recommendation_coverage(recommendations, all_products):
    """Analyze how many unique products are being recommended (catalog coverage)"""
    print_section("Analyzing Recommendation Coverage")

    # Get all recommended products
    all_recommended = set()
    for products in recommendations.values():
        all_recommended.update(products)

    # Calculate coverage
    coverage = len(all_recommended) / len(all_products)

    print(f"Catalog Coverage:")
    print(f"  Total products in catalog: {len(all_products)}")
    print(f"  Unique products recommended: {len(all_recommended)}")
    print(f"  Coverage: {coverage:.2%}")

    # Product popularity distribution
    product_counts = defaultdict(int)
    for products in recommendations.values():
        for product in products:
            product_counts[product] += 1

    counts = list(product_counts.values())
    print(f"\nProduct Recommendation Distribution:")
    print(f"  Mean recommendations per product: {np.mean(counts):.2f}")
    print(f"  Median recommendations per product: {np.median(counts):.2f}")
    print(f"  Max recommendations for a product: {np.max(counts)}")

    return coverage, len(all_recommended)


def analyze_user_coverage(recommendations, all_users):
    """Analyze recommendation coverage for users"""
    print_section("Analyzing User Coverage")

    users_with_recs = len(recommendations)
    coverage = users_with_recs / len(all_users)

    print(f"User Coverage:")
    print(f"  Total users: {len(all_users)}")
    print(f"  Users with recommendations: {users_with_recs}")
    print(f"  Coverage: {coverage:.2%}")

    # Analyze recommendation list lengths
    rec_lengths = [len(recs) for recs in recommendations.values()]
    print(f"\nRecommendation List Lengths:")
    print(f"  Mean: {np.mean(rec_lengths):.2f}")
    print(f"  Median: {np.median(rec_lengths):.2f}")
    print(f"  Min: {np.min(rec_lengths)}")
    print(f"  Max: {np.max(rec_lengths)}")

    return coverage


def perform_error_analysis(recommendations, ground_truth):
    """Perform error analysis on recommendations"""
    print_section("Error Analysis")

    # Users with no recommendations
    all_users = set(ground_truth.keys())
    users_with_recs = set(recommendations.keys())
    users_without_recs = all_users - users_with_recs

    print(f"Users without recommendations: {len(users_without_recs)}")

    # Users with empty recommendation lists
    empty_recs = [user for user, recs in recommendations.items() if len(recs) == 0]
    print(f"Users with empty recommendation lists: {len(empty_recs)}")

    # Users with perfect matches (all actual products in top K)
    perfect_matches = 0
    partial_matches = 0
    no_matches = 0

    for user_id in users_with_recs:
        if user_id not in ground_truth:
            continue

        actual = set(ground_truth[user_id])
        predicted = set(recommendations[user_id][:config.TOP_K])

        overlap = len(actual & predicted)

        if overlap == len(actual):
            perfect_matches += 1
        elif overlap > 0:
            partial_matches += 1
        else:
            no_matches += 1

    total = perfect_matches + partial_matches + no_matches

    print(f"\nMatch Analysis:")
    print(f"  Perfect matches: {perfect_matches} ({perfect_matches / total * 100:.1f}%)")
    print(f"  Partial matches: {partial_matches} ({partial_matches / total * 100:.1f}%)")
    print(f"  No matches: {no_matches} ({no_matches / total * 100:.1f}%)")


def save_evaluation_results(metrics, coverage_stats):
    """Save evaluation results to JSON"""
    print_section("Saving Evaluation Results")

    results = {
        'metrics': metrics,
        'coverage': coverage_stats,
        'configuration': {
            'TOP_K': config.TOP_K,
            'CF_WEIGHT': config.CF_WEIGHT,
            'DL_WEIGHT': config.DL_WEIGHT,
            'CF_N_FACTORS': config.CF_N_FACTORS,
            'DL_EMBEDDING_DIM': config.DL_EMBEDDING_DIM
        }
    }

    save_json(results, config.EVALUATION_METRICS_PATH)
    print(f"✓ Saved evaluation results to: {config.EVALUATION_METRICS_PATH}")


def main():
    """Main evaluation pipeline"""
    print("\n" + "=" * 60)
    print("  STEP 7: EVALUATION (MAP@20)")
    print("=" * 60)

    # Load data
    val_df = load_validation_data()
    recommendations = load_recommendations()

    # Get ground truth
    ground_truth = get_ground_truth(val_df)

    # Evaluate recommendations
    metrics = evaluate_recommendations(
        recommendations,
        ground_truth,
        k=config.TOP_K
    )

    # Analyze coverage
    all_products = val_df['product_id'].unique()
    catalog_coverage, num_unique_products = analyze_recommendation_coverage(
        recommendations,
        all_products
    )

    all_users = val_df['user_id'].unique()
    user_coverage = analyze_user_coverage(recommendations, all_users)

    # Error analysis
    perform_error_analysis(recommendations, ground_truth)

    # Prepare coverage stats
    coverage_stats = {
        'catalog_coverage': catalog_coverage,
        'unique_products_recommended': num_unique_products,
        'total_products': len(all_products),
        'user_coverage': user_coverage,
        'total_users': len(all_users)
    }

    # Save results
    save_evaluation_results(metrics, coverage_stats)

    # Print summary
    print_section("EVALUATION SUMMARY")
    print(f"\n🎯 Main Metric - MAP@{config.TOP_K}: {metrics[f'MAP@{config.TOP_K}']:.4f}")
    print(f"📊 Precision@{config.TOP_K}: {metrics[f'Precision@{config.TOP_K}']:.4f}")
    print(f"📊 Recall@{config.TOP_K}: {metrics[f'Recall@{config.TOP_K}']:.4f}")
    print(f"📊 NDCG@{config.TOP_K}: {metrics[f'NDCG@{config.TOP_K}']:.4f}")
    print(f"📊 Hit Rate@{config.TOP_K}: {metrics[f'Hit_Rate@{config.TOP_K}']:.4f}")
    print(f"\n📦 Catalog Coverage: {catalog_coverage:.2%}")
    print(f"👥 User Coverage: {user_coverage:.2%}")

    print("\n" + "=" * 60)
    print("  STEP 7 COMPLETED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()