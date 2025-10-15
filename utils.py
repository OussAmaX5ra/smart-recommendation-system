"""
Utility functions for the recommendation system.
Contains helper functions for data loading, visualization, and analysis.
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import config


# ==================== DATA LOADING UTILITIES ====================

def load_pickle(filepath):
    """Load data from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, filepath):
    """Save data to pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_all_features():
    """Load all engineered features."""
    features = {}
    files = [
        'mappings', 'user_item_matrix', 'sequences',
        'demographic_features', 'product_features', 'test_ground_truth'
    ]

    for name in files:
        path = f"{config.PROCESSED_DATA_DIR}/{name}.pkl"
        try:
            features[name] = load_pickle(path)
        except FileNotFoundError:
            print(f"Warning: {path} not found")

    return features


# ==================== DATA ANALYSIS UTILITIES ====================

def analyze_sparsity(user_item_matrix):
    """
    Analyze sparsity of user-item interaction matrix.

    Args:
        user_item_matrix: Sparse matrix of interactions

    Returns:
        Dictionary with sparsity statistics
    """
    n_users, n_items = user_item_matrix.shape
    n_interactions = user_item_matrix.nnz

    sparsity = 1 - (n_interactions / (n_users * n_items))

    # Interactions per user
    interactions_per_user = np.array(user_item_matrix.sum(axis=1)).flatten()

    # Interactions per item
    interactions_per_item = np.array(user_item_matrix.sum(axis=0)).flatten()

    stats = {
        'n_users': n_users,
        'n_items': n_items,
        'n_interactions': n_interactions,
        'sparsity': sparsity,
        'avg_interactions_per_user': interactions_per_user.mean(),
        'median_interactions_per_user': np.median(interactions_per_user),
        'avg_interactions_per_item': interactions_per_item.mean(),
        'median_interactions_per_item': np.median(interactions_per_item),
        'min_interactions_per_user': interactions_per_user.min(),
        'max_interactions_per_user': interactions_per_user.max(),
        'min_interactions_per_item': interactions_per_item.min(),
        'max_interactions_per_item': interactions_per_item.max()
    }

    return stats


def print_sparsity_stats(stats):
    """Print sparsity statistics in a formatted way."""
    print("\n" + "=" * 60)
    print("SPARSITY ANALYSIS")
    print("=" * 60)
    print(f"Users: {stats['n_users']:,}")
    print(f"Items: {stats['n_items']:,}")
    print(f"Interactions: {stats['n_interactions']:,}")
    print(f"Sparsity: {stats['sparsity']:.4%}")
    print(f"\nInteractions per User:")
    print(f"  Average: {stats['avg_interactions_per_user']:.2f}")
    print(f"  Median: {stats['median_interactions_per_user']:.2f}")
    print(f"  Min: {stats['min_interactions_per_user']}")
    print(f"  Max: {stats['max_interactions_per_user']}")
    print(f"\nInteractions per Item:")
    print(f"  Average: {stats['avg_interactions_per_item']:.2f}")
    print(f"  Median: {stats['median_interactions_per_item']:.2f}")
    print(f"  Min: {stats['min_interactions_per_item']}")
    print(f"  Max: {stats['max_interactions_per_item']}")


def analyze_sequences(sequences_dict):
    """
    Analyze user purchase sequences.

    Args:
        sequences_dict: Dictionary of user sequences

    Returns:
        Dictionary with sequence statistics
    """
    lengths = [len(seq) for seq in sequences_dict.values()]

    stats = {
        'n_users_with_sequences': len(sequences_dict),
        'avg_sequence_length': np.mean(lengths),
        'median_sequence_length': np.median(lengths),
        'min_sequence_length': np.min(lengths),
        'max_sequence_length': np.max(lengths),
        'std_sequence_length': np.std(lengths)
    }

    return stats


# ==================== VISUALIZATION UTILITIES ====================

def plot_interaction_distribution(user_item_matrix, save_path=None):
    """
    Plot distribution of interactions per user and per item.

    Args:
        user_item_matrix: Sparse interaction matrix
        save_path: Path to save figure (optional)
    """
    interactions_per_user = np.array(user_item_matrix.sum(axis=1)).flatten()
    interactions_per_item = np.array(user_item_matrix.sum(axis=0)).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # User distribution
    axes[0].hist(interactions_per_user, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Number of Interactions')
    axes[0].set_ylabel('Number of Users')
    axes[0].set_title('Distribution of Interactions per User')
    axes[0].grid(True, alpha=0.3)

    # Item distribution
    axes[1].hist(interactions_per_item, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel('Number of Interactions')
    axes[1].set_ylabel('Number of Items')
    axes[1].set_title('Distribution of Interactions per Item')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss curves).

    Args:
        history: Dictionary with 'train_loss' and 'val_loss'
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(history['train_loss']) + 1)

    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)

    if 'val_loss' in history and history['val_loss']:
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_metrics_comparison(results_dict, save_path=None):
    """
    Plot comparison of metrics across different models.

    Args:
        results_dict: Dictionary {model_name: {metric: value}}
        save_path: Path to save figure (optional)
    """
    # Extract metrics
    models = list(results_dict.keys())
    metrics = list(next(iter(results_dict.values())).keys())

    # Organize data
    data = []
    for model in models:
        for metric in metrics:
            data.append({
                'Model': model,
                'Metric': metric,
                'Value': results_dict[model].get(metric, 0)
            })

    df = pd.DataFrame(data)

    # Create subplots for different metric types
    metric_groups = {
        'Precision': [m for m in metrics if 'precision' in m],
        'Recall': [m for m in metrics if 'recall' in m],
        'NDCG': [m for m in metrics if 'ndcg' in m],
        'Hit Rate': [m for m in metrics if 'hit_rate' in m]
    }

    n_groups = sum(1 for group in metric_groups.values() if group)
    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5))

    if n_groups == 1:
        axes = [axes]

    ax_idx = 0
    for group_name, group_metrics in metric_groups.items():
        if not group_metrics:
            continue

        group_df = df[df['Metric'].isin(group_metrics)]

        group_df.pivot(index='Metric', columns='Model', values='Value').plot(
            kind='bar',
            ax=axes[ax_idx],
            rot=45
        )

        axes[ax_idx].set_title(group_name, fontsize=12, fontweight='bold')
        axes[ax_idx].set_ylabel('Score', fontsize=11)
        axes[ax_idx].legend(fontsize=10)
        axes[ax_idx].grid(True, alpha=0.3, axis='y')

        ax_idx += 1

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


# ==================== RECOMMENDATION UTILITIES ====================

def get_popular_items(user_item_matrix, top_k=20):
    """
    Get most popular items based on interaction count.

    Args:
        user_item_matrix: Sparse interaction matrix
        top_k: Number of top items to return

    Returns:
        List of (item_idx, count) tuples
    """
    item_counts = np.array(user_item_matrix.sum(axis=0)).flatten()
    top_indices = np.argsort(item_counts)[::-1][:top_k]

    popular_items = [(idx, item_counts[idx]) for idx in top_indices]
    return popular_items


def get_user_diversity_score(recommended_items, product_features):
    """
    Calculate diversity score of recommended items based on categories.

    Args:
        recommended_items: List of recommended item indices
        product_features: Product feature matrix

    Returns:
        Diversity score (0-1, higher is more diverse)
    """
    if len(recommended_items) < 2:
        return 0.0

    # Get categories of recommended items
    categories = product_features[recommended_items, 0]  # Assuming first column is category
    unique_categories = len(set(categories))

    # Diversity is ratio of unique categories to total items
    diversity = unique_categories / len(recommended_items)

    return diversity


# ==================== EXPORT UTILITIES ====================

def export_recommendations_to_csv(recommendations_dict, output_path):
    """
    Export recommendations to CSV file.

    Args:
        recommendations_dict: Dictionary {user_id: DataFrame of recommendations}
        output_path: Path to save CSV file
    """
    all_recommendations = []

    for user_id, recs_df in recommendations_dict.items():
        recs_df['user_id'] = user_id
        all_recommendations.append(recs_df)

    combined_df = pd.concat(all_recommendations, ignore_index=True)
    combined_df.to_csv(output_path, index=False)

    print(f"Recommendations exported to {output_path}")
    print(f"Total users: {len(recommendations_dict)}")
    print(f"Total recommendations: {len(combined_df)}")


# ==================== MAIN FUNCTION ====================

def main():
    """Demo of utility functions."""
    print("=" * 60)
    print("UTILITY FUNCTIONS DEMO")
    print("=" * 60)

    # Load features
    features = load_all_features()

    if 'user_item_matrix' in features:
        # Analyze sparsity
        stats = analyze_sparsity(features['user_item_matrix'])
        print_sparsity_stats(stats)

        # Get popular items
        popular = get_popular_items(features['user_item_matrix'], top_k=10)
        print("\n" + "=" * 60)
        print("TOP 10 POPULAR ITEMS")
        print("=" * 60)
        for idx, (item_idx, count) in enumerate(popular, 1):
            print(f"{idx}. Item {item_idx}: {count} interactions")

    if 'sequences' in features:
        # Analyze sequences
        seq_stats = analyze_sequences(features['sequences']['sequences'])
        print("\n" + "=" * 60)
        print("SEQUENCE ANALYSIS")
        print("=" * 60)
        print(f"Users with sequences: {seq_stats['n_users_with_sequences']:,}")
        print(f"Average sequence length: {seq_stats['avg_sequence_length']:.2f}")
        print(f"Median sequence length: {seq_stats['median_sequence_length']:.2f}")
        print(f"Min/Max: {seq_stats['min_sequence_length']} / {seq_stats['max_sequence_length']}")

    print("\n✓ Utility functions demo completed!")


if __name__ == "__main__":
    main()