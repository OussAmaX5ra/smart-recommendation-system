"""
Evaluation module for recommendation models.
Implements various metrics: Precision@K, Recall@K, NDCG@K, Hit Rate, MRR.
"""

import torch
import numpy as np
from collections import defaultdict
import pickle
import sys

sys.path.append('models')

import config
from models.collaborative_filtering import MatrixFactorization
from models.temporal_model import LSTMRecommender, TransformerRecommender
from models.hybrid_model import HybridRecommender


# ==================== EVALUATION METRICS ====================

def precision_at_k(recommended_items, relevant_items, k):
    """
    Calculate Precision@K.

    Args:
        recommended_items: List of recommended item indices
        relevant_items: Set of relevant (ground truth) item indices
        k: Top-K

    Returns:
        Precision@K score
    """
    if k == 0 or len(recommended_items) == 0:
        return 0.0

    recommended_at_k = set(recommended_items[:k])
    num_relevant_and_recommended = len(recommended_at_k & relevant_items)

    return num_relevant_and_recommended / k


def recall_at_k(recommended_items, relevant_items, k):
    """
    Calculate Recall@K.

    Args:
        recommended_items: List of recommended item indices
        relevant_items: Set of relevant (ground truth) item indices
        k: Top-K

    Returns:
        Recall@K score
    """
    if len(relevant_items) == 0:
        return 0.0

    recommended_at_k = set(recommended_items[:k])
    num_relevant_and_recommended = len(recommended_at_k & relevant_items)

    return num_relevant_and_recommended / len(relevant_items)


def ndcg_at_k(recommended_items, relevant_items, k):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K).

    Args:
        recommended_items: List of recommended item indices
        relevant_items: Set of relevant (ground truth) item indices
        k: Top-K

    Returns:
        NDCG@K score
    """
    if len(relevant_items) == 0:
        return 0.0

    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(recommended_items[:k]):
        if item in relevant_items:
            # Relevance is binary (1 if relevant, 0 otherwise)
            dcg += 1.0 / np.log2(i + 2)  # i+2 because index starts at 0

    # Calculate IDCG (best possible DCG)
    idcg = 0.0
    for i in range(min(len(relevant_items), k)):
        idcg += 1.0 / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def hit_rate_at_k(recommended_items, relevant_items, k):
    """
    Calculate Hit Rate@K (whether any relevant item is in top-K).

    Args:
        recommended_items: List of recommended item indices
        relevant_items: Set of relevant (ground truth) item indices
        k: Top-K

    Returns:
        Hit Rate@K score (0 or 1)
    """
    recommended_at_k = set(recommended_items[:k])
    return 1.0 if len(recommended_at_k & relevant_items) > 0 else 0.0


def mean_reciprocal_rank(recommended_items, relevant_items):
    """
    Calculate Mean Reciprocal Rank (MRR).

    Args:
        recommended_items: List of recommended item indices
        relevant_items: Set of relevant (ground truth) item indices

    Returns:
        MRR score
    """
    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            return 1.0 / (i + 1)
    return 0.0


# ==================== MODEL EVALUATION ====================

def evaluate_model(model, test_ground_truth, features, device='cuda',
                   top_k_list=[5, 10, 20], model_type='hybrid'):
    """
    Evaluate a recommendation model.

    Args:
        model: Trained model
        test_ground_truth: Dictionary {user_idx: [relevant_item_indices]}
        features: Feature dictionary
        device: Device
        top_k_list: List of K values to evaluate
        model_type: 'cf', 'temporal', or 'hybrid'

    Returns:
        results: Dictionary of evaluation metrics
    """
    model = model.to(device)
    model.eval()

    results = defaultdict(lambda: defaultdict(list))

    print(f"\nEvaluating on {len(test_ground_truth)} users...")

    sequences_dict = features['sequences']['sequences']
    demo_features = features['demographic_features']['matrix']
    n_items = features['mappings']['n_products']

    with torch.no_grad():
        for user_idx, relevant_items in test_ground_truth.items():
            relevant_items_set = set(relevant_items)

            # Get user's training history to exclude
            user_history = set(sequences_dict.get(user_idx, []))

            # Get recommendations based on model type
            if model_type == 'cf':
                scores = model.predict_all(user_idx).cpu().numpy()
                # Exclude training items
                for item_idx in user_history:
                    scores[item_idx] = -np.inf
                recommended_items = np.argsort(scores)[::-1]

            elif model_type == 'temporal':
                sequence = sequences_dict.get(user_idx, [])
                if len(sequence) == 0:
                    continue

                # Pad sequence
                max_len = config.MAX_SEQUENCE_LENGTH
                if len(sequence) > max_len:
                    sequence = sequence[-max_len:]
                padded_seq = sequence + [0] * (max_len - len(sequence))

                seq_tensor = torch.tensor([padded_seq]).to(device)

                if hasattr(model, 'lstm'):
                    length = torch.tensor([len(sequence)]).to(device)
                    logits = model(seq_tensor, length)
                else:
                    mask = torch.tensor([[False] * len(sequence) +
                                         [True] * (max_len - len(sequence))]).to(device)
                    logits = model(seq_tensor, mask)

                scores = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

                # Exclude training items
                for item_idx in user_history:
                    scores[item_idx] = -np.inf
                recommended_items = np.argsort(scores)[::-1]

            elif model_type == 'hybrid':
                sequence = sequences_dict.get(user_idx, [])
                if len(sequence) == 0:
                    sequence = [0]

                demo_feature = demo_features[user_idx]

                recommended_items, scores = model.predict_for_user(
                    user_idx,
                    sequence,
                    demo_feature,
                    device=device,
                    top_k=max(top_k_list),
                    exclude_items=user_history
                )

            # Calculate metrics for different K values
            for k in top_k_list:
                results[f'precision@{k}']['values'].append(
                    precision_at_k(recommended_items, relevant_items_set, k)
                )
                results[f'recall@{k}']['values'].append(
                    recall_at_k(recommended_items, relevant_items_set, k)
                )
                results[f'ndcg@{k}']['values'].append(
                    ndcg_at_k(recommended_items, relevant_items_set, k)
                )
                results[f'hit_rate@{k}']['values'].append(
                    hit_rate_at_k(recommended_items, relevant_items_set, k)
                )

            # MRR (not dependent on K)
            results['mrr']['values'].append(
                mean_reciprocal_rank(recommended_items, relevant_items_set)
            )

    # Calculate average metrics
    final_results = {}
    for metric_name, metric_data in results.items():
        final_results[metric_name] = np.mean(metric_data['values'])

    return final_results


def print_results(results, model_name):
    """Print evaluation results in a formatted table."""
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS - {model_name.upper()}")
    print("=" * 60)

    # Group by metric type
    metric_groups = {
        'Precision': [k for k in results.keys() if 'precision' in k],
        'Recall': [k for k in results.keys() if 'recall' in k],
        'NDCG': [k for k in results.keys() if 'ndcg' in k],
        'Hit Rate': [k for k in results.keys() if 'hit_rate' in k],
        'MRR': ['mrr']
    }

    for group_name, metrics in metric_groups.items():
        print(f"\n{group_name}:")
        for metric in metrics:
            if metric in results:
                print(f"  {metric:20s}: {results[metric]:.4f}")


def load_features():
    """Load all engineered features."""
    print("Loading features...")

    features = {}
    files = [
        'mappings', 'user_item_matrix', 'sequences',
        'demographic_features', 'product_features', 'test_ground_truth'
    ]

    for name in files:
        path = f"{config.PROCESSED_DATA_DIR}/{name}.pkl"
        with open(path, 'rb') as f:
            features[name] = pickle.load(f)

    print("✓ Features loaded!")
    return features


def load_model(model_path, model_class, features, device):
    """Load a trained model."""
    n_users = features['mappings']['n_users']
    n_items = features['mappings']['n_products']

    if model_class == 'cf':
        model = MatrixFactorization(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=config.CF_CONFIG['embedding_dim']
        )

    elif model_class == 'temporal':
        model_type = config.TEMPORAL_CONFIG['model_type']
        if model_type in ['lstm', 'gru', 'rnn']:
            model = LSTMRecommender(
                n_items=n_items,
                embedding_dim=config.TEMPORAL_CONFIG['embedding_dim'],
                hidden_dim=config.TEMPORAL_CONFIG['hidden_dim'],
                num_layers=config.TEMPORAL_CONFIG['num_layers'],
                dropout=config.TEMPORAL_CONFIG['dropout']
            )
        else:
            model = TransformerRecommender(
                n_items=n_items,
                embedding_dim=config.TEMPORAL_CONFIG['embedding_dim'],
                num_heads=config.TEMPORAL_CONFIG['num_heads'],
                num_layers=config.TEMPORAL_CONFIG['num_layers'],
                hidden_dim=config.TEMPORAL_CONFIG['hidden_dim'],
                dropout=config.TEMPORAL_CONFIG['dropout'],
                max_seq_len=config.MAX_SEQUENCE_LENGTH
            )

    elif model_class == 'hybrid':
        # First load CF and temporal models
        cf_model = MatrixFactorization(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=config.CF_CONFIG['embedding_dim']
        )
        cf_model.load_state_dict(torch.load(config.get_model_path('cf_model')))

        model_type = config.TEMPORAL_CONFIG['model_type']
        if model_type in ['lstm', 'gru', 'rnn']:
            temporal_model = LSTMRecommender(
                n_items=n_items,
                embedding_dim=config.TEMPORAL_CONFIG['embedding_dim'],
                hidden_dim=config.TEMPORAL_CONFIG['hidden_dim'],
                num_layers=config.TEMPORAL_CONFIG['num_layers'],
                dropout=config.TEMPORAL_CONFIG['dropout']
            )
        else:
            temporal_model = TransformerRecommender(
                n_items=n_items,
                embedding_dim=config.TEMPORAL_CONFIG['embedding_dim'],
                num_heads=config.TEMPORAL_CONFIG['num_heads'],
                num_layers=config.TEMPORAL_CONFIG['num_layers'],
                hidden_dim=config.TEMPORAL_CONFIG['hidden_dim'],
                dropout=config.TEMPORAL_CONFIG['dropout'],
                max_seq_len=config.MAX_SEQUENCE_LENGTH
            )
        temporal_model.load_state_dict(torch.load(config.get_model_path('temporal_model')))

        demo_dim = features['demographic_features']['matrix'].shape[1]
        model = HybridRecommender(
            cf_model=cf_model,
            temporal_model=temporal_model,
            n_users=n_users,
            n_items=n_items,
            demographic_dim=demo_dim,
            cf_weight=config.HYBRID_CONFIG['cf_weight'],
            temporal_weight=config.HYBRID_CONFIG['temporal_weight'],
            demographic_weight=config.HYBRID_CONFIG['demographic_weight'],
            fusion_method=config.HYBRID_CONFIG['fusion_method']
        )

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    print(f"✓ Model loaded from {model_path}")

    return model


def main():
    """Main evaluation pipeline."""
    print("=" * 60)
    print("HYBRID RECOMMENDATION SYSTEM - EVALUATION")
    print("=" * 60)

    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load features
    features = load_features()
    test_ground_truth = features['test_ground_truth']

    # Evaluate each model
    models_to_evaluate = [
        ('cf_model', 'cf', 'Collaborative Filtering'),
        ('temporal_model', 'temporal', 'Temporal Model'),
        ('hybrid_model', 'hybrid', 'Hybrid Model')
    ]

    all_results = {}

    for model_name, model_class, display_name in models_to_evaluate:
        try:
            print(f"\n{'=' * 60}")
            print(f"Evaluating {display_name}...")
            print('=' * 60)

            model_path = config.get_model_path(model_name)
            model = load_model(model_path, model_class, features, device)

            results = evaluate_model(
                model,
                test_ground_truth,
                features,
                device=device,
                top_k_list=config.TOP_K,
                model_type=model_class
            )

            all_results[display_name] = results
            print_results(results, display_name)

        except FileNotFoundError:
            print(f"✗ Model file not found: {model_path}")
            continue

    # Compare models
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        # Get all metrics
        all_metrics = list(next(iter(all_results.values())).keys())

        print(f"\n{'Metric':<25}", end='')
        for model_name in all_results.keys():
            print(f"{model_name:<20}", end='')
        print()
        print("-" * (25 + 20 * len(all_results)))

        for metric in all_metrics:
            print(f"{metric:<25}", end='')
            for model_name in all_results.keys():
                value = all_results[model_name].get(metric, 0)
                print(f"{value:<20.4f}", end='')
            print()

    print("\n✓ Evaluation completed!")


if __name__ == "__main__":
    main()