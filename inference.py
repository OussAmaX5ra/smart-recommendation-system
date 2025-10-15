"""
Inference module for generating recommendations.
Provides functions to get product recommendations for users.
"""

import torch
import numpy as np
import pandas as pd
import pickle
import sys

sys.path.append('models')

import config
from models.collaborative_filtering import MatrixFactorization
from models.temporal_model import LSTMRecommender, TransformerRecommender
from models.hybrid_model import HybridRecommender


class RecommendationEngine:
    """
    Main recommendation engine that loads models and generates recommendations.
    """

    def __init__(self, device='cuda'):
        """
        Initialize the recommendation engine.

        Args:
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Initializing Recommendation Engine on {self.device}")

        # Load features and mappings
        self._load_features()

        # Load models
        self._load_models()

        print("✓ Recommendation Engine ready!")

    def _load_features(self):
        """Load all necessary features and mappings."""
        print("Loading features...")

        features_dir = config.PROCESSED_DATA_DIR

        # Load mappings
        with open(f"{features_dir}/mappings.pkl", 'rb') as f:
            self.mappings = pickle.load(f)

        # Load sequences
        with open(f"{features_dir}/sequences.pkl", 'rb') as f:
            self.sequences = pickle.load(f)

        # Load demographic features
        with open(f"{features_dir}/demographic_features.pkl", 'rb') as f:
            demo_data = pickle.load(f)
            self.demographic_features = demo_data['matrix']
            self.demographic_feature_names = demo_data['feature_names']

        # Load product features
        with open(f"{features_dir}/product_features.pkl", 'rb') as f:
            product_data = pickle.load(f)
            self.product_features = product_data['matrix']

        # Load user item matrix (for training history)
        with open(f"{features_dir}/user_item_matrix.pkl", 'rb') as f:
            self.user_item_matrix = pickle.load(f)

        # Load original data for product names/details
        with open(config.PROCESSED_FILES['merged_data'], 'rb') as f:
            merged_data = pickle.load(f)
            self.products_df = merged_data['products']

        print("✓ Features loaded")

    def _load_models(self):
        """Load trained models."""
        print("Loading models...")

        n_users = self.mappings['n_users']
        n_items = self.mappings['n_products']

        # Load CF model
        self.cf_model = MatrixFactorization(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=config.CF_CONFIG['embedding_dim']
        )
        self.cf_model.load_state_dict(
            torch.load(config.get_model_path('cf_model'), map_location=self.device)
        )
        self.cf_model = self.cf_model.to(self.device)
        self.cf_model.eval()

        # Load Temporal model
        model_type = config.TEMPORAL_CONFIG['model_type']
        if model_type in ['lstm', 'gru', 'rnn']:
            self.temporal_model = LSTMRecommender(
                n_items=n_items,
                embedding_dim=config.TEMPORAL_CONFIG['embedding_dim'],
                hidden_dim=config.TEMPORAL_CONFIG['hidden_dim'],
                num_layers=config.TEMPORAL_CONFIG['num_layers'],
                dropout=config.TEMPORAL_CONFIG['dropout']
            )
        else:
            self.temporal_model = TransformerRecommender(
                n_items=n_items,
                embedding_dim=config.TEMPORAL_CONFIG['embedding_dim'],
                num_heads=config.TEMPORAL_CONFIG['num_heads'],
                num_layers=config.TEMPORAL_CONFIG['num_layers'],
                hidden_dim=config.TEMPORAL_CONFIG['hidden_dim'],
                dropout=config.TEMPORAL_CONFIG['dropout'],
                max_seq_len=config.MAX_SEQUENCE_LENGTH
            )
        self.temporal_model.load_state_dict(
            torch.load(config.get_model_path('temporal_model'), map_location=self.device)
        )
        self.temporal_model = self.temporal_model.to(self.device)
        self.temporal_model.eval()

        # Load Hybrid model
        demo_dim = self.demographic_features.shape[1]
        self.hybrid_model = HybridRecommender(
            cf_model=self.cf_model,
            temporal_model=self.temporal_model,
            n_users=n_users,
            n_items=n_items,
            demographic_dim=demo_dim,
            cf_weight=config.HYBRID_CONFIG['cf_weight'],
            temporal_weight=config.HYBRID_CONFIG['temporal_weight'],
            demographic_weight=config.HYBRID_CONFIG['demographic_weight'],
            fusion_method=config.HYBRID_CONFIG['fusion_method']
        )
        self.hybrid_model.load_state_dict(
            torch.load(config.get_model_path('hybrid_model'), map_location=self.device)
        )
        self.hybrid_model = self.hybrid_model.to(self.device)
        self.hybrid_model.eval()

        print("✓ Models loaded")

    def get_user_info(self, user_id):
        """
        Get information about a user.

        Args:
            user_id: Original user ID

        Returns:
            Dictionary with user information
        """
        if user_id not in self.mappings['user_to_idx']:
            return None

        user_idx = self.mappings['user_to_idx'][user_id]

        # Get purchase history
        user_items = self.user_item_matrix[user_idx].nonzero()[1]
        purchased_products = [
            self.mappings['idx_to_product'][item_idx]
            for item_idx in user_items
        ]

        # Get demographic features
        demo_features = self.demographic_features[user_idx]
        demo_dict = {
            name: float(value)
            for name, value in zip(self.demographic_feature_names, demo_features)
        }

        return {
            'user_id': user_id,
            'user_idx': user_idx,
            'num_purchases': len(purchased_products),
            'demographic_features': demo_dict,
            'recent_purchases': purchased_products[:10]  # Last 10
        }

    def recommend_for_user(self, user_id, top_k=10, model='hybrid'):
        """
        Generate product recommendations for a user.

        Args:
            user_id: Original user ID
            top_k: Number of recommendations to return
            model: Which model to use ('cf', 'temporal', 'hybrid')

        Returns:
            DataFrame with recommended products and scores
        """
        if user_id not in self.mappings['user_to_idx']:
            print(f"User {user_id} not found in training data")
            return None

        user_idx = self.mappings['user_to_idx'][user_id]

        # Get user's purchase history to exclude
        user_history = set(self.user_item_matrix[user_idx].nonzero()[1])

        # Get user sequence
        sequence = self.sequences['sequences'].get(user_idx, [0])

        # Get demographic features
        demographic = self.demographic_features[user_idx]

        # Generate recommendations based on selected model
        with torch.no_grad():
            if model == 'hybrid':
                recommended_indices, scores = self.hybrid_model.predict_for_user(
                    user_idx,
                    sequence,
                    demographic,
                    device=self.device,
                    top_k=top_k,
                    exclude_items=user_history
                )

            elif model == 'cf':
                cf_scores = self.cf_model.predict_all(user_idx).cpu().numpy()
                # Exclude history
                for item_idx in user_history:
                    cf_scores[item_idx] = -np.inf
                recommended_indices = np.argsort(cf_scores)[::-1][:top_k]
                scores = cf_scores[recommended_indices]

            elif model == 'temporal':
                # Prepare sequence
                max_len = config.MAX_SEQUENCE_LENGTH
                if len(sequence) > max_len:
                    sequence = sequence[-max_len:]
                padded_seq = sequence + [0] * (max_len - len(sequence))

                seq_tensor = torch.tensor([padded_seq]).to(self.device)

                if hasattr(self.temporal_model, 'lstm'):
                    length = torch.tensor([len(sequence)]).to(self.device)
                    logits = self.temporal_model(seq_tensor, length)
                else:
                    mask = torch.tensor([[False] * len(sequence) +
                                         [True] * (max_len - len(sequence))]).to(self.device)
                    logits = self.temporal_model(seq_tensor, mask)

                temporal_scores = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

                # Exclude history
                for item_idx in user_history:
                    temporal_scores[item_idx] = -np.inf
                recommended_indices = np.argsort(temporal_scores)[::-1][:top_k]
                scores = temporal_scores[recommended_indices]

        # Convert indices to product IDs
        recommended_product_ids = [
            self.mappings['idx_to_product'][idx]
            for idx in recommended_indices
        ]

        # Create recommendations DataFrame
        recommendations = []
        for product_id, score in zip(recommended_product_ids, scores):
            product_info = self.products_df[
                self.products_df['product_id'] == product_id
                ]

            if not product_info.empty:
                rec = {
                    'product_id': product_id,
                    'score': float(score),
                }

                # Add product details if available
                for col in ['category', 'sub_category', 'category_id', 'sub_category_id']:
                    if col in product_info.columns:
                        rec[col] = product_info[col].values[0]

                recommendations.append(rec)

        return pd.DataFrame(recommendations)

    def recommend_batch(self, user_ids, top_k=10, model='hybrid'):
        """
        Generate recommendations for multiple users.

        Args:
            user_ids: List of user IDs
            top_k: Number of recommendations per user
            model: Which model to use

        Returns:
            Dictionary {user_id: recommendations_df}
        """
        results = {}

        for user_id in user_ids:
            recommendations = self.recommend_for_user(user_id, top_k, model)
            if recommendations is not None:
                results[user_id] = recommendations

        return results

    def compare_models(self, user_id, top_k=10):
        """
        Compare recommendations from different models for a user.

        Args:
            user_id: User ID
            top_k: Number of recommendations

        Returns:
            Dictionary with recommendations from each model
        """
        results = {}

        for model_name in ['cf', 'temporal', 'hybrid']:
            recommendations = self.recommend_for_user(user_id, top_k, model_name)
            if recommendations is not None:
                results[model_name] = recommendations

        return results


def main():
    """
    Demo script showing how to use the recommendation engine.
    """
    print("=" * 60)
    print("HYBRID RECOMMENDATION SYSTEM - INFERENCE")
    print("=" * 60)

    # Initialize engine
    engine = RecommendationEngine(device=config.DEVICE)

    # Get a random user for demo
    random_user_idx = np.random.choice(list(engine.mappings['idx_to_user'].keys()))
    demo_user_id = engine.mappings['idx_to_user'][random_user_idx]

    print(f"\n{'=' * 60}")
    print(f"DEMO: Recommendations for User {demo_user_id}")
    print('=' * 60)

    # Show user info
    user_info = engine.get_user_info(demo_user_id)
    print("\nUser Information:")
    print(f"  Total purchases: {user_info['num_purchases']}")
    print(f"  Demographic features: {user_info['demographic_features']}")

    # Get recommendations from hybrid model
    print(f"\n{'=' * 60}")
    print("HYBRID MODEL RECOMMENDATIONS")
    print('=' * 60)
    recommendations = engine.recommend_for_user(demo_user_id, top_k=10, model='hybrid')
    print(recommendations.to_string(index=False))

    # Compare models
    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON")
    print('=' * 60)
    comparison = engine.compare_models(demo_user_id, top_k=5)

    for model_name, recs in comparison.items():
        print(f"\n{model_name.upper()} Model:")
        print(recs[['product_id', 'score']].to_string(index=False))

    print("\n✓ Inference demo completed!")


if __name__ == "__main__":
    main()