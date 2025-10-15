"""
Configuration file for the hybrid recommendation system.
Contains all hyperparameters, paths, and model configurations.
"""

import os

# ==================== DATA PATHS ====================
DATA_DIR = "data/"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = "models/saved_models"

# CSV file paths
CSV_FILES = {
    'users': os.path.join(RAW_DATA_DIR, 'users_df.csv'),
    'user_test': os.path.join(RAW_DATA_DIR, 'user_test_df.csv'),
    'orders': os.path.join(RAW_DATA_DIR, 'orders_df.csv'),
    'orders_products': os.path.join(RAW_DATA_DIR, 'orders_products_df.csv'),
    'products': os.path.join(RAW_DATA_DIR, 'products_df.csv'),
    'categories': os.path.join(RAW_DATA_DIR, 'category_df.csv'),
    'sub_categories': os.path.join(RAW_DATA_DIR, 'sub_category_df.csv')
}

# Processed data paths
# Processed data paths
PROCESSED_FILES = {
    'merged_data': os.path.join(PROCESSED_DATA_DIR, 'merged_data.pkl'),
    'user_item_matrix': os.path.join(PROCESSED_DATA_DIR, 'user_item_matrix.pkl'),
    'sequences': os.path.join(PROCESSED_DATA_DIR, 'sequences.pkl'),
    'encoders': os.path.join(PROCESSED_DATA_DIR, 'encoders.pkl'),
    'train_data': os.path.join(PROCESSED_DATA_DIR, 'train_data.pkl'),
    'test_data': os.path.join(PROCESSED_DATA_DIR, 'test_data.pkl'),
    'user_features': os.path.join(PROCESSED_DATA_DIR, 'user_features.pkl'),  # ADD THIS
    'products': os.path.join(PROCESSED_DATA_DIR, 'products.pkl')  # ADD THIS
}

# ==================== DATA PREPROCESSING ====================
# Train/test split
TRAIN_TEST_SPLIT_RATIO = 0.8
RANDOM_SEED = 42

# Minimum thresholds for filtering
MIN_USER_INTERACTIONS = 5  # Minimum orders per user
MIN_PRODUCT_INTERACTIONS = 5  # Minimum times a product is ordered

# Sequence parameters
MAX_SEQUENCE_LENGTH = 20  # Maximum order history length per user

# ==================== MODEL HYPERPARAMETERS ====================

# Collaborative Filtering (Matrix Factorization)
CF_CONFIG = {
    'embedding_dim': 64,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'epochs': 50,
    'batch_size': 512
}

# Temporal Model (RNN/Transformer)
TEMPORAL_CONFIG = {
    'model_type': 'transformer',  # 'rnn', 'lstm', 'gru', or 'transformer'
    'embedding_dim': 128,
    'hidden_dim': 256,
    'num_layers': 2,
    'num_heads': 4,  # For transformer
    'dropout': 0.2,
    'learning_rate': 0.0005,
    'epochs': 30,
    'batch_size': 128
}

# Hybrid Model
HYBRID_CONFIG = {
    'cf_weight': 0.4,  # Weight for collaborative filtering
    'temporal_weight': 0.4,  # Weight for temporal model
    'demographic_weight': 0.2,  # Weight for demographic features
    'fusion_method': 'weighted_sum',  # 'weighted_sum' or 'mlp'
    'mlp_hidden_dims': [256, 128],  # If using MLP fusion
    'learning_rate': 0.0001,
    'epochs': 20,
    'batch_size': 256
}

# ==================== TRAINING PARAMETERS ====================
DEVICE = 'cuda'  # 'cuda' or 'cpu'
EARLY_STOPPING_PATIENCE = 5
VALIDATION_SPLIT = 0.1

# ==================== EVALUATION PARAMETERS ====================
TOP_K = [5, 10, 20]  # For Precision@K, Recall@K, NDCG@K
EVAL_METRICS = ['precision', 'recall', 'ndcg', 'hit_rate', 'mrr']

# ==================== INFERENCE PARAMETERS ====================
NUM_RECOMMENDATIONS = 10  # Number of products to recommend per user

# ==================== FEATURE ENGINEERING ====================
# Demographic features
DEMOGRAPHIC_FEATURES = ['gender', 'age']
AGE_BINS = [0, 18, 25, 35, 45, 55, 65, 100]  # Age groups

# Temporal features
TEMPORAL_FEATURES = [
    'order_hour',
    'order_day',
    'days_since_last_order'
]

# Product features
PRODUCT_FEATURES = [
    'category_id',
    'sub_category_id'
]

# User behavior features
BEHAVIOR_FEATURES = [
    'total_orders',
    'avg_basket_size',
    'reorder_ratio',
    'avg_days_between_orders'
]

# ==================== HELPER FUNCTIONS ====================
def create_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_DIR
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def get_model_path(model_name):
    """Get the full path for a saved model."""
    return os.path.join(MODEL_DIR, f"{model_name}.pt")

if __name__ == "__main__":
    create_directories()
    print("Configuration loaded successfully!")
    print(f"Data directory: {DATA_DIR}")
    print(f"Model directory: {MODEL_DIR}")