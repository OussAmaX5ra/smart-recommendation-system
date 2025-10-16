"""
Configuration file for the Hybrid Recommendation System
Contains all paths, hyperparameters, and constants
"""

import os

# ==================== PATHS ====================
# Data paths
DATA_DIR = 'data'
USERS_PATH = os.path.join(DATA_DIR, 'users_df.csv')
USER_TEST_PATH = os.path.join(DATA_DIR, 'user_test_df.csv')
ORDERS_PATH = os.path.join(DATA_DIR, 'orders_df.csv')
ORDERS_PRODUCTS_PATH = os.path.join(DATA_DIR, 'orders_products_df.csv')
PRODUCT_PATH = os.path.join(DATA_DIR, 'products_df.csv')
CATEGORY_PATH = os.path.join(DATA_DIR, 'category_df.csv')
SUB_CATEGORY_PATH = os.path.join(DATA_DIR, 'sub_category_df.csv')

# Processed data paths
PROCESSED_DIR = 'processed'
MERGED_DATA_PATH = os.path.join(PROCESSED_DIR, 'merged_data.csv')
PREPROCESSED_DATA_PATH = os.path.join(PROCESSED_DIR, 'preprocessed_data.csv')
FEATURES_PATH = os.path.join(PROCESSED_DIR, 'features.csv')
TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, 'train_data.csv')
VAL_DATA_PATH = os.path.join(PROCESSED_DIR, 'val_data.csv')

# Model paths
MODELS_DIR = 'models'
CF_MODEL_PATH = os.path.join(MODELS_DIR, 'cf_model.pkl')
DL_MODEL_PATH = os.path.join(MODELS_DIR, 'dl_model.h5')
USER_ENCODER_PATH = os.path.join(MODELS_DIR, 'user_encoder.pkl')
PRODUCT_ENCODER_PATH = os.path.join(MODELS_DIR, 'product_encoder.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

# Output paths
OUTPUT_DIR = 'output'
RECOMMENDATIONS_PATH = os.path.join(OUTPUT_DIR, 'recommendations.csv')
EVALUATION_METRICS_PATH = os.path.join(OUTPUT_DIR, 'evaluation_metrics.json')

# ==================== HYPERPARAMETERS ====================
# Collaborative Filtering parameters
CF_N_FACTORS = 100  # Number of latent factors for matrix factorization
CF_N_EPOCHS = 20    # Number of training epochs
CF_LR_ALL = 0.005   # Learning rate
CF_REG_ALL = 0.02   # Regularization term

# Deep Learning parameters
DL_EMBEDDING_DIM = 64       # Embedding dimension for users and products
DL_HIDDEN_UNITS = [128, 64] # Hidden layer sizes
DL_DROPOUT_RATE = 0.3       # Dropout rate
DL_BATCH_SIZE = 512         # Batch size for training
DL_EPOCHS = 20              # Number of training epochs
DL_LEARNING_RATE = 0.001    # Learning rate
DL_VALIDATION_SPLIT = 0.2   # Validation split ratio

# RNN/Transformer parameters
SEQUENCE_LENGTH = 10        # Maximum sequence length for user history
RNN_UNITS = 128            # Number of RNN units
ATTENTION_HEADS = 4        # Number of attention heads for Transformer

# Hybrid model parameters
CF_WEIGHT = 0.5            # Weight for CF predictions in hybrid model
DL_WEIGHT = 0.5            # Weight for DL predictions in hybrid model

# ==================== PREPROCESSING ====================
MIN_USER_ORDERS = 3        # Minimum orders per user to keep
MIN_PRODUCT_ORDERS = 5     # Minimum orders per product to keep
TRAIN_SPLIT = 0.8          # Train/validation split ratio

# ==================== RECOMMENDATION ====================
TOP_K = 20                 # Number of recommendations per user (for MAP@20)
RECOMMENDATION_CANDIDATES = 200  # Number of candidates to generate before filtering

# ==================== OTHER ====================
RANDOM_SEED = 42          # Random seed for reproducibility

# Create directories if they don't exist
for directory in [PROCESSED_DIR, MODELS_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)