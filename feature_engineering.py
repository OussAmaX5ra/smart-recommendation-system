"""
Feature engineering module.
Creates user-item matrix, sequences, and additional features for models.
"""

import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from collections import defaultdict
import config


def load_preprocessed_data():
    """Load preprocessed data."""
    print("Loading preprocessed data...")
    data = {}
    files_to_load = ['train_data', 'test_data', 'user_features', 'products', 'encoders']

    for name in files_to_load:
        with open(config.PROCESSED_FILES[name], 'rb') as f:
            data[name] = pickle.load(f)

    print("✓ Data loaded successfully!")
    return data


def create_id_mappings(train_data):
    """
    Create mappings from original IDs to continuous indices.
    This is essential for embedding layers.
    """
    print("\nCreating ID mappings...")

    # Get unique users and products from training data
    unique_users = sorted(train_data['user_id'].unique())
    unique_products = sorted(train_data['product_id'].unique())

    # Create mappings
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}

    product_to_idx = {prod_id: idx for idx, prod_id in enumerate(unique_products)}
    idx_to_product = {idx: prod_id for prod_id, idx in product_to_idx.items()}

    print(f"✓ Mapped {len(user_to_idx):,} users and {len(product_to_idx):,} products")

    return {
        'user_to_idx': user_to_idx,
        'idx_to_user': idx_to_user,
        'product_to_idx': product_to_idx,
        'idx_to_product': idx_to_product,
        'n_users': len(user_to_idx),
        'n_products': len(product_to_idx)
    }


def create_user_item_matrix(train_data, mappings):
    """
    Create user-item interaction matrix for collaborative filtering.
    Uses implicit feedback (binary interactions).
    """
    print("\nCreating user-item matrix...")

    n_users = mappings['n_users']
    n_products = mappings['n_products']

    # Map IDs to indices
    user_indices = train_data['user_id'].map(mappings['user_to_idx'])
    product_indices = train_data['product_id'].map(mappings['product_to_idx'])

    # Create binary interaction matrix (1 if user bought product)
    interactions = np.ones(len(train_data))

    # Create sparse matrix
    user_item_matrix = csr_matrix(
        (interactions, (user_indices, product_indices)),
        shape=(n_users, n_products)
    )

    # Calculate sparsity
    sparsity = 1 - (user_item_matrix.nnz / (n_users * n_products))

    print(f"✓ Matrix shape: {user_item_matrix.shape}")
    print(f"✓ Sparsity: {sparsity:.4%}")

    return user_item_matrix


def create_user_sequences(train_data, mappings, max_length=None):
    """
    Create temporal sequences of product purchases for each user.
    This is used for the temporal model (RNN/Transformer).
    OPTIMIZED VERSION - Much faster than the original.
    """
    print("\nCreating user sequences...")

    if max_length is None:
        max_length = config.MAX_SEQUENCE_LENGTH

    # Sort by user and order number
    train_data_sorted = train_data.sort_values(
        ['user_id', 'order_number', 'add_to_cart_order']
    ).copy()

    # Filter to only users in mappings
    train_data_sorted = train_data_sorted[
        train_data_sorted['user_id'].isin(mappings['user_to_idx'].keys())
    ]

    # Map IDs to indices once
    train_data_sorted['user_idx'] = train_data_sorted['user_id'].map(mappings['user_to_idx'])
    train_data_sorted['product_idx'] = train_data_sorted['product_id'].map(mappings['product_to_idx'])

    # Drop rows where mapping failed
    train_data_sorted = train_data_sorted.dropna(subset=['user_idx', 'product_idx'])
    train_data_sorted['user_idx'] = train_data_sorted['user_idx'].astype(int)
    train_data_sorted['product_idx'] = train_data_sorted['product_idx'].astype(int)

    # Vectorized groupby aggregation
    print("  Aggregating sequences...")
    sequences_series = train_data_sorted.groupby('user_idx')['product_idx'].apply(list)
    sequences = sequences_series.to_dict()

    # Handle timestamps if available
    timestamps = {}
    if 'order_day' in train_data_sorted.columns and 'order_hour' in train_data_sorted.columns:
        print("  Processing timestamps...")
        train_data_sorted['timestamp_tuple'] = list(
            zip(train_data_sorted['order_day'], train_data_sorted['order_hour'])
        )
        timestamps_series = train_data_sorted.groupby('user_idx')['timestamp_tuple'].apply(list)
        timestamps = timestamps_series.to_dict()

    # Truncate sequences to max_length (keep last items)
    print("  Truncating sequences...")
    processed_sequences = {}
    processed_timestamps = {}

    for user_idx, seq in sequences.items():
        if len(seq) > max_length:
            seq = seq[-max_length:]
        processed_sequences[user_idx] = seq

        if user_idx in timestamps:
            ts = timestamps[user_idx]
            if len(ts) > max_length:
                ts = ts[-max_length:]
            processed_timestamps[user_idx] = ts
        else:
            processed_timestamps[user_idx] = [(0, 0)] * len(seq)

    avg_length = np.mean([len(s) for s in processed_sequences.values()])
    print(f"✓ Created sequences for {len(processed_sequences):,} users")
    print(f"✓ Average sequence length: {avg_length:.1f}")

    return {
        'sequences': processed_sequences,
        'timestamps': processed_timestamps,
        'max_length': max_length
    }


def extract_temporal_features(train_data):
    """
    Extract temporal features from the data.
    """
    print("\nExtracting temporal features...")

    temporal_features = train_data[['user_id', 'product_id']].copy()

    # Order-based features
    if 'order_hour' in train_data.columns:
        # Hour of day (cyclic encoding)
        temporal_features['hour_sin'] = np.sin(
            2 * np.pi * train_data['order_hour'] / 24
        )
        temporal_features['hour_cos'] = np.cos(
            2 * np.pi * train_data['order_hour'] / 24
        )

    if 'order_day' in train_data.columns:
        # Day of week (cyclic encoding)
        temporal_features['day_sin'] = np.sin(
            2 * np.pi * train_data['order_day'] / 7
        )
        temporal_features['day_cos'] = np.cos(
            2 * np.pi * train_data['order_day'] / 7
        )

    if 'days_since_last_order' in train_data.columns:
        # Normalize days since last order
        temporal_features['days_since_last'] = (
            train_data['days_since_last_order'].fillna(0)
        )
        temporal_features['days_since_last'] = (
                temporal_features['days_since_last'] /
                temporal_features['days_since_last'].max()
        )

    print(f"✓ Extracted {temporal_features.shape[1] - 2} temporal features")

    return temporal_features


def create_user_demographic_features(user_features, mappings):
    """
    Create demographic feature matrix for users.
    """
    print("\nCreating demographic features...")

    n_users = mappings['n_users']
    user_features_filtered = user_features[
        user_features['user_id'].isin(mappings['user_to_idx'].keys())
    ].copy()

    # Map user IDs to indices
    user_features_filtered['user_idx'] = (
        user_features_filtered['user_id'].map(mappings['user_to_idx'])
    )

    # Select demographic features
    demo_cols = []
    if 'gender_encoded' in user_features_filtered.columns:
        demo_cols.append('gender_encoded')
    if 'age' in user_features_filtered.columns:
        demo_cols.append('age')
    if 'age_group' in user_features_filtered.columns:
        demo_cols.append('age_group')

    # Select behavior features
    behavior_cols = []
    for feat in config.BEHAVIOR_FEATURES:
        if feat in user_features_filtered.columns:
            behavior_cols.append(feat)

    all_feature_cols = demo_cols + behavior_cols

    # Create feature matrix sorted by user index
    user_features_filtered = user_features_filtered.sort_values('user_idx')
    demographic_matrix = user_features_filtered[all_feature_cols].values

    print(f"✓ Created demographic matrix: {demographic_matrix.shape}")
    print(f"✓ Features: {', '.join(all_feature_cols)}")

    return {
        'matrix': demographic_matrix,
        'feature_names': all_feature_cols
    }


def create_product_features(products, mappings):
    """
    Create product feature matrix.
    """
    print("\nCreating product features...")

    products_filtered = products[
        products['product_id'].isin(mappings['product_to_idx'].keys())
    ].copy()

    # Map product IDs to indices
    products_filtered['product_idx'] = (
        products_filtered['product_id'].map(mappings['product_to_idx'])
    )

    # Select product features
    product_feature_cols = []
    if 'category_encoded' in products_filtered.columns:
        product_feature_cols.append('category_encoded')
    if 'sub_category_encoded' in products_filtered.columns:
        product_feature_cols.append('sub_category_encoded')
    if 'category_id' in products_filtered.columns:
        product_feature_cols.append('category_id')
    if 'sub_category_id' in products_filtered.columns:
        product_feature_cols.append('sub_category_id')

    # Create feature matrix sorted by product index
    products_filtered = products_filtered.sort_values('product_idx')
    product_matrix = products_filtered[product_feature_cols].values

    print(f"✓ Created product matrix: {product_matrix.shape}")
    print(f"✓ Features: {', '.join(product_feature_cols)}")

    return {
        'matrix': product_matrix,
        'feature_names': product_feature_cols
    }


def create_test_evaluation_data(test_data, mappings):
    """
    Prepare test data for evaluation.
    Returns ground truth user-product pairs.
    """
    print("\nPreparing test evaluation data...")

    # Filter test data to only include known users and products
    test_data_filtered = test_data[
        test_data['user_id'].isin(mappings['user_to_idx'].keys()) &
        test_data['product_id'].isin(mappings['product_to_idx'].keys())
        ].copy()

    # Map to indices
    test_data_filtered['user_idx'] = (
        test_data_filtered['user_id'].map(mappings['user_to_idx'])
    )
    test_data_filtered['product_idx'] = (
        test_data_filtered['product_id'].map(mappings['product_to_idx'])
    )

    # Create ground truth dictionary: user_idx -> list of product_idx
    ground_truth = defaultdict(set)
    for _, row in test_data_filtered.iterrows():
        ground_truth[row['user_idx']].add(row['product_idx'])

    # Convert sets to lists
    ground_truth = {k: list(v) for k, v in ground_truth.items()}

    print(f"✓ Prepared ground truth for {len(ground_truth):,} users")

    return ground_truth


def save_engineered_features(features_dict):
    """Save all engineered features."""
    print("\nSaving engineered features...")

    output_path = config.PROCESSED_DATA_DIR

    with open(f"{output_path}/mappings.pkl", 'wb') as f:
        pickle.dump(features_dict['mappings'], f)

    with open(f"{output_path}/user_item_matrix.pkl", 'wb') as f:
        pickle.dump(features_dict['user_item_matrix'], f)

    with open(f"{output_path}/sequences.pkl", 'wb') as f:
        pickle.dump(features_dict['sequences'], f)

    with open(f"{output_path}/demographic_features.pkl", 'wb') as f:
        pickle.dump(features_dict['demographic_features'], f)

    with open(f"{output_path}/product_features.pkl", 'wb') as f:
        pickle.dump(features_dict['product_features'], f)

    with open(f"{output_path}/test_ground_truth.pkl", 'wb') as f:
        pickle.dump(features_dict['test_ground_truth'], f)

    print("✓ All features saved successfully!")


def main():
    """Main execution function."""
    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # Load preprocessed data
    data = load_preprocessed_data()

    # Create ID mappings
    mappings = create_id_mappings(data['train_data'])

    # Create user-item matrix for collaborative filtering
    user_item_matrix = create_user_item_matrix(data['train_data'], mappings)

    # Create user sequences for temporal model
    sequences = create_user_sequences(data['train_data'], mappings)

    # Create demographic features
    demographic_features = create_user_demographic_features(
        data['user_features'], mappings
    )

    # Create product features
    product_features = create_product_features(data['products'], mappings)

    # Prepare test evaluation data
    test_ground_truth = create_test_evaluation_data(data['test_data'], mappings)

    # Package all features
    features_dict = {
        'mappings': mappings,
        'user_item_matrix': user_item_matrix,
        'sequences': sequences,
        'demographic_features': demographic_features,
        'product_features': product_features,
        'test_ground_truth': test_ground_truth
    }

    # Save features
    save_engineered_features(features_dict)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"Users: {mappings['n_users']:,}")
    print(f"Products: {mappings['n_products']:,}")
    print(f"Interactions: {user_item_matrix.nnz:,}")
    print(f"Sequences: {len(sequences['sequences']):,}")
    print(f"Test users: {len(test_ground_truth):,}")
    print("\n✓ Feature engineering completed successfully!")


if __name__ == "__main__":
    main()
