"""
Data preprocessing module.
Handles data cleaning, filtering, normalization, and train/test split.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import config


def load_merged_data():
    """Load the merged datasets from pickle file."""
    print("Loading merged data...")
    with open(config.PROCESSED_FILES['merged_data'], 'rb') as f:
        merged_data = pickle.load(f)
    print("✓ Data loaded successfully!")
    return merged_data


def clean_order_details(order_details):
    """
    Clean the order details dataframe.
    Remove duplicates, handle missing values, and filter outliers.
    """
    print("\nCleaning order details...")
    initial_shape = order_details.shape

    # Remove duplicates
    order_details = order_details.drop_duplicates()

    # Remove rows with missing critical values
    critical_cols = ['user_id', 'product_id', 'order_id']
    order_details = order_details.dropna(subset=critical_cols)

    # Fill missing values in other columns
    if 'days_since_last_order' in order_details.columns:
        order_details['days_since_last_order'] = (
            order_details['days_since_last_order'].fillna(0)
        )

    if 'reordered' in order_details.columns:
        order_details['reordered'] = order_details['reordered'].fillna(0)

    # Ensure proper data types
    order_details['user_id'] = order_details['user_id'].astype(int)
    order_details['product_id'] = order_details['product_id'].astype(int)
    order_details['order_id'] = order_details['order_id'].astype(int)

    print(f"✓ Cleaned: {initial_shape} → {order_details.shape}")
    return order_details


def filter_by_interactions(order_details):
    """
    Filter users and products by minimum interaction thresholds.
    This removes cold-start items and sparse data.
    """
    print("\nFiltering by interaction thresholds...")
    initial_users = order_details['user_id'].nunique()
    initial_products = order_details['product_id'].nunique()

    # Iteratively filter until convergence
    prev_size = 0
    current_size = len(order_details)
    iteration = 0

    while prev_size != current_size and iteration < 10:
        prev_size = current_size

        # Count interactions per user and product
        user_counts = order_details['user_id'].value_counts()
        product_counts = order_details['product_id'].value_counts()

        # Filter users with minimum interactions
        valid_users = user_counts[
            user_counts >= config.MIN_USER_INTERACTIONS
            ].index
        order_details = order_details[
            order_details['user_id'].isin(valid_users)
        ]

        # Filter products with minimum interactions
        valid_products = product_counts[
            product_counts >= config.MIN_PRODUCT_INTERACTIONS
            ].index
        order_details = order_details[
            order_details['product_id'].isin(valid_products)
        ]

        current_size = len(order_details)
        iteration += 1

    final_users = order_details['user_id'].nunique()
    final_products = order_details['product_id'].nunique()

    print(f"✓ Users: {initial_users:,} → {final_users:,}")
    print(f"✓ Products: {initial_products:,} → {final_products:,}")
    print(f"✓ Interactions: {len(order_details):,}")

    return order_details


def encode_categorical_features(user_features, products):
    """
    Encode categorical features using LabelEncoder.
    Returns encoded dataframes and the encoders for later use.
    """
    print("\nEncoding categorical features...")
    encoders = {}

    # Encode user gender
    if 'gender' in user_features.columns:
        le_gender = LabelEncoder()
        user_features['gender_encoded'] = le_gender.fit_transform(
            user_features['gender'].fillna('unknown')
        )
        encoders['gender'] = le_gender

    # Encode product categories
    if 'category' in products.columns:
        le_category = LabelEncoder()
        products['category_encoded'] = le_category.fit_transform(
            products['category'].fillna('unknown')
        )
        encoders['category'] = le_category

    if 'sub_category' in products.columns:
        le_subcategory = LabelEncoder()
        products['sub_category_encoded'] = le_subcategory.fit_transform(
            products['sub_category'].fillna('unknown')
        )
        encoders['sub_category'] = le_subcategory

    print(f"✓ Encoded {len(encoders)} categorical features")
    return user_features, products, encoders


def create_age_groups(user_features):
    """Bin age into groups for better generalization."""
    print("\nCreating age groups...")

    if 'age' in user_features.columns:
        user_features['age_group'] = pd.cut(
            user_features['age'],
            bins=config.AGE_BINS,
            labels=False,
            include_lowest=True
        )
        user_features['age_group'] = user_features['age_group'].fillna(0)
        print("✓ Age groups created")

    return user_features


def normalize_features(user_features):
    """Normalize numerical features for better model training."""
    print("\nNormalizing numerical features...")

    numeric_features = [
        'age', 'total_orders', 'total_products',
        'avg_basket_size', 'reorder_ratio', 'avg_days_between_orders'
    ]

    # Only normalize features that exist
    features_to_normalize = [
        f for f in numeric_features if f in user_features.columns
    ]

    scaler = StandardScaler()
    user_features[features_to_normalize] = scaler.fit_transform(
        user_features[features_to_normalize]
    )

    print(f"✓ Normalized {len(features_to_normalize)} features")
    return user_features, scaler


def create_train_test_split(order_details):
    """
    Split data into train and test sets.
    For each user, use the last N orders as test set.
    """
    print("\nCreating train/test split...")

    # Sort by user and order number
    order_details = order_details.sort_values(
        ['user_id', 'order_number']
    ).reset_index(drop=True)

    train_data = []
    test_data = []

    # For each user, split their orders
    for user_id, user_orders in order_details.groupby('user_id'):
        n_orders = len(user_orders)
        split_idx = int(n_orders * config.TRAIN_TEST_SPLIT_RATIO)

        # Ensure at least 1 order in test set
        if split_idx >= n_orders:
            split_idx = n_orders - 1

        train_data.append(user_orders.iloc[:split_idx])
        test_data.append(user_orders.iloc[split_idx:])

    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)

    print(f"✓ Train set: {len(train_df):,} interactions")
    print(f"✓ Test set: {len(test_df):,} interactions")

    return train_df, test_df


def save_preprocessed_data(data_dict):
    """Save all preprocessed data."""
    print("\nSaving preprocessed data...")

    for name, data in data_dict.items():
        if name in config.PROCESSED_FILES:
            path = config.PROCESSED_FILES[name]
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Saved {name}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    # Load merged data
    merged_data = load_merged_data()

    # Clean order details
    order_details = clean_order_details(merged_data['order_details'])

    # Filter by interaction thresholds
    order_details = filter_by_interactions(order_details)

    # Encode categorical features
    user_features, products, encoders = encode_categorical_features(
        merged_data['user_features'],
        merged_data['products']
    )

    # Create age groups
    user_features = create_age_groups(user_features)

    # Normalize features
    user_features, scaler = normalize_features(user_features)

    # Filter user_features to only include users in order_details
    valid_users = order_details['user_id'].unique()
    user_features = user_features[user_features['user_id'].isin(valid_users)]

    # Filter products to only include products in order_details
    valid_products = order_details['product_id'].unique()
    products = products[products['product_id'].isin(valid_products)]

    # Create train/test split
    train_data, test_data = create_train_test_split(order_details)

    # Prepare data dictionary to save
    preprocessed_data = {
        'train_data': train_data,
        'test_data': test_data,
        'user_features': user_features,
        'products': products,
        'encoders': {**encoders, 'scaler': scaler}
    }

    # Save preprocessed data
    save_preprocessed_data(preprocessed_data)

    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Users: {user_features['user_id'].nunique():,}")
    print(f"Products: {products['product_id'].nunique():,}")
    print(f"Train interactions: {len(train_data):,}")
    print(f"Test interactions: {len(test_data):,}")
    print("\n✓ Preprocessing completed successfully!")


if __name__ == "__main__":
    main()
