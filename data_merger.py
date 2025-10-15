"""
Data merger module.
Strategically merges CSV files to create datasets that benefit the model.
We keep separate dataframes for different purposes rather than merging everything.
"""

import pandas as pd
import pickle
import config


def load_csv_files():
    """Load all CSV files into pandas DataFrames."""
    print("Loading CSV files...")

    dataframes = {}
    for name, path in config.CSV_FILES.items():
        try:
            df = pd.read_csv(path)
            dataframes[name] = df
            print(f"✓ Loaded {name}: {df.shape}")
        except FileNotFoundError:
            print(f"✗ Warning: {path} not found. Skipping {name}.")
            dataframes[name] = None

    return dataframes


def merge_product_hierarchy(dfs):
    """
    Merge product, category, and sub_category tables.
    This creates a complete product feature table.
    """
    print("\nMerging product hierarchy...")

    products = dfs['products'].copy()

    # Merge with sub_categories
    if dfs['sub_categories'] is not None:
        products = products.merge(
            dfs['sub_categories'],
            on='sub_category_id',
            how='left'
        )

    # Merge with categories
    if dfs['categories'] is not None:
        products = products.merge(
            dfs['categories'],
            on='category_id',
            how='left'
        )

    print(f"✓ Product hierarchy merged: {products.shape}")
    return products


def merge_order_details(dfs):
    """
    Merge orders with orders_products to create order-level details.
    This is the core transactional data for the model.
    """
    print("\nMerging order details...")

    # Merge orders with orders_products
    order_details = dfs['orders_products'].merge(
        dfs['orders'],
        on='order_id',
        how='left'
    )

    print(f"✓ Order details merged: {order_details.shape}")
    return order_details


def create_user_features(dfs, order_details):
    """
    Create enriched user features by combining users table with order statistics.
    """
    print("\nCreating user features...")

    users = dfs['users'].copy()

    # Calculate user behavior statistics from orders
    user_stats = order_details.groupby('user_id').agg({
        'order_id': 'nunique',  # Total number of orders
        'product_id': 'count',  # Total products ordered
        'reordered': 'mean',  # Reorder ratio
        'days_since_last_order': 'mean'  # Average days between orders
    }).reset_index()

    user_stats.columns = [
        'user_id', 'total_orders', 'total_products',
        'reorder_ratio', 'avg_days_between_orders'
    ]

    # Calculate average basket size
    user_stats['avg_basket_size'] = (
            user_stats['total_products'] / user_stats['total_orders']
    )

    # Merge with user demographics
    user_features = users.merge(user_stats, on='user_id', how='left')

    # Fill NaN values for users with no orders
    numeric_cols = user_features.select_dtypes(include=['float64', 'int64']).columns
    user_features[numeric_cols] = user_features[numeric_cols].fillna(0)

    print(f"✓ User features created: {user_features.shape}")
    return user_features


def create_merged_datasets(dfs):
    """
    Create strategically merged datasets for different model components.

    Returns:
        dict: Dictionary containing different merged datasets
    """
    merged = {}

    # 1. Product hierarchy (for product embeddings and features)
    merged['products'] = merge_product_hierarchy(dfs)

    # 2. Order details (core transactional data)
    merged['order_details'] = merge_order_details(dfs)

    # 3. User features (demographics + behavior)
    merged['user_features'] = create_user_features(dfs, merged['order_details'])

    # 4. Keep user_test separate for final evaluation
    merged['user_test'] = dfs['user_test']

    return merged


def save_merged_data(merged_data, output_path):
    """Save merged datasets to pickle file."""
    print(f"\nSaving merged data to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(merged_data, f)
    print("✓ Data saved successfully!")


def main():
    """Main execution function."""
    print("=" * 60)
    print("DATA MERGER")
    print("=" * 60)

    # Create directories
    config.create_directories()

    # Load CSV files
    dfs = load_csv_files()

    # Create merged datasets
    merged_data = create_merged_datasets(dfs)

    # Display summary
    print("\n" + "=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)
    for name, df in merged_data.items():
        if df is not None:
            print(f"{name:20s}: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Save merged data
    save_merged_data(merged_data, config.PROCESSED_FILES['merged_data'])

    print("\n✓ Data merging completed successfully!")


if __name__ == "__main__":
    main()