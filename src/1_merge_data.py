"""
Step 1: Merge Data
Merges relevant CSV files to create a unified dataset for recommendation system.
We merge: orders -> orders_products -> product -> sub_category -> category
         and separately keep users information
"""

import pandas as pd
import config
from utils import print_section, print_stats


def load_data():
    """Load all CSV files"""
    print_section("Loading CSV Files")

    # Load all datasets
    users = pd.read_csv(config.USERS_PATH)
    orders = pd.read_csv(config.ORDERS_PATH)
    orders_products = pd.read_csv(config.ORDERS_PRODUCTS_PATH)
    product = pd.read_csv(config.PRODUCT_PATH)
    category = pd.read_csv(config.CATEGORY_PATH)
    sub_category = pd.read_csv(config.SUB_CATEGORY_PATH)

    print(f"✓ Loaded users: {users.shape}")
    print(f"✓ Loaded orders: {orders.shape}")
    print(f"✓ Loaded orders_products: {orders_products.shape}")
    print(f"✓ Loaded product: {product.shape}")
    print(f"✓ Loaded category: {category.shape}")
    print(f"✓ Loaded sub_category: {sub_category.shape}")

    return users, orders, orders_products, product, category, sub_category


def merge_order_data(orders, orders_products, product, sub_category, category):
    """
    Merge order-related data to get complete order information with product details

    Merge strategy:
    1. orders + orders_products (to get which products in which orders)
    2. + product (to get product category information)
    3. + sub_category (to get subcategory names)
    4. + category (to get category names)
    """
    print_section("Merging Order and Product Data")

    # Step 1: Merge orders with orders_products
    print("\n1. Merging orders with orders_products...")
    merged = pd.merge(
        orders,
        orders_products,
        on='order_id',
        how='inner'
    )
    print(f"   After merge: {merged.shape}")

    # Step 2: Merge with product information
    print("2. Merging with product information...")
    merged = pd.merge(
        merged,
        product,
        on='product_id',
        how='left'
    )
    print(f"   After merge: {merged.shape}")

    # Step 3: Merge with sub_category
    print("3. Merging with sub_category...")
    merged = pd.merge(
        merged,
        sub_category,
        on='sub_category_id',
        how='left'
    )
    print(f"   After merge: {merged.shape}")

    # Step 4: Merge with category
    print("4. Merging with category...")
    merged = pd.merge(
        merged,
        category,
        on='category_id',
        how='left'
    )
    print(f"   After merge: {merged.shape}")

    return merged


def merge_user_data(order_data, users):
    """
    Merge user demographic information with order data
    """
    print_section("Merging User Data")

    merged = pd.merge(
        order_data,
        users,
        on='user_id',
        how='left'
    )

    print(f"Final merged shape: {merged.shape}")
    print(f"\nColumns in merged data: {list(merged.columns)}")

    return merged


def save_merged_data(merged_data):
    """Save merged data to CSV"""
    print_section("Saving Merged Data")

    merged_data.to_csv(config.MERGED_DATA_PATH, index=False)
    print(f"✓ Saved merged data to: {config.MERGED_DATA_PATH}")
    print(f"  Shape: {merged_data.shape}")
    print(f"  Size: {merged_data.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")


def main():
    """Main function to execute data merging pipeline"""
    print("\n" + "=" * 60)
    print("  STEP 1: DATA MERGING")
    print("=" * 60)

    # Load all data
    users, orders, orders_products, product, category, sub_category = load_data()

    # Merge order and product data
    order_data = merge_order_data(orders, orders_products, product, sub_category, category)

    # Merge with user data
    merged_data = merge_user_data(order_data, users)

    # Display sample of merged data
    print_stats(merged_data, "Merged Data")

    # Check for missing values
    print_section("Missing Values Check")
    missing = merged_data.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print("\nColumns with missing values:")
        print(missing)
    else:
        print("\n✓ No missing values found")

    # Save merged data
    save_merged_data(merged_data)

    print("\n" + "=" * 60)
    print("  STEP 1 COMPLETED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()