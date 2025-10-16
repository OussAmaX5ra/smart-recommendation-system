"""
Step 2: Data Preprocessing
Cleans and preprocesses the merged data:
- Handle missing values
- Remove duplicates
- Filter users and products with minimum interactions
- Create train/validation splits
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import config
from utils import print_section, print_stats


def load_merged_data():
    """Load the merged data from step 1"""
    print_section("Loading Merged Data")

    data = pd.read_csv(config.MERGED_DATA_PATH)
    print(f"✓ Loaded data: {data.shape}")

    return data


def handle_missing_values(df):
    """Handle missing values in the dataset"""
    print_section("Handling Missing Values")

    # Check missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        print("\nMissing values found:")
        print(missing)

        # Fill missing categorical values with 'Unknown'
        categorical_cols = ['category', 'sub_category']
        for col in categorical_cols:
            if col in df.columns and df[col].isnull().any():
                df[col].fillna('Unknown', inplace=True)
                print(f"  Filled {col} with 'Unknown'")

        # Fill missing numerical values with median
        numerical_cols = ['age']
        for col in numerical_cols:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  Filled {col} with median: {median_val}")

        # Fill missing gender with 'Unknown'
        if 'gender' in df.columns and df['gender'].isnull().any():
            df['gender'].fillna('Unknown', inplace=True)
            print(f"  Filled gender with 'Unknown'")

        # Drop rows with critical missing values (user_id, product_id, order_id)
        critical_cols = ['user_id', 'product_id', 'order_id']
        before_drop = len(df)
        df.dropna(subset=critical_cols, inplace=True)
        after_drop = len(df)
        if before_drop != after_drop:
            print(f"  Dropped {before_drop - after_drop} rows with missing critical values")
    else:
        print("\n✓ No missing values found")

    return df


def remove_duplicates(df):
    """Remove duplicate rows"""
    print_section("Removing Duplicates")

    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)

    print(f"Rows before: {before}")
    print(f"Rows after: {after}")
    print(f"Duplicates removed: {before - after}")

    return df


def filter_cold_start_items(df):
    """
    Filter out users and products with too few interactions to reduce cold start problem
    """
    print_section("Filtering Cold Start Users and Products")

    initial_users = df['user_id'].nunique()
    initial_products = df['product_id'].nunique()
    initial_rows = len(df)

    print(f"Initial statistics:")
    print(f"  Users: {initial_users}")
    print(f"  Products: {initial_products}")
    print(f"  Interactions: {initial_rows}")

    # Iterative filtering (users and products influence each other)
    prev_rows = 0
    iteration = 0

    while len(df) != prev_rows and iteration < 10:
        prev_rows = len(df)
        iteration += 1

        # Filter users with minimum orders
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= config.MIN_USER_ORDERS].index
        df = df[df['user_id'].isin(valid_users)]

        # Filter products with minimum orders
        product_counts = df['product_id'].value_counts()
        valid_products = product_counts[product_counts >= config.MIN_PRODUCT_ORDERS].index
        df = df[df['product_id'].isin(valid_products)]

        if iteration > 1:
            print(f"  Iteration {iteration}: {len(df)} rows remaining")

    final_users = df['user_id'].nunique()
    final_products = df['product_id'].nunique()
    final_rows = len(df)

    print(f"\nAfter filtering:")
    print(f"  Users: {final_users} (removed {initial_users - final_users})")
    print(f"  Products: {final_products} (removed {initial_products - final_products})")
    print(f"  Interactions: {final_rows} (removed {initial_rows - final_rows})")

    return df


def create_train_val_split(df):
    """
    Create train/validation split
    For each user, take last few orders as validation set
    """
    print_section("Creating Train/Validation Split")

    # Sort by user and order date/number
    if 'order_day' in df.columns and 'order_hour' in df.columns:
        df = df.sort_values(['user_id', 'order_number', 'order_day', 'order_hour'])
    else:
        df = df.sort_values(['user_id', 'order_number'])

    train_data = []
    val_data = []

    # For each user, split their orders
    for user_id, user_df in df.groupby('user_id'):
        user_orders = user_df['order_id'].unique()
        n_orders = len(user_orders)

        # Calculate split point (80/20 split)
        split_point = int(n_orders * config.TRAIN_SPLIT)

        if split_point < 1:
            # If user has very few orders, put all in training
            train_data.append(user_df)
        else:
            train_orders = user_orders[:split_point]
            val_orders = user_orders[split_point:]

            train_data.append(user_df[user_df['order_id'].isin(train_orders)])
            val_data.append(user_df[user_df['order_id'].isin(val_orders)])

    train_df = pd.concat(train_data, ignore_index=True)
    val_df = pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame()

    print(f"Train set:")
    print(f"  Rows: {len(train_df)}")
    print(f"  Users: {train_df['user_id'].nunique()}")
    print(f"  Products: {train_df['product_id'].nunique()}")

    print(f"\nValidation set:")
    print(f"  Rows: {len(val_df)}")
    print(f"  Users: {val_df['user_id'].nunique()}")
    print(f"  Products: {val_df['product_id'].nunique()}")

    return train_df, val_df


def add_data_quality_checks(df):
    """Perform data quality checks"""
    print_section("Data Quality Checks")

    # Check for negative values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (df[col] < 0).any():
            print(f"⚠ Warning: Negative values found in {col}")

    # Check for reasonable age values
    if 'age' in df.columns:
        age_stats = df['age'].describe()
        print(f"\nAge statistics:")
        print(age_stats)
        if df['age'].max() > 120 or df['age'].min() < 0:
            print("⚠ Warning: Unrealistic age values detected")

    # Check unique counts
    print(f"\nUnique value counts:")
    print(f"  Users: {df['user_id'].nunique()}")
    print(f"  Products: {df['product_id'].nunique()}")
    print(f"  Orders: {df['order_id'].nunique()}")
    if 'category' in df.columns:
        print(f"  Categories: {df['category'].nunique()}")
    if 'sub_category' in df.columns:
        print(f"  Sub-categories: {df['sub_category'].nunique()}")

    return df


def save_preprocessed_data(df, train_df, val_df):
    """Save preprocessed data"""
    print_section("Saving Preprocessed Data")

    # Save full preprocessed data
    df.to_csv(config.PREPROCESSED_DATA_PATH, index=False)
    print(f"✓ Saved preprocessed data to: {config.PREPROCESSED_DATA_PATH}")

    # Save train data
    train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    print(f"✓ Saved train data to: {config.TRAIN_DATA_PATH}")

    # Save validation data
    if not val_df.empty:
        val_df.to_csv(config.VAL_DATA_PATH, index=False)
        print(f"✓ Saved validation data to: {config.VAL_DATA_PATH}")


def main():
    """Main preprocessing pipeline"""
    print("\n" + "=" * 60)
    print("  STEP 2: DATA PREPROCESSING")
    print("=" * 60)

    # Load merged data
    df = load_merged_data()

    # Handle missing values
    df = handle_missing_values(df)

    # Remove duplicates
    df = remove_duplicates(df)

    # Filter cold start items
    df = filter_cold_start_items(df)

    # Data quality checks
    df = add_data_quality_checks(df)

    # Create train/validation split
    train_df, val_df = create_train_val_split(df)

    # Display statistics
    print_stats(df, "Preprocessed Data")

    # Save preprocessed data
    save_preprocessed_data(df, train_df, val_df)

    print("\n" + "=" * 60)
    print("  STEP 2 COMPLETED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()