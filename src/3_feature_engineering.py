"""
Step 3: Feature Engineering
Creates features for the recommendation models:
- User features (purchase frequency, diversity, etc.)
- Product features (popularity, reorder rate, etc.)
- Interaction features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import config
from utils import print_section, print_stats, save_pickle


def load_preprocessed_data():
    """Load preprocessed data"""
    print_section("Loading Preprocessed Data")

    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(config.VAL_DATA_PATH)

    print(f"✓ Train data: {train_df.shape}")
    print(f"✓ Validation data: {val_df.shape}")

    return train_df, val_df


def create_user_features(df):
    """
    Create user-level features
    """
    print_section("Creating User Features")

    user_features = df.groupby('user_id').agg({
        'order_id': 'nunique',  # Total number of orders
        'product_id': 'nunique',  # Total unique products purchased
        'reordered': 'mean',  # Reorder rate
        'add_to_cart_order': 'mean',  # Average cart position
        'days_since_last_order': 'mean',  # Average days between orders
        'order_hour': 'mean',  # Average order hour
        'order_day': 'mean'  # Average order day
    }).reset_index()

    user_features.columns = [
        'user_id', 'user_total_orders', 'user_total_products',
        'user_reorder_rate', 'user_avg_cart_position',
        'user_avg_days_between_orders', 'user_avg_order_hour',
        'user_avg_order_day'
    ]

    # Calculate user diversity (number of unique categories)
    if 'category_id' in df.columns:
        user_diversity = df.groupby('user_id')['category_id'].nunique().reset_index()
        user_diversity.columns = ['user_id', 'user_category_diversity']
        user_features = pd.merge(user_features, user_diversity, on='user_id', how='left')

    # Calculate average basket size
    basket_size = df.groupby(['user_id', 'order_id']).size().reset_index(name='basket_size')
    avg_basket = basket_size.groupby('user_id')['basket_size'].mean().reset_index()
    avg_basket.columns = ['user_id', 'user_avg_basket_size']
    user_features = pd.merge(user_features, avg_basket, on='user_id', how='left')

    print(f"Created {len(user_features.columns) - 1} user features")
    print(f"For {len(user_features)} users")

    return user_features


def create_product_features(df):
    """
    Create product-level features
    """
    print_section("Creating Product Features")

    product_features = df.groupby('product_id').agg({
        'order_id': 'nunique',  # Number of orders containing this product
        'user_id': 'nunique',  # Number of unique users who bought this
        'reordered': 'mean',  # Product reorder rate
        'add_to_cart_order': 'mean'  # Average cart position
    }).reset_index()

    product_features.columns = [
        'product_id', 'product_total_orders', 'product_total_users',
        'product_reorder_rate', 'product_avg_cart_position'
    ]

    # Calculate product popularity (normalized)
    product_features['product_popularity'] = (
            product_features['product_total_orders'] /
            product_features['product_total_orders'].max()
    )

    # Calculate product purchase frequency
    total_orders = df['order_id'].nunique()
    product_features['product_frequency'] = (
            product_features['product_total_orders'] / total_orders
    )

    print(f"Created {len(product_features.columns) - 1} product features")
    print(f"For {len(product_features)} products")

    return product_features


def create_user_product_features(df):
    """
    Create user-product interaction features
    """
    print_section("Creating User-Product Features")

    user_product_features = df.groupby(['user_id', 'product_id']).agg({
        'order_id': 'count',  # Number of times user bought this product
        'reordered': 'sum',  # Number of times reordered
        'add_to_cart_order': 'mean',  # Average position in cart
        'days_since_last_order': 'mean'  # Average days since last order
    }).reset_index()

    user_product_features.columns = [
        'user_id', 'product_id', 'up_order_count', 'up_reorder_count',
        'up_avg_cart_position', 'up_avg_days_since_last'
    ]

    # Calculate user-product purchase rate
    user_order_counts = df.groupby('user_id')['order_id'].nunique().reset_index()
    user_order_counts.columns = ['user_id', 'user_order_count']

    user_product_features = pd.merge(
        user_product_features,
        user_order_counts,
        on='user_id',
        how='left'
    )

    user_product_features['up_purchase_rate'] = (
            user_product_features['up_order_count'] /
            user_product_features['user_order_count']
    )

    user_product_features.drop('user_order_count', axis=1, inplace=True)

    print(f"Created {len(user_product_features.columns) - 2} user-product features")
    print(f"For {len(user_product_features)} user-product pairs")

    return user_product_features


def create_category_features(df):
    """
    Create category-based features
    """
    print_section("Creating Category Features")

    if 'category_id' not in df.columns or 'sub_category_id' not in df.columns:
        print("⚠ Category columns not found, skipping category features")
        return None

    # User-category interaction
    user_category = df.groupby(['user_id', 'category_id']).size().reset_index(name='uc_purchase_count')

    # User-subcategory interaction
    user_subcategory = df.groupby(['user_id', 'sub_category_id']).size().reset_index(name='usc_purchase_count')

    print(f"Created category features")
    print(f"  User-Category pairs: {len(user_category)}")
    print(f"  User-Subcategory pairs: {len(user_subcategory)}")

    return user_category, user_subcategory


def merge_all_features(df, user_features, product_features, user_product_features):
    """
    Merge all features into the main dataframe
    """
    print_section("Merging All Features")

    # Merge user features
    df = pd.merge(df, user_features, on='user_id', how='left')
    print(f"After merging user features: {df.shape}")

    # Merge product features
    df = pd.merge(df, product_features, on='product_id', how='left')
    print(f"After merging product features: {df.shape}")

    # Merge user-product features
    df = pd.merge(df, user_product_features, on=['user_id', 'product_id'], how='left')
    print(f"After merging user-product features: {df.shape}")

    # Fill any remaining NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


def encode_categorical_features(train_df, val_df):
    """
    Encode categorical features
    """
    print_section("Encoding Categorical Features")

    encoders = {}

    # Encode user_id
    user_encoder = LabelEncoder()
    train_df['user_encoded'] = user_encoder.fit_transform(train_df['user_id'])

    # For validation, handle unseen users
    val_users = val_df['user_id'].unique()
    train_users = set(user_encoder.classes_)
    unseen_users = set(val_users) - train_users

    if unseen_users:
        print(f"⚠ Warning: {len(unseen_users)} users in validation not seen in training")
        # Add unseen users to encoder
        user_encoder.classes_ = np.append(user_encoder.classes_, list(unseen_users))

    val_df['user_encoded'] = user_encoder.transform(val_df['user_id'])
    encoders['user_encoder'] = user_encoder

    # Encode product_id
    product_encoder = LabelEncoder()
    train_df['product_encoded'] = product_encoder.fit_transform(train_df['product_id'])

    # Handle unseen products in validation
    val_products = val_df['product_id'].unique()
    train_products = set(product_encoder.classes_)
    unseen_products = set(val_products) - train_products

    if unseen_products:
        print(f"⚠ Warning: {len(unseen_products)} products in validation not seen in training")
        product_encoder.classes_ = np.append(product_encoder.classes_, list(unseen_products))

    val_df['product_encoded'] = product_encoder.transform(val_df['product_id'])
    encoders['product_encoder'] = product_encoder

    # Encode gender
    if 'gender' in train_df.columns:
        gender_encoder = LabelEncoder()
        train_df['gender_encoded'] = gender_encoder.fit_transform(train_df['gender'].astype(str))

        # Handle unseen genders
        val_genders = set(val_df['gender'].astype(str).unique())
        train_genders = set(gender_encoder.classes_)
        if val_genders - train_genders:
            gender_encoder.classes_ = np.append(
                gender_encoder.classes_,
                list(val_genders - train_genders)
            )

        val_df['gender_encoded'] = gender_encoder.transform(val_df['gender'].astype(str))
        encoders['gender_encoder'] = gender_encoder

    print(f"Encoded features:")
    print(f"  Users: {len(user_encoder.classes_)}")
    print(f"  Products: {len(product_encoder.classes_)}")

    return train_df, val_df, encoders


def normalize_features(train_df, val_df):
    """
    Normalize numerical features
    """
    print_section("Normalizing Features")

    # Select numerical features to normalize (exclude IDs and encoded features)
    exclude_cols = ['user_id', 'product_id', 'order_id', 'user_encoded',
                    'product_encoded', 'gender_encoded', 'reordered',
                    'category_id', 'sub_category_id']

    numerical_cols = [col for col in train_df.select_dtypes(include=[np.number]).columns
                      if col not in exclude_cols]

    print(f"Normalizing {len(numerical_cols)} numerical features")

    scaler = StandardScaler()
    train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
    val_df[numerical_cols] = scaler.transform(val_df[numerical_cols])

    return train_df, val_df, scaler


def save_features_and_encoders(train_df, val_df, encoders, scaler):
    """
    Save feature-engineered data and encoders
    """
    print_section("Saving Features and Encoders")

    # Save data
    train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    val_df.to_csv(config.VAL_DATA_PATH, index=False)
    print(f"✓ Saved updated train data: {train_df.shape}")
    print(f"✓ Saved updated validation data: {val_df.shape}")

    # Save encoders
    save_pickle(encoders['user_encoder'], config.USER_ENCODER_PATH)
    save_pickle(encoders['product_encoder'], config.PRODUCT_ENCODER_PATH)
    save_pickle(scaler, config.SCALER_PATH)
    print(f"✓ Saved encoders and scaler")


def main():
    """Main feature engineering pipeline"""
    print("\n" + "=" * 60)
    print("  STEP 3: FEATURE ENGINEERING")
    print("=" * 60)

    # Load data
    train_df, val_df = load_preprocessed_data()

    # Create features
    user_features = create_user_features(train_df)
    product_features = create_product_features(train_df)
    user_product_features = create_user_product_features(train_df)
    category_features = create_category_features(train_df)

    # Merge features
    train_df = merge_all_features(train_df, user_features, product_features, user_product_features)
    val_df = merge_all_features(val_df, user_features, product_features, user_product_features)

    # Encode categorical features
    train_df, val_df, encoders = encode_categorical_features(train_df, val_df)

    # Normalize features
    train_df, val_df, scaler = normalize_features(train_df, val_df)

    # Display final statistics
    print_stats(train_df, "Feature-Engineered Train Data")

    # Save everything
    save_features_and_encoders(train_df, val_df, encoders, scaler)

    print("\n" + "=" * 60)
    print("  STEP 3 COMPLETED SUCCESSFULLY!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()