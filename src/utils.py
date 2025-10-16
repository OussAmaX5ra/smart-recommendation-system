"""
Utility functions for the recommendation system
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import json
import pickle


def load_pickle(filepath: str):
    """Load a pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, filepath: str):
    """Save object to pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def save_json(data: dict, filepath: str):
    """Save dictionary to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_map_at_k(actual: List[List[int]], predicted: List[List[int]], k: int = 20) -> float:
    """
    Calculate Mean Average Precision at K

    Args:
        actual: List of lists of actual relevant items for each user
        predicted: List of lists of predicted items for each user
        k: Number of recommendations to consider

    Returns:
        MAP@K score
    """
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")

    average_precisions = []

    for act, pred in zip(actual, predicted):
        if len(act) == 0:
            continue

        # Take top k predictions
        pred_k = pred[:k]

        # Calculate precision at each position
        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(pred_k):
            if p in act:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if num_hits > 0:
            average_precisions.append(score / min(len(act), k))
        else:
            average_precisions.append(0.0)

    return np.mean(average_precisions) if average_precisions else 0.0


def get_user_purchased_products(df: pd.DataFrame, user_id: int) -> set:
    """
    Get all products purchased by a user

    Args:
        df: DataFrame with user_id and product_id columns
        user_id: User ID

    Returns:
        Set of product IDs
    """
    user_products = df[df['user_id'] == user_id]['product_id'].unique()
    return set(user_products)


def filter_already_purchased(recommendations: Dict[int, List[int]],
                             user_purchased: Dict[int, set]) -> Dict[int, List[int]]:
    """
    Filter out products that users have already purchased

    Args:
        recommendations: Dictionary mapping user_id to list of recommended product_ids
        user_purchased: Dictionary mapping user_id to set of purchased product_ids

    Returns:
        Filtered recommendations dictionary
    """
    filtered_recs = {}

    for user_id, product_list in recommendations.items():
        purchased = user_purchased.get(user_id, set())
        # Filter out already purchased products
        filtered = [p for p in product_list if p not in purchased]
        filtered_recs[user_id] = filtered

    return filtered_recs


def create_interaction_matrix(df: pd.DataFrame,
                              user_col: str = 'user_id',
                              item_col: str = 'product_id',
                              rating_col: str = None) -> Tuple[pd.DataFrame, dict, dict]:
    """
    Create user-item interaction matrix

    Args:
        df: DataFrame with user and item interactions
        user_col: Name of user column
        item_col: Name of item column
        rating_col: Name of rating column (if None, will use count of interactions)

    Returns:
        Interaction matrix, user mapping, item mapping
    """
    if rating_col is None:
        # Count interactions
        interaction_df = df.groupby([user_col, item_col]).size().reset_index(name='rating')
    else:
        interaction_df = df[[user_col, item_col, rating_col]].copy()
        interaction_df.rename(columns={rating_col: 'rating'}, inplace=True)

    # Create mappings
    users = interaction_df[user_col].unique()
    items = interaction_df[item_col].unique()

    user_to_idx = {user: idx for idx, user in enumerate(users)}
    item_to_idx = {item: idx for idx, item in enumerate(items)}

    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}

    # Create matrix
    matrix = pd.pivot_table(
        interaction_df,
        values='rating',
        index=user_col,
        columns=item_col,
        fill_value=0
    )

    mappings = {
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_item': idx_to_item
    }

    return matrix, mappings


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_stats(df: pd.DataFrame, name: str = "DataFrame"):
    """Print basic statistics about a DataFrame"""
    print(f"\n{name} Statistics:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    print(f"\nFirst few rows:")
    print(df.head())