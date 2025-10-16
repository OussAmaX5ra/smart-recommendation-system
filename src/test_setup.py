"""
Test Setup Script
Validates data files and system configuration before running the pipeline
"""

import os
import pandas as pd
import sys
import config


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check_directories():
    """Check if required directories exist"""
    print_section("Checking Directories")

    directories = [
        config.DATA_DIR,
        config.PROCESSED_DIR,
        config.MODELS_DIR,
        config.OUTPUT_DIR
    ]

    all_exist = True
    for directory in directories:
        if os.path.exists(directory):
            print(f"✓ {directory} exists")
        else:
            print(f"✗ {directory} does not exist - creating it...")
            os.makedirs(directory, exist_ok=True)
            all_exist = False

    return all_exist


def check_data_files():
    """Check if all required CSV files exist"""
    print_section("Checking Data Files")

    required_files = {
        'users_df.csv': config.USERS_PATH,
        'user_test_df.csv': config.USER_TEST_PATH,
        'orders_df.csv': config.ORDERS_PATH,
        'orders_products_df.csv': config.ORDERS_PRODUCTS_PATH,
        'product_df.csv': config.PRODUCT_PATH,
        'category_df.csv': config.CATEGORY_PATH,
        'sub_category_df.csv': config.SUB_CATEGORY_PATH
    }

    missing_files = []

    for name, path in required_files.items():
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                print(f"✓ {name}: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                print(f"✗ {name}: Error reading file - {e}")
                missing_files.append(name)
        else:
            print(f"✗ {name}: File not found at {path}")
            missing_files.append(name)

    return len(missing_files) == 0, missing_files


def validate_data_schema():
    """Validate that CSV files have expected columns"""
    print_section("Validating Data Schema")

    schemas = {
        'users_df.csv': ['user_id', 'gender', 'age'],
        'orders_df.csv': ['order_id', 'user_id', 'order_number', 'order_day', 'order_hour'],
        'orders_products_df.csv': ['order_id', 'product_id', 'add_to_cart_order', 'reordered'],
        'product_df.csv': ['product_id', 'sub_category_id', 'category_id'],
        'category_df.csv': ['category_id', 'category'],
        'sub_category_df.csv': ['sub_category_id', 'sub_category']
    }

    all_valid = True

    for filename, expected_cols in schemas.items():
        filepath = os.path.join(config.DATA_DIR, filename)

        if not os.path.exists(filepath):
            continue

        try:
            df = pd.read_csv(filepath)
            actual_cols = df.columns.tolist()

            missing_cols = set(expected_cols) - set(actual_cols)
            extra_cols = set(actual_cols) - set(expected_cols)

            if missing_cols:
                print(f"✗ {filename}: Missing columns: {missing_cols}")
                all_valid = False
            elif extra_cols:
                print(f"⚠ {filename}: Has extra columns: {extra_cols}")
                print(f"  Expected: {expected_cols}")
                print(f"  Actual: {actual_cols}")
            else:
                print(f"✓ {filename}: Schema valid")

        except Exception as e:
            print(f"✗ {filename}: Error validating - {e}")
            all_valid = False

    return all_valid


def check_data_quality():
    """Check basic data quality"""
    print_section("Checking Data Quality")

    try:
        # Check users
        users = pd.read_csv(config.USERS_PATH)
        print(f"\nUsers:")
        print(f"  Total users: {users['user_id'].nunique()}")
        print(f"  Duplicate user_ids: {users['user_id'].duplicated().sum()}")

        # Check orders
        orders = pd.read_csv(config.ORDERS_PATH)
        print(f"\nOrders:")
        print(f"  Total orders: {orders['order_id'].nunique()}")
        print(f"  Unique users: {orders['user_id'].nunique()}")
        print(f"  Average orders per user: {len(orders) / orders['user_id'].nunique():.2f}")

        # Check orders_products
        orders_products = pd.read_csv(config.ORDERS_PRODUCTS_PATH)
        print(f"\nOrders-Products:")
        print(f"  Total records: {len(orders_products)}")
        print(f"  Unique products: {orders_products['product_id'].nunique()}")
        print(f"  Reorder rate: {orders_products['reordered'].mean():.2%}")

        # Check products
        products = pd.read_csv(config.PRODUCT_PATH)
        print(f"\nProducts:")
        print(f"  Total products: {products['product_id'].nunique()}")

        # Check categories
        categories = pd.read_csv(config.CATEGORY_PATH)
        print(f"\nCategories:")
        print(f"  Total categories: {categories['category_id'].nunique()}")

        # Check test users
        if os.path.exists(config.USER_TEST_PATH):
            test_users = pd.read_csv(config.USER_TEST_PATH)
            print(f"\nTest Users:")
            print(f"  Total test users: {len(test_users)}")

        return True

    except Exception as e:
        print(f"✗ Error checking data quality: {e}")
        return False


def check_dependencies():
    """Check if required Python packages are installed"""
    print_section("Checking Dependencies")

    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'surprise': 'scikit-surprise',
        'tensorflow': 'tensorflow'
    }

    missing_packages = []

    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name} is installed")
        except ImportError:
            print(f"✗ {package_name} is NOT installed")
            missing_packages.append(package_name)

    return len(missing_packages) == 0, missing_packages


def print_configuration():
    """Print current configuration"""
    print_section("Current Configuration")

    print(f"\nCollaborative Filtering:")
    print(f"  Factors: {config.CF_N_FACTORS}")
    print(f"  Epochs: {config.CF_N_EPOCHS}")
    print(f"  Learning Rate: {config.CF_LR_ALL}")

    print(f"\nDeep Learning:")
    print(f"  Embedding Dim: {config.DL_EMBEDDING_DIM}")
    print(f"  Hidden Units: {config.DL_HIDDEN_UNITS}")
    print(f"  Epochs: {config.DL_EPOCHS}")
    print(f"  Batch Size: {config.DL_BATCH_SIZE}")

    print(f"\nHybrid Model:")
    print(f"  CF Weight: {config.CF_WEIGHT}")
    print(f"  DL Weight: {config.DL_WEIGHT}")

    print(f"\nRecommendation:")
    print(f"  Top K: {config.TOP_K}")

    print(f"\nPreprocessing:")
    print(f"  Min User Orders: {config.MIN_USER_ORDERS}")
    print(f"  Min Product Orders: {config.MIN_PRODUCT_ORDERS}")


def main():
    """Run all setup tests"""
    print("\n" + "=" * 60)
    print("  SETUP VALIDATION FOR RECOMMENDATION SYSTEM")
    print("=" * 60)

    all_passed = True

    # Check dependencies
    deps_ok, missing_deps = check_dependencies()
    if not deps_ok:
        print(f"\n⚠ Missing packages: {missing_deps}")
        print("Install with: pip install -r requirements.txt")
        all_passed = False

    # Check directories
    dirs_ok = check_directories()

    # Check data files
    files_ok, missing_files = check_data_files()
    if not files_ok:
        print(f"\n⚠ Missing files: {missing_files}")
        print("Please place all required CSV files in the data/ directory")
        all_passed = False

    # Validate schema
    if files_ok:
        schema_ok = validate_data_schema()
        if not schema_ok:
            print("\n⚠ Some files have incorrect schema")
            all_passed = False

        # Check data quality
        quality_ok = check_data_quality()
        if not quality_ok:
            all_passed = False

    # Print configuration
    print_configuration()

    # Final summary
    print_section("VALIDATION SUMMARY")

    if all_passed:
        print("✓ All checks passed!")
        print("\nYou can now run the pipeline:")
        print("  python run_pipeline.py")
        print("\nOr run individual steps:")
        print("  python 1_merge_data.py")
        print("  python 2_preprocessing.py")
        print("  ...")
    else:
        print("✗ Some checks failed")
        print("\nPlease fix the issues above before running the pipeline")
        sys.exit(1)

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()