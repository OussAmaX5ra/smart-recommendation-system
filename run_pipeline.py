"""
Complete pipeline script to run the entire recommendation system.
This script runs all steps from data merging to evaluation.
"""

import sys
import time
from datetime import datetime


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def run_step(step_name, module_name):
    """
    Run a pipeline step.

    Args:
        step_name: Name of the step for display
        module_name: Python module to run

    Returns:
        True if successful, False otherwise
    """
    print_header(f"STEP: {step_name}")
    start_time = time.time()

    try:
        # Import and run the module
        module = __import__(module_name)
        module.main()

        elapsed = time.time() - start_time
        print(f"\n✓ {step_name} completed in {elapsed:.2f} seconds")
        return True

    except Exception as e:
        print(f"\n✗ Error in {step_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the complete pipeline."""
    print("=" * 70)
    print("  HYBRID RECOMMENDATION SYSTEM - COMPLETE PIPELINE")
    print("=" * 70)
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    pipeline_start = time.time()

    # Define pipeline steps
    steps = [
        ("Data Merging", "data_merger"),
        ("Data Preprocessing", "data_preprocessing"),
        ("Feature Engineering", "feature_engineering"),
        ("Model Training", "train"),
        ("Model Evaluation", "evaluate")
    ]

    # Track results
    results = []

    # Run each step
    for step_name, module_name in steps:
        success = run_step(step_name, module_name)
        results.append((step_name, success))

        if not success:
            print(f"\n⚠️  Pipeline stopped at: {step_name}")
            print("Please fix the errors and try again.")
            break

    # Print summary
    print_header("PIPELINE SUMMARY")

    all_success = all(success for _, success in results)

    for step_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {step_name:30s} {status}")

    total_time = time.time() - pipeline_start
    print(f"\nTotal pipeline time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

    if all_success:
        print("\n" + "=" * 70)
        print("  🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 70)
        print("\nYou can now:")
        print("  1. Run 'python inference.py' to generate recommendations")
        print("  2. Use the RecommendationEngine class in your own scripts")
        print("  3. Check the evaluation results for model performance")
        print("\nModel files saved in: models/saved_models/")
        print("Processed data saved in: data/processed/")
    else:
        print("\n" + "=" * 70)
        print("  ⚠️  PIPELINE INCOMPLETE")
        print("=" * 70)
        print("\nPlease check the error messages above and fix the issues.")

    return all_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)