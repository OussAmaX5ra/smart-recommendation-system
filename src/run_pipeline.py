"""
Master Pipeline Script
Runs all 8 steps of the recommendation system in sequence
"""

import sys
import time
import subprocess
from pathlib import Path


def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def run_step(step_number, step_name, script_name):
    """Run a single pipeline step"""
    print_banner(f"STEP {step_number}: {step_name}")

    start_time = time.time()

    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )

        elapsed_time = time.time() - start_time

        print(f"\n✓ Step {step_number} completed successfully in {elapsed_time:.2f} seconds")
        return True

    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ Step {step_number} failed after {elapsed_time:.2f} seconds")
        print(f"Error: {e}")
        return False
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ Step {step_number} encountered an error after {elapsed_time:.2f} seconds")
        print(f"Error: {e}")
        return False


def main():
    """Run the complete pipeline"""
    print_banner("HYBRID RECOMMENDATION SYSTEM - COMPLETE PIPELINE")

    print("This script will execute all 8 steps of the recommendation pipeline:")
    print("1. Merge Data")
    print("2. Preprocessing")
    print("3. Feature Engineering")
    print("4. Train Collaborative Filtering Model")
    print("5. Train Deep Learning Model")
    print("6. Create Hybrid Recommendations")
    print("7. Evaluate with MAP@20")
    print("8. Generate Recommendations for Test Users")

    input("\nPress Enter to start the pipeline...")

    # Define all steps
    steps = [
        (1, "Merge Data", "1_merge_data.py"),
        (2, "Preprocessing", "2_preprocessing.py"),
        (3, "Feature Engineering", "3_feature_engineering.py"),
        (4, "Train Collaborative Filtering", "4_train_collaborative.py"),
        (5, "Train Deep Learning Model", "5_train_deep_learning.py"),
        (6, "Hybrid Recommender", "6_hybrid_recommender.py"),
        (7, "Evaluation (MAP@20)", "7_evaluation.py"),
        (8, "Inference", "8_inference.py")
    ]

    # Track timing
    total_start_time = time.time()
    successful_steps = 0

    # Run each step
    for step_number, step_name, script_name in steps:
        success = run_step(step_number, step_name, script_name)

        if success:
            successful_steps += 1
        else:
            print("\n" + "=" * 70)
            print(f"  PIPELINE STOPPED AT STEP {step_number}")
            print("=" * 70)
            print(f"\nCompleted {successful_steps}/{len(steps)} steps")
            sys.exit(1)

    # Calculate total time
    total_elapsed_time = time.time() - total_start_time

    # Print final summary
    print_banner("PIPELINE COMPLETED SUCCESSFULLY! 🎉")

    print(f"Total time: {total_elapsed_time:.2f} seconds ({total_elapsed_time / 60:.2f} minutes)")
    print(f"Steps completed: {successful_steps}/{len(steps)}")

    print("\n📊 Results:")
    print(f"  - Evaluation metrics: output/evaluation_metrics.json")
    print(f"  - Recommendations: output/recommendations.csv")

    print("\n" + "=" * 70)
    print("  Next steps:")
    print("  1. Review evaluation_metrics.json for model performance")
    print("  2. Check recommendations.csv for final predictions")
    print("  3. Adjust hyperparameters in config.py if needed")
    print("  4. Re-run specific steps or the entire pipeline")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()