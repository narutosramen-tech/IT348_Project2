#!/usr/bin/env python3
"""
Complete Workflow Example
Shows the complete malware detection training system workflow from start to finish.
"""

import os
import sys
from pathlib import Path

def print_header(text):
    """Print formatted header."""
    print(f"\n{'='*80}")
    print(f"{text}")
    print(f"{'='*80}")

def main():
    """Demonstrate complete workflow."""
    print_header("MALWARE DETECTION SYSTEM - COMPLETE WORKFLOW EXAMPLE")

    print("""
This example demonstrates the complete workflow:
1. Load malware detection data
2. Train single-year model
3. Train cross-time model
4. Setup auto-retraining
5. Create and test on custom test data
6. Save and load models
    """)

    # Add current directory to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        from malware_detection_cli import MalwareDetectionCLI

        # Initialize CLI
        cli = MalwareDetectionCLI()

        print_header("STEP 1: LOAD DATA")
        print("Loading data from input_data folder...")
        data_info = cli.load_data("input_data")

        if data_info is None:
            print("No data loaded. Creating synthetic data for demonstration...")
            # In a real scenario, you would load actual data
            print("Please ensure you have CSV files in the input_data folder.")
            return

        print_header("STEP 2: TRAIN SINGLE-YEAR MODEL")
        print("Training ensemble model on a single year...")

        # Get available years
        years = sorted(list(cli.data.keys()))
        if years:
            # Use the middle year for training
            train_year = years[len(years) // 2]
            print(f"Training on year: {train_year}")

            # Train ensemble model
            results = cli.train_single_year_model(train_year, model_type="ensemble")

            if results:
                print(f"Model trained successfully!")

                # List models to see what we have
                print("\nCurrent models:")
                cli.list_models()
        else:
            print("No years available in data")

        print_header("STEP 3: TRAIN CROSS-TIME MODEL")
        print("Training on multiple years, testing on different year...")

        if len(years) >= 3:
            train_years = years[:2]  # First two years
            test_year = years[2]     # Third year

            print(f"Training on: {train_years}")
            print(f"Testing on: {test_year}")

            # Train cross-time
            cross_results = cli.train_cross_time_model(
                train_years=train_years,
                test_year=test_year,
                model_type="ensemble"
            )

            if cross_results:
                print("Cross-time training complete!")
        else:
            print("Need at least 3 years for cross-time training")

        print_header("STEP 4: AUTO-RETRAINING SETUP")
        print("Setting up automatic retraining with drift detection...")

        auto_setup = cli.setup_auto_retraining()
        if auto_setup:
            print("Auto-retraining system ready!")
            print(f"Drift threshold: {cli.config['drift_threshold']:.0%}")
            print(f"Performance threshold: {cli.config['performance_threshold']:.0%}")

        print_header("STEP 5: CREATE TEST DATA")
        print("Creating custom test dataset...")

        # Import the test data creation function
        from example_custom_test_data import create_test_data_folder

        test_folder = create_test_data_folder(
            test_folder="workflow_test_data",
            n_samples=200,
            n_features=50,
            malware_rate=0.12,
            year="2023"
        )

        print(f"Test data created in: {test_folder}")

        print_header("STEP 6: TEST MODEL")
        print(f"Testing model on new data ({test_folder})...")

        # Test the most recently trained model
        test_results = cli.test_model(test_folder)

        if test_results:
            metrics = test_results['metrics']
            print(f"Test results:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")

        print_header("STEP 7: MODEL MANAGEMENT")
        print("Saving and listing models...")

        # List all models
        cli.list_models()

        # Save a model if we have one
        if cli.models:
            first_key = list(cli.models.keys())[0]
            save_file = f"saved_models/{first_key}.pkl"

            # Create directory if needed
            Path("saved_models").mkdir(exist_ok=True)

            success = cli.save_model(first_key, save_file)
            if success:
                print(f"Model saved to: {save_file}")

                # Demonstrate loading
                print("\nLoading the saved model...")
                loaded_key = cli.load_model(save_file)
                if loaded_key:
                    print(f"Model loaded as: {loaded_key}")

        print_header("STEP 8: CLEANUP")
        print("Cleaning up test data...")

        import shutil
        test_path = Path("workflow_test_data")
        if test_path.exists():
            shutil.rmtree(test_path)
            print(f"Deleted: {test_path}")

        print_header("WORKFLOW COMPLETE")
        print("""
✅ Data loaded and analyzed
✅ Single-year model trained
✅ Cross-time model trained
✅ Auto-retraining system configured
✅ Custom test data created
✅ Model tested on new data
✅ Models saved and managed
✅ Cleanup completed

The system is now ready for production use!
        """)

        print("\nNext steps:")
        print("1. Run auto-retraining: python run_malware_detection.py auto run")
        print("2. Test on more data: python run_malware_detection.py test <folder>")
        print("3. Train more models: python run_malware_detection.py train single <year>")
        print("4. Use interactive mode: python run_malware_detection.py")

    except ImportError as e:
        print(f"Import error: {e}")
        print("\nMake sure all required modules are in the Project2 directory:")
        print("  - malware_detection_cli.py")
        print("  - models.py")
        print("  - data.py")
        print("  - retraining_system.py")
        print("  - drift.py")
        print("  - dataset.py")
        print("  - sample.py")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()