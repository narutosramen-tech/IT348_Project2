#!/usr/bin/env python3
"""
Example: Create custom test data and test the malware detection system.
Useful for creating test folders with different malware rates.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

def create_test_data_folder(test_folder: str = "test_data_demo",
                           n_samples: int = 500,
                           n_features: int = 100,
                           malware_rate: float = 0.15,
                           year: str = "2022"):
    """
    Create a test data folder with synthetic malware data.

    Args:
        test_folder: Folder to create
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        malware_rate: Percentage of malware samples (0.0-1.0)
        year: Year label for the data
    """

    print(f"\nCreating test data in: {test_folder}")
    print(f"Parameters: {n_samples} samples, {n_features} features, {malware_rate:.1%} malware rate")

    # Create directory structure
    test_path = Path(test_folder)
    test_path.mkdir(exist_ok=True)

    # Create benign data
    X_benign = pd.DataFrame(
        np.random.normal(0, 1, (int(n_samples * (1 - malware_rate)), n_features)),
        columns=[f'feature_{i:03d}' for i in range(n_features)]
    )

    # Create malware data (slightly different distribution)
    X_malware = pd.DataFrame(
        np.random.normal(0.5, 1.2, (int(n_samples * malware_rate), n_features)),
        columns=[f'feature_{i:03d}' for i in range(n_features)]
    )

    # Save files (matching expected naming convention)
    benign_file = test_path / f"sampled_{year}_benign_api.csv"
    malware_file = test_path / f"sampled_{year}_malware_api.csv"

    # Add an ID column (simulating apkname)
    X_benign.insert(0, 'apkname', [f'benign_{i}' for i in range(len(X_benign))])
    X_malware.insert(0, 'apkname', [f'malware_{i}' for i in range(len(X_malware))])

    # Save to CSV
    X_benign.to_csv(benign_file, index=False)
    X_malware.to_csv(malware_file, index=False)

    print(f"Created files:")
    print(f"  - {benign_file} ({len(X_benign)} samples)")
    print(f"  - {malware_file} ({len(X_malware)} samples)")
    print(f"Total: {len(X_benign) + len(X_malware)} samples")

    return test_folder

def test_system_with_custom_data():
    """Test the malware detection system with custom test data."""
    print(f"\n{'='*80}")
    print("SYSTEM TEST WITH CUSTOM DATA")
    print(f"{'='*80}")

    # Import the CLI system
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        from malware_detection_cli import MalwareDetectionCLI

        # Initialize CLI
        cli = MalwareDetectionCLI()

        # Step 1: Load main training data
        print(f"\n1. Loading main training data...")
        cli.load_data("input_data")  # Use your actual data folder

        # Step 2: Train a model on 2019 data
        print(f"\n2. Training model on 2019 data...")
        results = cli.train_single_year_model("2019", model_type="ensemble")

        if results is None:
            # If 2019 doesn't exist in your data, use the first available year
            if cli.data:
                first_year = sorted(list(cli.data.keys()))[0]
                print(f"  2019 not found, using {first_year} instead")
                results = cli.train_single_year_model(first_year, model_type="ensemble")

        # Step 3: Create test data with different malware rates
        test_folders = []

        print(f"\n3. Creating test datasets with varying malware rates:")

        test_cases = [
            ("test_low_malware", 0.05),   # 5% malware (easy)
            ("test_medium_malware", 0.15), # 15% malware (typical)
            ("test_high_malware", 0.30),   # 30% malware (high)
            ("test_very_low_malware", 0.02) # 2% malware (very imbalanced)
        ]

        for folder_name, malware_rate in test_cases:
            folder = create_test_data_folder(
                test_folder=folder_name,
                n_samples=300,
                n_features=50,
                malware_rate=malware_rate,
                year="2022"
            )
            test_folders.append((folder, malware_rate))

        # Step 4: Test model on each test dataset
        print(f"\n4. Testing model on different test datasets:")

        for i, (test_folder, malware_rate) in enumerate(test_folders, 1):
            print(f"\n   Test {i}: {test_folder} ({malware_rate:.1%} malware)")
            print(f"   {'-'*40}")

            test_results = cli.test_model(test_folder)

            if test_results:
                metrics = test_results['metrics']
                print(f"   Results:")
                print(f"     Accuracy:  {metrics['accuracy']:.4f}")
                print(f"     F1-Score:  {metrics['f1_score']:.4f}")
                print(f"     Precision: {metrics['precision']:.4f}")
                print(f"     Recall:    {metrics['recall']:.4f}")

                # Calculate expected vs actual malware detection
                expected_malware = int(300 * malware_rate)
                actual_malware = test_results.get('confusion_matrix', [[0, 0], [0, 0]])[1].sum()
                print(f"     Expected malware: {expected_malware}")
                print(f"     Detected malware: {actual_malware}")

        # Step 5: Demonstrate cross-time training
        print(f"\n{'='*80}")
        print("5. DEMONSTRATING CROSS-TIME TRAINING")
        print(f"{'='*80}")

        if cli.data:
            years = sorted(list(cli.data.keys()))
            if len(years) >= 3:
                print(f"\nTraining on {years[0]},{years[1]}, testing on {years[2]}")
                cross_results = cli.train_cross_time_model(
                    train_years=years[:2],
                    test_year=years[2],
                    model_type="ensemble"
                )

                if cross_results and 'ensemble_results' in cross_results:
                    metrics = cross_results['ensemble_results']['metrics']
                    print(f"\nCross-time results:")
                    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
                    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
                    improvement = cross_results.get('improvement', 0)
                    if improvement > 0:
                        print(f"  Improvement over best single model: +{improvement:.4f}")

        # Step 6: Cleanup test folders
        print(f"\n{'='*80}")
        print("CLEANUP")
        print(f"{'='*80}")

        cleanup = input("\nDelete test folders? (y/n): ").lower()
        if cleanup == 'y':
            for folder_name, _ in test_folders:
                import shutil
                if Path(folder_name).exists():
                    shutil.rmtree(folder_name)
                    print(f"  Deleted: {folder_name}")

        print(f"\nTest complete!")

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all required modules are in the same directory.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def quick_start_guide():
    """Quick start guide for using the system."""
    print(f"\n{'='*80}")
    print("QUICK START GUIDE")
    print(f"{'='*80}")

    guide = """
1. BASIC USAGE:
   python run_malware_detection.py
     - Starts interactive mode
     - Follow the prompts

   OR

   python run_malware_detection.py load input_data
     - Loads data from input_data folder

2. TRAINING MODELS:
   Single year training:
     python run_malware_detection.py train single 2019

   Cross-time training (train on 2014-2016, test on 2017):
     python run_malware_detection.py train cross 2014,2015,2016 2017

3. AUTO-RETRAINING:
   Setup auto-retraining:
     python run_malware_detection.py auto setup

   Run auto-retraining:
     python run_malware_detection.py auto run

4. TESTING:
   Test on new data:
     python run_malware_detection.py test test_folder

5. UTILITIES:
   List trained models:
     python run_malware_detection.py list

   Show configuration:
     python run_malware_detection.py config

6. CREATING TEST DATA:
   Use this script to create test datasets:
     python example_custom_test_data.py
    """

    print(guide)

if __name__ == "__main__":
    # Create a menu for this example script
    print(f"\n{'='*80}")
    print("MALWARE DETECTION SYSTEM - EXAMPLE SCRIPT")
    print(f"{'='*80}")

    print("""
1. Create custom test data and test the system
2. Show quick start guide
3. Exit
    """)

    choice = input("\nSelect option (1-3): ").strip()

    if choice == "1":
        test_system_with_custom_data()
    elif choice == "2":
        quick_start_guide()
    else:
        print("Goodbye!")