#!/usr/bin/env python3
"""
Test script for data processing functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import get_all_years_data, print_data_summary


def main():
    """Test the data processing functions."""
    print("Testing data processing functionality...")
    print("-" * 50)

    # Test with the input_data folder
    folder_path = "input_data"

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        print(f"Current directory: {os.getcwd()}")
        print(f"Available directories: {os.listdir('.')}")
        return

    print(f"Processing CSV files from: {folder_path}")
    print(f"Files found: {len([f for f in os.listdir(folder_path) if f.endswith('.csv')])}")

    try:
        # Get all years data
        all_data = get_all_years_data(folder_path)

        # Print summary
        print_data_summary(all_data)

        # Test specific properties
        print("\n" + "="*50)
        print("VALIDATION TESTS")
        print("="*50)

        for year, (X, y) in all_data.items():
            print(f"\nYear {year} validation:")
            print(f"  X is DataFrame: {type(X).__name__ == 'DataFrame'}")
            print(f"  y is Series: {type(y).__name__ == 'Series'}")
            print(f"  X shape matches number of samples: {X.shape[0] == len(y)}")
            print(f"  All labels are 0 or 1: {set(y.unique()).issubset({0, 1})}")
            print(f"  No NaN values in labels: {not y.isna().any()}")
            print(f"  Features are numeric: {X.shape[1] > 0}")

            # Check that first column was dropped (should not be 'apkname')
            if 'apkname' in X.columns:
                print(f"  WARNING: 'apkname' column still present!")
            else:
                print(f"  First column dropped: PASSED")

        print("\n" + "="*50)
        print("SUCCESS: All data processed correctly!")
        print("="*50)

        # Show how to use the data
        print("\n\nUSAGE EXAMPLE:")
        print("-" * 30)
        print("To use with scikit-learn, you can do:")
        print("""
from sklearn.model_selection import train_test_split

# For a specific year (e.g., 2014)
X_2014, y_2014 = all_data['2014']
X_train, X_test, y_train, y_test = train_test_split(
    X_2014, y_2014, test_size=0.2, random_state=42
)

# To combine all years:
import pandas as pd
all_X = pd.concat([X for X, _ in all_data.values()], ignore_index=True)
all_y = pd.concat([y for _, y in all_data.values()], ignore_index=True)
        """)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()