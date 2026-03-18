"""
Example usage of the data processing functions for the specific requirement:
"Load all .csv files from a user specified folder, discard the first column of data,
extract labels from the file name, concatenate all files into a single dataset
based on the year with each row representing an app and each column representing
API call information. Return all years of data from within the folder."
"""

from data import get_all_years_data
import pandas as pd


def main():
    """
    Example demonstrating the exact functionality required.

    This example shows how to:
    1. Specify a folder containing CSV files
    2. Load and process all files
    3. Get datasets organized by year
    4. Work with the processed data
    """

    # Step 1: Specify the folder containing CSV files
    folder_path = "input_data"  # User-specified folder

    print("="*70)
    print("DATA PROCESSING DEMONSTRATION")
    print("="*70)
    print(f"Processing CSV files from folder: {folder_path}")
    print()

    # Step 2: Get all years of data
    all_years_data = get_all_years_data(folder_path)

    print("="*70)
    print("PROCESSED DATA BY YEAR")
    print("="*70)

    # Step 3: Show what we got for each year
    for year, (X, y) in all_years_data.items():
        print(f"\nYear {year}:")
        print(f"  • Apps (rows): {X.shape[0]}")
        print(f"  • API calls (columns): {X.shape[1]}")
        print(f"  • Benign apps (label 0): {(y == 0).sum()}")
        print(f"  • Malware apps (label 1): {(y == 1).sum()}")

        # Show first few rows and columns
        print(f"  • First few rows of features (shape {X.shape}):")
        print(f"    {X.iloc[0, :5].to_dict()}")  # First row, first 5 columns

    print("\n" + "="*70)
    print("USING THE DATA FOR MACHINE LEARNING")
    print("="*70)

    # Example: Using the data with scikit-learn
    from sklearn.model_selection import train_test_split

    # Work with a specific year (e.g., 2014)
    X_2014, y_2014 = all_years_data['2014']

    print(f"\nExample with 2014 data:")
    print(f"  Splitting 2014 data (n={len(X_2014)}) into train/test...")

    X_train, X_test, y_train, y_test = train_test_split(
        X_2014, y_2014, test_size=0.2, random_state=42
    )

    print(f"  Training set: {X_train.shape[0]} apps")
    print(f"  Test set: {X_test.shape[0]} apps")

    # Example: Train a simple model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    print(f"\n  Training Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Test accuracy: {accuracy:.3f}")

    print("\n" + "="*70)
    print("COMBINING ALL YEARS")
    print("="*70)

    # Combine all years if needed
    combined_X = pd.concat([X for X, _ in all_years_data.values()], ignore_index=True)
    combined_y = pd.concat([y for _, y in all_years_data.values()], ignore_index=True)

    print(f"\nCombined dataset from all years:")
    print(f"  • Total apps: {combined_X.shape[0]}")
    print(f"  • Total features (API calls): {combined_X.shape[1]}")
    print(f"  • Total benign: {(combined_y == 0).sum()}")
    print(f"  • Total malware: {(combined_y == 1).sum()}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nThe program successfully processed data from {len(all_years_data)} years:")
    print(f"  Years: {', '.join(sorted(all_years_data.keys()))}")

    print("\nFor each year, we have:")
    print("  1. X: DataFrame with apps as rows and API calls as columns")
    print("  2. y: Series with binary labels (0=benign, 1=malware)")
    print("\nThe first column (apkname) was automatically discarded.")
    print("Labels were extracted from filenames (benign -> 0, malware -> 1).")


if __name__ == "__main__":
    main()