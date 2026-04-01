"""
    Author: Jon Bailey
    with help testing from Claude AI.
"""
import pandas as pd
import numpy as np
import os
import glob
import re
from typing import Dict, Tuple, List, cast
from scipy.stats import ks_2samp, kstest
from scipy.stats._stats_py import KstestResult
from dataset import Dataset
from sample import Sample


def load_and_process_csv_files(folder_path: str) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Load all .csv files from the specified folder, process them, and concatenate by year.

    Args:
        folder_path: Path to the folder containing CSV files

    Returns:
        Dictionary where keys are years (as strings) and values are tuples of (X, y)
        where X is the feature matrix (apps as rows, API calls as columns) and
        y is the target labels (0 for benign, 1 for malware)
    """

    # Get all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in folder: {folder_path}")

    # Dictionary to store dataframes by year
    year_data = {}

    # Process each CSV file
    for file_path in csv_files:
        # Extract filename without path
        filename = os.path.basename(file_path)

        # Parse filename to extract year and label
        match = re.match(r'sampled_(\d+)_(\w+)_api\.csv', filename)
        if not match:
            print(f"Skipping file with unexpected naming pattern: {filename}")
            continue

        year = match.group(1)
        label_str = match.group(2).lower()

        # Convert label to binary (benign=0, malware=1)
        if label_str == 'benign':
            label = 0  # negative (benign)
        elif label_str == 'malware':
            label = 1  # positive (malware)
        else:
            print(f"Skipping file with unknown label '{label_str}': {filename}")
            continue

        print(f"Processing {filename}: year={year}, label={label_str}->{label}")

        # Load the CSV file
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            continue

        # Check if dataframe is empty
        if df.empty:
            print(f"File {filename} is empty, skipping")
            continue

        # Drop the first column (typically 'apkname' or similar identifier)
        if len(df.columns) > 0:
            first_column = df.columns[0]
            df = df.drop(columns=[first_column])
            print(f"  Dropped first column: '{first_column}'")

        # Add label column (use assign to avoid fragmentation warning)
        df = df.assign(label=label)

        # Store in dictionary by year
        if year not in year_data:
            year_data[year] = []
        year_data[year].append(df)

    # Process each year: concatenate benign and malware data, separate features and labels
    result = {}

    for year, dfs in year_data.items():
        if len(dfs) == 0:
            print(f"Warning: No data for year {year}")
            continue

        # Concatenate all dataframes for this year
        combined_df = pd.concat(dfs, ignore_index=True)

        # Separate features (X) and labels (y)
        if 'label' not in combined_df.columns:
            print(f"Warning: No label column found for year {year}")
            continue

        y = combined_df['label'].copy()
        X = combined_df.drop(columns=['label'])

        # Ensure X has proper numeric dtype
        X = X.apply(pd.to_numeric, errors='coerce')

        # Print some info
        print(f"\nYear {year}:")
        print(f"  Total samples: {len(X)}")
        print(f"  Benign samples (0): {(y == 0).sum()}")
        print(f"  Malware samples (1): {(y == 1).sum()}")
        print(f"  Feature dimensions: {X.shape[1]} API calls")

        result[year] = (X, y)

    # Sort years in ascending order
    sorted_result = {year: result[year] for year in sorted(result.keys())}

    print(f"\nProcessed {len(sorted_result)} year(s): {list(sorted_result.keys())}")
    return sorted_result


def get_all_years_data(folder_path: str) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Wrapper function that returns all years of data from within the folder.

    Args:
        folder_path: Path to the folder containing CSV files

    Returns:
        Dictionary where keys are years (as strings) and values are tuples of (X, y)
    """
    return load_and_process_csv_files(folder_path)


def print_data_summary(data_dict: Dict[str, Tuple[pd.DataFrame, pd.Series]]):
    """Print summary information about the processed data."""
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)

    for year, (X, y) in data_dict.items():
        print(f"\nYear {year}:")
        print(f"  Samples: {len(X)}")
        print(f"  Features: {X.shape[1]} API calls")
        print(f"  Benign (0): {(y == 0).sum()} samples")
        print(f"  Malware (1): {(y == 1).sum()} samples")
        print(f"  Class balance: {(y == 0).sum()/len(y)*100:.1f}% benign, {(y == 1).sum()/len(y)*100:.1f}% malware")

    print("\n" + "="*60)

def load_single_file_data(file_path: str) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Load a single .csv file, process it, and return it in the (X, y) format.

    Args:
        file_path: Path to the specific CSV file
    """

    if not os.path.isfile(file_path):
        raise ValueError(f"File not found: {file_path}")
    
    filename = os.path.basename(file_path)

    match = re.match(r'sampled_(\d+)_(\w+)_api\.csv', filename)

    if match:
        year = match.group(1)
        label_str = match.group(2).lower()
        label = 1 if label_str == 'malware' else 0
    else:
        year = "Unkown"
        label = None
    
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError(f'File {filename} is empty')
        
        # Drop the first column (identifier)
        if len(df.columns) > 0:
            df = df.drop(columns = [df.columns[0]])

        if label is not None:
            df = df.assign(label = label)
        elif 'label' not in df.columns:
            raise ValueError(f'No label found in filename or CSV content for {filename}')
        
        y = df['label'].copy()
        X = df.drop(columns = ['label']).apply(pd.to_numeric, errors = 'coerce')

        print(f"Processed single file {filename}: year = {year}, samples = {len(X)}")
        return {year: (X, y)}
    except Exception as e:
        print(f"Error reading single file {filename}: {e}")
        raise

# Example usage
if __name__ == "__main__":
    # Define the folder path (adjust as needed)
    folder_path = "input_data"

    try:
        # Load and process all CSV files
        all_data = get_all_years_data(folder_path)

        # Print summary
        print_data_summary(all_data)

        # Access individual years
        for year, (X, y) in all_data.items():
            print(f"\nYear {year} data available:")
            print(f"  X shape: {X.shape}")
            print(f"  y shape: {y.shape}")

            # You can now use X and y for machine learning tasks
            # Example: from sklearn.model_selection import train_test_split
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
