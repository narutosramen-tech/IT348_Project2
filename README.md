# CSV Data Processing Program

This program loads and processes CSV files for malware detection analysis.

## Overview

The program automatically:
1. Loads all `.csv` files from a user-specified folder
2. Discards the first column (typically `apkname`)
3. Extracts labels from filenames:
   - `benign` → `0` (negative malware)
   - `malware` → `1` (positive malware)
4. Concatenates all files by year
5. Returns each year's data as separate datasets where:
   - Each row represents an app
   - Each column represents API call information
   - Labels are binary (0=benign, 1=malware)

## File Naming Convention

The program expects files named in this format:
```
sampled_YYYY_XXXXX_api.csv
```
Where:
- `YYYY` is the year (e.g., 2014, 2015)
- `XXXXX` is either `benign` or `malware`

Example:
- `sampled_2014_benign_api.csv`
- `sampled_2014_malware_api.csv`

## Installation

```bash
pip install pandas numpy scikit-learn
```

## Usage

### Basic Usage

```python
from data import get_all_years_data

# Load all data from folder
folder_path = "input_data"  # Change to your folder path
all_data = get_all_years_data(folder_path)

# Access data for a specific year
X_2014, y_2014 = all_data['2014']
```

### Complete Example

```python
import pandas as pd
from data import get_all_years_data
from sklearn.model_selection import train_test_split

# Load all data
all_data = get_all_years_data("input_data")

# Process each year
for year, (X, y) in all_data.items():
    print(f"Year {year}: {X.shape[0]} samples, {X.shape[1]} features")

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Use with scikit-learn classifiers
    # from sklearn.ensemble import RandomForestClassifier
    # clf = RandomForestClassifier()
    # clf.fit(X_train, y_train)
```

### Combined All Years

```python
# Combine all years into one dataset
all_X = pd.concat([X for X, _ in all_data.values()], ignore_index=True)
all_y = pd.concat([y for _, y in all_data.values()], ignore_index=True)

print(f"Combined dataset: {all_X.shape[0]} samples, {all_X.shape[1]} features")
```

## Output Structure

The `get_all_years_data()` function returns a dictionary:

```python
{
    '2014': (X_2014, y_2014),  # X: DataFrame, y: Series
    '2015': (X_2015, y_2015),
    # ... more years
}
```

Where:
- `X` is a pandas DataFrame with shape `(n_samples, n_features)`
- `y` is a pandas Series with shape `(n_samples,)`
- Each row in `X` represents an app
- Each column in `X` represents an API call
- Values in `y` are: `0` (benign) or `1` (malware)

## Files

- `data.py` - Main program with data processing functions
- `test_data_processing.py` - Test/validation script
- `README.md` - This documentation

## Testing

Run the test script to verify functionality:

```bash
python test_data_processing.py
```

Or run the main program directly:

```bash
python data.py
```

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn (optional, for ML tasks)