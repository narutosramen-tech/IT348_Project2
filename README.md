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

## Architecture & Classes

The system is organized into several modular components:

### Core Data Structures

#### 1. `Sample` Class ([sample.py](sample.py))
Represents a dataset sample for a specific year.

```python
from sample import Sample

# Create a sample
sample = Sample(
    year="2014",
    features=X_dataframe,  # pandas DataFrame
    labels=y_series         # pandas Series
)

# Properties
print(f"Year: {sample.year}")
print(f"Has labels: {sample.has_labels}")
print(f"Number of samples: {sample.num_samples}")
print(f"Number of features: {sample.num_features}")
```

#### 2. `Dataset` Class ([dataset.py](dataset.py))
Holds a collection of `Sample` objects and provides utility methods.

```python
from dataset import Dataset
from data import get_all_years_data

# Create dataset from processed data
all_data = get_all_years_data("input_data")
dataset = Dataset(all_data)

# Methods
print(f"Dataset contains {len(dataset)} years")
print(f"Available years: {dataset.years()}")
sample_2014 = dataset.get_year("2014")
dataset.summary()  # Print detailed summary
```

#### 3. `DataPreprocessor` Class ([data_preprocessor.py](data_preprocessor.py))
Handles feature scaling, dataset splitting, and preprocessing operations.

```python
from data_preprocessor import DataPreprocessor

# Initialize with dataset
preprocessor = DataPreprocessor(dataset)

# Get combined dataset
X, y = preprocessor.get_full_dataset()

# Random split (train/test)
X_train, X_test, y_train, y_test = preprocessor.random_split(test_size=0.2)

# Temporal split (train on earlier years, test on later years)
X_train, X_test, y_train, y_test = preprocessor.temporal_split(
    train_years=["2014", "2015"],
    test_years=["2016", "2017"]
)

# Feature scaling
X_train_scaled, X_test_scaled, scaler = preprocessor.scale_features(
    X_train, X_test, method="standard"  # or "minmax"
)
```

### Data Processing Functions

#### 4. `data.py`
Contains functions for loading and processing CSV files.

Key functions:
- `load_and_process_csv_files(folder_path)` - Core processing function
- `get_all_years_data(folder_path)` - Wrapper function
- `print_data_summary(data_dict)` - Utility for data inspection

### Model Training

#### 5. `models.py`
Contains model training and evaluation utilities.

```python
from models import train_and_evaluate_classifiers

# Train and evaluate classifiers
results = train_and_evaluate_classifiers(X_train, X_test, y_train, y_test)

# Access results
lr_results = results["LogisticRegression"]
rf_results = results["RandomForest"]
print(f"Logistic Regression Accuracy: {lr_results['accuracy']:.4f}")
```

### Examples & Testing

#### 6. `example_usage.py`
Complete example demonstrating the entire workflow.

#### 7. `test_data_processing.py`
Test script for validating data processing functionality.

#### 8. `drift.py` and `test_drift_features.py`
Concept drift detection and analysis tools.

## Complete Workflow Example

```python
from data import get_all_years_data
from dataset import Dataset
from data_preprocessor import DataPreprocessor
from models import train_and_evaluate_classifiers

# 1. Load and process data
raw_data = get_all_years_data("input_data")

# 2. Create structured dataset
dataset = Dataset(raw_data)
dataset.summary()

# 3. Initialize preprocessor
preprocessor = DataPreprocessor(dataset)

# Option A: Random split
X_train, X_test, y_train, y_test = preprocessor.random_split(test_size=0.2)

# Option B: Temporal split (train on earlier years, test on later)
# X_train, X_test, y_train, y_test = preprocessor.temporal_split(
#     train_years=["2014", "2015"],
#     test_years=["2016", "2017"]
# )

# 4. Scale features
X_train_scaled, X_test_scaled, scaler = preprocessor.scale_features(
    X_train, X_test, method="standard"
)

# 5. Train and evaluate models
results = train_and_evaluate_classifiers(X_train_scaled, X_test_scaled,
                                       y_train, y_test)

print(f"Random Forest Accuracy: {results['RandomForest']['accuracy']:.4f}")
```

### Advanced Usage Examples

**1. Working with Sample Objects**
```python
from sample import Sample
import pandas as pd

# Create a sample manually
X_sample = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['f1', 'f2', 'f3'])
y_sample = pd.Series([0, 1])
sample = Sample(year="2020", features=X_sample, labels=y_sample)

print(f"Sample: {sample.num_samples} samples, {sample.num_features} features")
```

**2. Combined Dataset Analysis**
```python
# Combine all years for global analysis
X_all, y_all = preprocessor.get_full_dataset()
print(f"Total: {X_all.shape[0]} samples, {X_all.shape[1]} features")
print(f"Class balance: {(y_all == 0).sum()} benign, {(y_all == 1).sum()} malware")
```

**3. Year-by-Year Processing**
```python
# Process each year separately
for year in dataset.years():
    sample = dataset.get_year(year)
    if sample.has_labels:
        print(f"Year {year}: {sample.num_samples} samples")
        benign = (sample.labels == 0).sum()
        malware = (sample.labels == 1).sum()
        print(f"  Benign: {benign}, Malware: {malware}")
```

## Files

The project contains the following files:

### Core Modules
- `data.py` - Main program with data processing functions
- `dataset.py` - `Dataset` class for managing collections of samples
- `sample.py` - `Sample` class representing individual year data
- `data_preprocessor.py` - `DataPreprocessor` class for data splitting and scaling
- `models.py` - Model training and evaluation utilities

### Examples and Testing
- `example_usage.py` - Complete workflow example
- `test_data_processing.py` - Test/validation script
- `test_drift_features.py` - Concept drift testing
- `drift.py` - Concept drift analysis tools

### Documentation
- `README.md` - This documentation
- `requirements.txt` - Python dependencies (if exists)

## Testing and Usage

### Quick Start
```bash
# Run the example usage script
python example_usage.py

# Test basic data processing
python test_data_processing.py

# Run concept drift tests (optional)
python test_drift_features.py
```

### Run Individual Components
```bash
# Load and process data
python data.py

# Example with custom folder
python -c "
from data import get_all_years_data
data = get_all_years_data('input_data')
print(f'Processed {len(data)} years')
"
```

### Complete Pipeline Test
```bash
# Test the full pipeline using example data
python -c "
from data import get_all_years_data
from dataset import Dataset
from data_preprocessor import DataPreprocessor

# Load data
raw_data = get_all_years_data('input_data')
dataset = Dataset(raw_data)

print(f'Dataset ready: {len(dataset)} years')
dataset.summary()
"
```

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn (optional, for ML tasks)