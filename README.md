# Malware Detection System

A comprehensive malware detection system with command-line interface, ensemble classifiers, and drift-aware retraining.

## Quick Start - CLI Usage

### Interactive Mode
```bash
python run_malware_detection.py
```

### Direct Commands

#### Load Data
```bash
python run_malware_detection.py load input_data
```

#### Train Models
```bash
# Train on single year
python run_malware_detection.py train single 2019

# Cross-time training
python run_malware_detection.py train cross 2014,2015,2016 2017
```

#### Test Models
```bash
python run_malware_detection.py test test_folder
```

#### Auto-Retraining
```bash
# Setup auto-retraining
python run_malware_detection.py auto setup
python run_malware_detection.py auto run
```

#### Utilities
```bash
# List trained models
python run_malware_detection.py list

# Show configuration
python run_malware_detection.py config

# Save/load models
python run_malware_detection.py save model_name model.pkl
python run_malware_detection.py load-model model.pkl
```

### Help Command
```bash
# Show available commands and usage
python run_malware_detection.py
```

**For complete CLI command reference**, see [Detailed CLI Documentation](#complete-malware-detection-training-system)

### Example Workflow
```bash
# 1. Load data
python run_malware_detection.py load input_data

# 2. Train on a single year
python run_malware_detection.py train single 2019

# 3. Create test data
python example_custom_test_data.py

# 4. Test the model
python run_malware_detection.py test test_data_demo
```

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

## Classifier Evaluation for Malware Detection

The project now includes a comprehensive `ClassifierEvaluator` class for evaluating malware detection classifiers with metrics in precedence order.

### Key Features

1. **Metrics in Precedence Order**: Accuracy > F1-Score > Precision > Recall
2. **Imbalanced Data Support**: Uses macro-averaging for F1, precision, recall
3. **Confusion Matrix**: Raw and normalized versions
4. **Classifier Comparison**: Systematic comparison with tie-breaking
5. **Visualization**: Optional confusion matrix plots

## Complete Malware Detection Training System

The project now includes a complete command-line interface for training, evaluating, and managing malware detection models with drift-aware retraining.

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 MALWARE DETECTION TRAINING SYSTEM            │
├─────────────────────────────────────────────────────────────┤
│  • Data Loading & Management                                │
│  • Single Year & Cross-Time Training                       │
│  • 3-Model Ensemble with Security-First Voting             │
│  • Drift-Aware Auto-Retraining                             │
│  • Model Testing & Evaluation                              │
│  • Configuration & Model Management                        │
└─────────────────────────────────────────────────────────────┘
```

### Quick Start

```bash
# Start interactive mode
python run_malware_detection.py

# Or use direct commands
python run_malware_detection.py load input_data
python run_malware_detection.py train single 2019
python run_malware_detection.py test test_folder
```

### Key Components

#### 1. Main CLI Interface (`malware_detection_cli.py`)
- **Data Management**: Load, analyze, and manage malware datasets
- **Model Training**: Single year and cross-time training
- **Auto-Retraining**: Drift-aware automatic model updates
- **Testing**: Test models on new datasets
- **Model Management**: Save, load, and list trained models

#### 2. 3-Model Ensemble (`SecurityFirstEnsemble` in `models.py`)
- **Model Trio**: LogisticRegression + RandomForest + GradientBoosting
- **Security-First Voting**: Defaults to malware when models disagree
- **Multiple Strategies**: Hard, soft, and stacked voting
- **Tie-Breaking**: Malware (security-first), confidence, or reject

#### 3. Drift-Aware Retraining (`retraining_system.py`)
- **Drift Detection**: Automatic concept drift monitoring
- **Performance Tracking**: Model performance degradation detection
- **Intelligent Retraining**: Retrain only when needed
- **Model Registry**: Versioned model storage and management

### Available Commands

#### Data Commands
```bash
# Load data from folder
python run_malware_detection.py load input_data

# Load specific folder
python run_malware_detection.py load /path/to/data
```

#### Training Commands
```bash
# Train on single year
python run_malware_detection.py train single 2019
python run_malware_detection.py train single 2019 --model individual

# Cross-time training
python run_malware_detection.py train cross 2014,2015,2016 2017
python run_malware_detection.py train cross 2014,2015 2016 --model individual
```

#### Auto-Retraining Commands
```bash
# Setup auto-retraining
python run_malware_detection.py auto setup
python run_malware_detection.py auto setup --start 2016 --end 2020

# Run auto-retraining
python run_malware_detection.py auto run
```

#### Testing Commands
```bash
# Test model on new data
python run_malware_detection.py test test_data_folder

# Test specific model
python run_malware_detection.py test test_data --model ensemble_2019_20250101_120000
```

#### Utility Commands
```bash
# List trained models
python run_malware_detection.py list

# Save model to file
python run_malware_detection.py save ensemble_2019_20250101_120000 my_model.pkl

# Load model from file
python run_malware_detection.py load-model my_model.pkl

# Show configuration
python run_malware_detection.py config
```

### Interactive Mode

Run without arguments to start interactive mode:

```bash
python run_malware_detection.py
```

Interactive mode provides a menu-driven interface with all the same functionality.

### Configuration

Configuration is stored in `md_config.json`:

```json
{
  "data_folder": "input_data",
  "model_registry": "model_registry",
  "drift_threshold": 0.3,
  "performance_threshold": 0.05,
  "default_voting": "hard",
  "default_tie_breaker": "malware",
  "models_folder": "saved_models",
  "test_results_folder": "test_results"
}
```

### Creating Test Data

Use `example_custom_test_data.py` to create test datasets:

```bash
# Create custom test data
python example_custom_test_data.py

# Then test the system
python run_malware_detection.py test test_folder_name
```

### Example Workflows

#### 1. Basic Training & Testing
```bash
# Load data
python run_malware_detection.py load input_data

# Train on 2019
python run_malware_detection.py train single 2019

# Create test data
python example_custom_test_data.py

# Test the model
python run_malware_detection.py test test_data_demo
```

#### 2. Cross-Time Analysis
```bash
# Train on 2014-2016, test on 2017
python run_malware_detection.py train cross 2014,2015,2016 2017

# Train on 2017-2019, test on 2020
python run_malware_detection.py train cross 2017,2018,2019 2020
```

#### 3. Auto-Retraining Pipeline
```bash
# Setup auto-retraining
python run_malware_detection.py auto setup

# Run progressive validation across all years
python run_malware_detection.py auto run
```

### File Structure

```
Project2/
├── malware_detection_cli.py      # Main CLI interface
├── run_malware_detection.py      # Wrapper script with help
├── example_custom_test_data.py   # Create test data
├── md_config.json               # System configuration
├── models.py                    # Models and ensemble
├── retraining_system.py         # Auto-retraining system
├── drift.py                     # Drift detection
├── data.py                      # Data loading
├── dataset.py                   # Dataset management
├── sample.py                    # Sample class
└── input_data/                  # Your malware dataset
```

### System Features

1. **Flexible Data Loading**: Loads standard CSV format with year labels
2. **Multiple Training Modes**: Single year, cross-time, and auto-retraining
3. **Ensemble Methods**: 3-model voting with security-first tie-breaking
4. **Drift Awareness**: Automatic retraining when concept drift detected
5. **Model Management**: Save, load, and version control for models
6. **Comprehensive Testing**: Test models on new datasets with full evaluation
7. **Configurable**: All parameters adjustable via configuration file

### Supported Data Format

The system expects CSV files named:
```
sampled_YYYY_benign_api.csv
sampled_YYYY_malware_api.csv
```

Where `YYYY` is the year (e.g., 2014, 2015, etc.).

### Performance Metrics

All evaluations include:
- **Accuracy**: Overall correct predictions
- **F1-Score**: Balance of precision and recall
- **Precision**: Of predicted malware, how many are actually malware
- **Recall**: Of actual malware, how many did we detect
- **Confusion Matrix**: Detailed breakdown of predictions
- **Model Agreement**: How often all 3 models agree (ensembles only)

### Next Steps

1. **Visualization**: Add performance charts and drift visualization
2. **Web Interface**: Build Flask/Dash web interface
3. **Real-time Monitoring**: Add streaming data support
4. **Advanced Ensembles**: Add more model types and voting methods
5. **Deployment**: Package for production deployment

### Import and Basic Usage

```python
from models import ClassifierEvaluator

# Create evaluator with predictions
evaluator = ClassifierEvaluator(
    classifier_name="MalwareDetectorV1",
    y_true=y_test,
    y_pred=y_pred
)

# Comprehensive evaluation
results = evaluator.evaluate(
    verbose=True,
    include_confusion_matrix=True,
    plot_confusion_matrix=False  # Set to True to plot
)

# Access metrics
metrics = evaluator.calculate_metrics()
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")

# Confusion matrix
cm = evaluator.get_confusion_matrix()
cm_norm = evaluator.get_confusion_matrix(normalize=True)
```

### Integrated Training Function

```python
from models import train_and_evaluate_classifiers

# Train and evaluate multiple classifiers
results = train_and_evaluate_classifiers(
    X_train, X_test, y_train, y_test,
    use_evaluator=True  # Enables ClassifierEvaluator
)

# Access results for each classifier
for name, result in results.items():
    metrics = result['evaluation']['metrics']
    print(f"{name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
```

### Quick Evaluation

```python
from models import quick_evaluate_classifier

results = quick_evaluate_classifier(
    classifier_name="MalwareScanner",
    y_true=y_test,
    y_pred=y_pred,
    plot_cm=True,
    normalize_cm=True
)
```

### Classifier Comparison

```python
# Compare two classifiers
evaluator1 = ClassifierEvaluator("ClassifierA", y_test, y_pred1)
evaluator2 = ClassifierEvaluator("ClassifierB", y_test, y_pred2)

comparison = evaluator1.compare_with_other(evaluator2, verbose=True)
print(f"Overall winner: {comparison['overall_winner']}")
```

### Why Precedence Order Matters for Malware Detection

Malware detection is typically imbalanced (few malware samples). The precedence order helps:

1. **Accuracy**: Overall performance, but can be misleading for imbalanced data
2. **F1-Score**: Balances precision and recall (important for security)
3. **Precision**: Minimizes false positives (user experience)
4. **Recall**: Maximizes malware detection (security)

### Example Test Scripts

- `test_evaluator.py`: Full demonstration of all features
- `example_evaluator_usage.py`: Practical usage examples
- `example_usage.py`: Original data processing examples

### Installation (Additional Dependencies)

```bash
# For basic functionality
pip install scikit-learn pandas numpy

# For visualization features
pip install matplotlib seaborn
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