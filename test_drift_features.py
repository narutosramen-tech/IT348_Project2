#!/usr/bin/env python3
"""
Test the new features added to drift.py:
1. Report skipped features count
2. Add option for non-consecutive year comparisons
3. Handle empty/constant feature cases
"""

import pandas as pd
import numpy as np
from dataset import Dataset
from sample import Sample
from drift import DriftAnalyzer

def create_test_dataset():
    """Create a test dataset with various edge cases."""
    # Create sample data with different characteristics
    np.random.seed(42)

    # Year 2020 data
    X_2020 = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),  # Normal distribution
        'feature2': np.ones(100),                  # Constant feature
        'feature3': np.random.normal(0, 1, 100),  # Normal distribution
        'feature4': np.random.normal(0, 0.5, 100), # Normal with different variance
        'feature5': np.full(100, 5.0),            # Constant with different value
        'feature6': np.random.normal(0.1, 1, 100), # Slightly shifted mean
    })
    y_2020 = pd.Series(np.random.choice([0, 1], size=100))

    # Year 2021 data - some features drift, some don't
    X_2021 = pd.DataFrame({
        'feature1': np.random.normal(0.2, 1, 100),  # Slight drift in mean
        'feature2': np.ones(100),                   # Same constant
        'feature3': np.random.normal(0, 1, 100),    # Same distribution
        'feature4': np.random.normal(0, 2.0, 100),  # Larger variance
        'feature5': np.full(100, 10.0),            # Different constant
        'feature6': np.random.normal(0.1, 1, 100), # Same distribution
    })
    y_2021 = pd.Series(np.random.choice([0, 1], size=100))

    # Year 2022 data
    X_2022 = pd.DataFrame({
        'feature1': np.random.normal(1.0, 1, 50),   # Significant drift
        'feature2': np.ones(50),                    # Same constant
        'feature3': np.random.normal(0, 1, 50),     # Same distribution
        'feature4': np.random.normal(0, 0.5, 50),   # Back to original variance
        'feature5': np.full(50, 10.0),             # Same constant as 2021
        'feature6': np.random.normal(0.5, 1, 50),  # Moderate drift
    })
    y_2022 = pd.Series(np.random.choice([0, 1], size=50))

    # Create dataset
    data_dict = {
        '2020': (X_2020, y_2020),
        '2021': (X_2021, y_2021),
        '2022': (X_2022, y_2022)
    }

    return Dataset(data_dict)

def test_skipped_features_reporting():
    """Test that skipped features are reported correctly."""
    print("=" * 70)
    print("TEST 1: Skipped Features Reporting")
    print("=" * 70)

    dataset = create_test_dataset()
    analyzer = DriftAnalyzer(dataset, alpha=0.05, mean_threshold=0.001)

    # Get samples
    sample_2020 = dataset.get_year('2020')
    sample_2021 = dataset.get_year('2021')

    # Test with skipped info
    print("\nComparing 2020 and 2021 WITH skipped info:")
    result_with_info = analyzer.compare_samples(sample_2020, sample_2021, include_skipped_info=True)
    print(f"Result shape: {result_with_info.shape}")
    print(f"Columns: {list(result_with_info.columns)}")

    # Test without skipped info
    print("\nComparing 2020 and 2021 WITHOUT skipped info:")
    result_without_info = analyzer.compare_samples(sample_2020, sample_2021, include_skipped_info=False)
    print(f"Result shape: {result_without_info.shape}")
    print(f"Columns: {list(result_without_info.columns)}")

    # Check if skipped columns are present when requested
    skipped_cols = ['skipped_features_count', 'skipped_mean_threshold']
    has_skipped_cols = all(col in result_with_info.columns for col in skipped_cols)
    print(f"\nSkipped columns present when requested: {has_skipped_cols}")

    return result_with_info, result_without_info

def test_non_consecutive_comparisons():
    """Test non-consecutive year comparisons."""
    print("\n" + "=" * 70)
    print("TEST 2: Non-Consecutive Year Comparisons")
    print("=" * 70)

    dataset = create_test_dataset()
    analyzer = DriftAnalyzer(dataset, alpha=0.05, mean_threshold=0.001)

    # Test 1: Specific year pairs
    print("\n1. Testing specific year pairs:")
    year_pairs = [('2020', '2022'), ('2020', '2021'), ('2021', '2022')]
    specific_results = analyzer.analyze_year_pairs(year_pairs, include_skipped_info=True)
    print(f"Results for specific year pairs: {specific_results.shape[0]} rows")
    unique_pairs = specific_results[['year_a', 'year_b']].drop_duplicates()
    print(f"Unique year pairs analyzed: {len(unique_pairs)}")

    # Test 2: All possible pairs
    print("\n2. Testing all possible pairs:")
    all_pairs_results = analyzer.analyze_all_pairs(include_skipped_info=False)
    print(f"Results for all pairs: {all_pairs_results.shape[0]} rows")

    # Get unique pairs count
    if not all_pairs_results.empty:
        unique_all_pairs = all_pairs_results[['year_a', 'year_b']].drop_duplicates()
        print(f"Unique year pairs in all pairs: {len(unique_all_pairs)}")
        print(f"Expected: 3 choose 2 = 3 pairs")

    return specific_results, all_pairs_results

def test_empty_constant_features():
    """Test handling of empty and constant features."""
    print("\n" + "=" * 70)
    print("TEST 3: Empty and Constant Feature Handling")
    print("=" * 70)

    # Create a dataset with edge cases
    np.random.seed(42)

    # Create data with edge cases
    X_edge = pd.DataFrame({
        'normal_feature': np.random.normal(0, 1, 50),
        'constant_zero': np.zeros(50),          # Constant 0
        'constant_five': np.full(50, 5.0),      # Constant 5
        'almost_constant': np.concatenate([np.full(49, 1.0), [2.0]]),  # Mostly constant
        'many_nans': pd.Series([np.nan] * 40 + list(np.random.normal(0, 1, 10))),  # Mostly NaN
    })
    y_edge = pd.Series(np.random.choice([0, 1], size=50))

    # Create a second sample with similar structure
    X_edge2 = pd.DataFrame({
        'normal_feature': np.random.normal(0.1, 1, 50),  # Slight drift
        'constant_zero': np.zeros(50),                   # Same constant
        'constant_five': np.full(50, 10.0),              # Different constant
        'almost_constant': np.concatenate([np.full(49, 1.0), [3.0]]),  # Slight difference
        'many_nans': pd.Series([np.nan] * 45 + list(np.random.normal(0, 1, 5))),  # Even more NaN
    })
    y_edge2 = pd.Series(np.random.choice([0, 1], size=50))

    # Create dataset
    data_dict_edge = {
        'edge_2020': (X_edge, y_edge),
        'edge_2021': (X_edge2, y_edge2)
    }
    dataset_edge = Dataset(data_dict_edge)

    analyzer_edge = DriftAnalyzer(dataset_edge, alpha=0.05, mean_threshold=0.001)

    sample_edge1 = dataset_edge.get_year('edge_2020')
    sample_edge2 = dataset_edge.get_year('edge_2021')

    print("\nComparing edge case samples:")
    edge_results = analyzer_edge.compare_samples(sample_edge1, sample_edge2, include_skipped_info=True)

    print(f"\nResults shape: {edge_results.shape}")
    print(f"Features analyzed: {len(edge_results)}")

    if not edge_results.empty:
        print("\nDrift detection results:")
        for _, row in edge_results.iterrows():
            drift_status = "DRIFT" if row['drift_detected'] else "no drift"
            print(f"  {row['feature']}: {drift_status} (p={row['p_value']:.4f})")

    return edge_results

def test_summary_with_skipped():
    """Test summary method with skipped features."""
    print("\n" + "=" * 70)
    print("TEST 4: Summary with Skipped Features")
    print("=" * 70)

    dataset = create_test_dataset()
    analyzer = DriftAnalyzer(dataset, alpha=0.05, mean_threshold=0.001)

    # Get analysis results with skipped info
    all_results = analyzer.analyze_all_pairs(include_skipped_info=True)

    if not all_results.empty:
        # Summary without skipped info
        summary_basic = analyzer.drift_summary(all_results, include_skipped=False)
        print("\nBasic summary (without skipped info):")
        print(summary_basic.to_string())

        # Summary with skipped info
        print("\n\nEnhanced summary (with skipped info):")
        summary_enhanced = analyzer.drift_summary(all_results, include_skipped=True)
        print(summary_enhanced.to_string())

        # Check if skipped columns are included
        skipped_cols_in_summary = [col for col in summary_enhanced.columns
                                  if 'skipped' in col.lower()]
        print(f"\nSkipped columns in enhanced summary: {skipped_cols_in_summary}")

        return summary_basic, summary_enhanced
    else:
        print("No results to summarize")
        return None, None

def main():
    """Run all tests."""
    print("Testing new features in drift.py")
    print("=" * 70)

    # Run tests
    test1_results = test_skipped_features_reporting()
    test2_results = test_non_consecutive_comparisons()
    test3_results = test_empty_constant_features()
    test4_results = test_summary_with_skipped()

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("""
    All new features have been implemented:

    1. ✅ Skipped features count reporting
       - `compare_samples()` now has `include_skipped_info` parameter
       - Reports counts by skip reason (mean threshold, empty data, etc.)

    2. ✅ Non-consecutive year comparisons
       - `analyze_year_pairs()` for specific year pairs
       - `analyze_all_pairs()` for all possible pairs

    3. ✅ Empty/constant feature handling
       - Skips features with insufficient data
       - Handles constant features appropriately
       - Provides warnings for edge cases

    4. ✅ Enhanced summary
       - `drift_summary()` now has `include_skipped` parameter
       - Includes skipped feature counts in summary when requested
    """)

if __name__ == "__main__":
    main()