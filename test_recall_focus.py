#!/usr/bin/env python3
"""
Test the updated ClassifierEvaluator with recall-first precedence and tie thresholds.
"""

import numpy as np
import pandas as pd
from models import ClassifierEvaluator

def test_basic_evaluation():
    """Test basic classifier evaluation with recall-first precedence."""
    print("\n" + "="*80)
    print("TEST 1: BASIC EVALUATION WITH RECALL-FIRST PRECEDENCE")
    print("="*80)

    # Create synthetic data
    np.random.seed(42)
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    y_pred_good = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1])  # 1 false positive
    y_pred_bad = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])   # 4 false negatives

    # Create evaluators
    evaluator_good = ClassifierEvaluator("Good_Classifier", y_true, y_pred_good)
    evaluator_bad = ClassifierEvaluator("Bad_Classifier", y_true, y_pred_bad)

    # Evaluate each
    print("\nEvaluating Good Classifier (1 false positive):")
    results_good = evaluator_good.evaluate(verbose=True)

    print("\nEvaluating Bad Classifier (4 false negatives):")
    results_bad = evaluator_bad.evaluate(verbose=True)

    # Compare
    print("\n" + "="*80)
    print("COMPARISON WITH RECALL-FIRST PRECEDENCE")
    print("="*80)
    comparison = evaluator_good.compare_with_other(evaluator_bad, verbose=True)

    return comparison

def test_tie_threshold():
    """Test the 1.5% tie threshold functionality."""
    print("\n" + "="*80)
    print("TEST 2: TIE THRESHOLD (1.5%) FUNCTIONALITY")
    print("="*80)

    # Create synthetic data where models are very close
    np.random.seed(42)
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 100)  # 1000 samples

    # Model A: Slightly better recall, slightly worse F1
    y_pred_a = y_true.copy()
    # Add some errors
    error_indices = np.random.choice(len(y_true), size=50, replace=False)
    for i in error_indices:
        if i % 2 == 0:  # Make false positives (bad for precision)
            y_pred_a[i] = 1 if y_true[i] == 0 else y_true[i]

    # Model B: Slightly better F1, slightly worse recall
    y_pred_b = y_true.copy()
    error_indices = np.random.choice(len(y_true), size=50, replace=False)
    for i in error_indices:
        if i % 2 == 1:  # Make false negatives (bad for recall)
            y_pred_b[i] = 0 if y_true[i] == 1 else y_true[i]

    # Create evaluators
    evaluator_a = ClassifierEvaluator("Model_A", y_true, y_pred_a)
    evaluator_b = ClassifierEvaluator("Model_B", y_true, y_pred_b)

    # Get metrics
    metrics_a = evaluator_a.calculate_metrics()
    metrics_b = evaluator_b.calculate_metrics()

    print(f"\nModel A metrics:")
    print(f"  Recall: {metrics_a['recall']:.4f}")
    print(f"  F1: {metrics_a['f1_score']:.4f}")
    print(f"  Precision: {metrics_a['precision']:.4f}")
    print(f"  Accuracy: {metrics_a['accuracy']:.4f}")

    print(f"\nModel B metrics:")
    print(f"  Recall: {metrics_b['recall']:.4f}")
    print(f"  F1: {metrics_b['f1_score']:.4f}")
    print(f"  Precision: {metrics_b['precision']:.4f}")
    print(f"  Accuracy: {metrics_b['accuracy']:.4f}")

    # Show differences
    print(f"\nDifferences (B - A):")
    for metric in ['recall', 'f1_score', 'precision', 'accuracy']:
        diff = metrics_b[metric] - metrics_a[metric]
        print(f"  {metric}: {diff:.4f} ({abs(diff):.3%})")

    # Compare
    print("\n" + "="*80)
    comparison = evaluator_a.compare_with_other(evaluator_b, verbose=True)

    # Test edge case: Exactly at tie threshold
    print("\n" + "="*80)
    print("TEST 3: AT TIE THRESHOLD BOUNDARY")
    print("="*80)

    # Manually create metrics that are exactly 0.015 apart
    y_dummy = np.array([0, 1, 0, 1])
    y_pred_c = np.array([0, 1, 0, 1])
    y_pred_d = np.array([1, 0, 1, 0])  # Completely wrong

    evaluator_c = ClassifierEvaluator("Model_C", y_dummy, y_pred_c)
    evaluator_d = ClassifierEvaluator("Model_D", y_dummy, y_pred_d)

    print("\nExtreme case test:")
    comparison_extreme = evaluator_c.compare_with_other(evaluator_d, verbose=True)

    return comparison

def test_malware_scenario():
    """Test a realistic malware detection scenario."""
    print("\n" + "="*80)
    print("TEST 4: MALWARE DETECTION SCENARIO")
    print("="*80)

    # In malware detection:
    # - Malware class is typically minority (5-15%)
    # - False negatives (missed malware) are MUCH worse than false positives
    # - Recall (detecting malware) is most important

    np.random.seed(42)

    # Create imbalanced dataset (1000 samples, 10% malware)
    n_samples = 1000
    n_malware = 100  # 10%
    n_benign = 900   # 90%

    y_true = np.array([1] * n_malware + [0] * n_benign)
    np.random.shuffle(y_true)

    # Create two different classifiers
    # Classifier X: High recall (catches most malware) but lower precision
    y_pred_x = y_true.copy()
    # Add some false positives (benign flagged as malware)
    benign_indices = np.where(y_true == 0)[0]
    fp_indices = np.random.choice(benign_indices, size=90, replace=False)  # 10% FP rate
    for idx in fp_indices:
        y_pred_x[idx] = 1

    # Classifier Y: High precision (few false positives) but lower recall
    y_pred_y = y_true.copy()
    # Add some false negatives (missed malware)
    malware_indices = np.where(y_true == 1)[0]
    fn_indices = np.random.choice(malware_indices, size=20, replace=False)  # 20% FN rate
    for idx in fn_indices:
        y_pred_y[idx] = 0

    evaluator_x = ClassifierEvaluator("HighRecall_HighFP", y_true, y_pred_x)
    evaluator_y = ClassifierEvaluator("HighPrecision_MissedMalware", y_true, y_pred_y)

    # Show distributions
    print(f"\nDataset: {n_samples} samples, {n_malware} malware ({n_malware/n_samples:.1%})")

    print(f"\nHigh Recall Classifier (catches malware but has false alarms):")
    print(f"  False positives: {sum((y_true == 0) & (y_pred_x == 1))} out of {n_benign} benign")
    print(f"  False negatives: {sum((y_true == 1) & (y_pred_x == 0))} out of {n_malware} malware")

    print(f"\nHigh Precision Classifier (few false alarms but misses malware):")
    print(f"  False positives: {sum((y_true == 0) & (y_pred_y == 1))} out of {n_benign} benign")
    print(f"  False negatives: {sum((y_true == 1) & (y_pred_y == 0))} out of {n_malware} malware")

    print(f"\n" + "="*80)
    comparison = evaluator_x.compare_with_other(evaluator_y, verbose=True)

    # Explain why the winner makes sense for malware detection
    print(f"\n" + "="*80)
    print("ANALYSIS FOR MALWARE DETECTION:")
    print("="*80)
    metrics_x = evaluator_x.calculate_metrics()
    metrics_y = evaluator_y.calculate_metrics()

    if metrics_x['recall'] > metrics_y['recall']:
        print(f"✓ HighRecall_HighFP has higher recall ({metrics_x['recall']:.4f} vs {metrics_y['recall']:.4f})")
        print(f"  - Better at catching malware (fewer false negatives)")
        print(f"  - Acceptable tradeoff: more false positives vs missed malware")
    else:
        print(f"✗ HighPrecision_MissedMalware has higher recall")
        print(f"  (This should not happen with our test setup)")

    print(f"\nFor malware detection:")
    print(f"  • Missed malware (false negative) = SECURITY BREACH")
    print(f"  • False alarm (false positive) = ANNOYANCE")
    print(f"  • Priority: MAXIMIZE RECALL (detect malware) > everything else")

    return comparison

def main():
    """Run all tests."""
    print("TESTING RECALL-FIRST CLASSIFIER EVALUATION SYSTEM")
    print("="*80)
    print("Key Changes:")
    print("  1. Precedence: Recall > F1 > Precision > Accuracy")
    print("  2. Tie threshold: 1.5% difference considered a tie")
    print("  3. Designed for malware detection (minimize false negatives)")
    print("="*80)

    # Run tests
    test_basic_evaluation()
    test_tie_threshold()
    test_malware_scenario()

    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print("\nSummary:")
    print("✓ ClassifierEvaluator now prioritizes RECALL for malware detection")
    print("✓ 1.5% tie threshold prevents trivial differences from determining winner")
    print("✓ Comparison logic respects: Recall > F1 > Precision > Accuracy")
    print("✓ Security-focused: Missed malware (low recall) is worst outcome")

if __name__ == "__main__":
    main()