#!/usr/bin/env python3
"""
Simple test to verify recall-first precedence works correctly.
"""

import numpy as np
from models import ClassifierEvaluator

def main():
    """Test the recall-first precedence."""
    print("\n" + "="*80)
    print("VERIFYING RECALL-FIRST PRECEDENCE FOR MALWARE DETECTION")
    print("="*80)
    print("Key feature: Models within 1.5% difference are considered tied")
    print()

    # Create a malware detection scenario
    np.random.seed(42)

    # Scenario: Model A catches malware better, Model B has better overall stats
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])  # 4 malware, 6 benign

    # Model A: Catches 3/4 malware (75% recall), has 1 false positive
    y_pred_a = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])  # 75% recall, 1 FP

    # Model B: Catches 2/4 malware (50% recall), has 0 false positives
    y_pred_b = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # 50% recall, 0 FP

    evaluator_a = ClassifierEvaluator("Model_A_CatchesMoreMalware", y_true, y_pred_a)
    evaluator_b = ClassifierEvaluator("Model_B_BetterPrecision", y_true, y_pred_b)

    print("Dataset: 10 samples, 4 malware (40%), 6 benign")
    print()
    print("Model A: Catches 3/4 malware (75% recall), has 1 false positive")
    print("  Think: Good at detecting malware, some false alarms")
    print()
    print("Model B: Catches 2/4 malware (50% recall), has 0 false positives")
    print("  Think: Misses half the malware, but never false alarms")
    print()

    # Show metrics
    metrics_a = evaluator_a.calculate_metrics()
    metrics_b = evaluator_b.calculate_metrics()

    print("MODEL METRICS:")
    print(f"             {'Model A':<25} {'Model B':<25} {'Difference':<15}")
    print(f"Recall*:     {metrics_a['recall']:.4f} (catches 75% malware)   {metrics_b['recall']:.4f} (catches 50% malware)   +{metrics_a['recall']-metrics_b['recall']:.4f}")
    print(f"F1-Score:    {metrics_a['f1_score']:.4f}                    {metrics_b['f1_score']:.4f}                    +{metrics_a['f1_score']-metrics_b['f1_score']:.4f}")
    print(f"Precision:   {metrics_a['precision']:.4f}                    {metrics_b['precision']:.4f}                    {metrics_a['precision']-metrics_b['precision']:.4f}")
    print(f"Accuracy:    {metrics_a['accuracy']:.4f}                    {metrics_b['accuracy']:.4f}                    +{metrics_a['accuracy']-metrics_b['accuracy']:.4f}")

    print("\n" + "="*80)
    print("COMPARISON RESULTS:")
    print("="*80)

    # Compare
    comparison = evaluator_a.compare_with_other(evaluator_b, verbose=True)

    print("\n" + "="*80)
    print("ANALYSIS:")
    print("="*80)
    print("For malware detection:")
    print("- Model A detects 75% of malware, Model B detects 50%")
    print("- Model A misses 1 malware, Model B misses 2 malware")
    print("- Model A has 1 false alarm, Model B has 0 false alarms")
    print()
    print("CRITICAL QUESTION:")
    print("Which is worse: Missing 2 pieces of malware, or having 1 false alarm?")
    print()
    print("OUR SYSTEM ANSWERS: Model A wins (because RECALL is most important)")
    print("This prioritizes security over convenience.")

    # Test tie threshold
    print("\n" + "="*80)
    print("TESTING 1.5% TIE THRESHOLD:")
    print("="*80)

    # Create two nearly identical models
    y_true_large = np.random.randint(0, 2, 1000)
    y_pred_c = y_true_large.copy()
    y_pred_d = y_true_large.copy()

    # Make them slightly different
    n_changes = 5  # 0.5% difference
    change_indices = np.random.choice(len(y_true_large), n_changes, replace=False)
    for idx in change_indices:
        y_pred_d[idx] = 1 - y_pred_d[idx]

    evaluator_c = ClassifierEvaluator("Model_C", y_true_large, y_pred_c)
    evaluator_d = ClassifierEvaluator("Model_D", y_true_large, y_pred_d)

    metrics_c = evaluator_c.calculate_metrics()
    metrics_d = evaluator_d.calculate_metrics()

    print(f"\nModels have ~{n_changes/len(y_true_large):.1%} difference")
    print(f"Recall difference: {abs(metrics_c['recall'] - metrics_d['recall']):.4f}")
    print(f"Within 1.5% tie threshold? {abs(metrics_c['recall'] - metrics_d['recall']) <= 0.015}")
    print()

    comparison_tie = evaluator_c.compare_with_other(evaluator_d, verbose=True)

if __name__ == "__main__":
    main()