#!/usr/bin/env python3
"""
Example usage of the ClassifierEvaluator for malware detection scenarios.
Shows how to evaluate classifiers with metrics in precedence order.
"""

import numpy as np
import pandas as pd

# Import the evaluator from our models module
from models import ClassifierEvaluator, train_and_evaluate_classifiers, quick_evaluate_classifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def example_basic_usage():
    """Basic usage example with synthetic data."""
    print("=" * 70)
    print("BASIC USAGE EXAMPLE")
    print("=" * 70)

    # Create imbalanced synthetic data (similar to malware detection)
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        weights=[0.9, 0.1],  # 90% benign, 10% malware
        random_state=42
    )

    # Convert to DataFrame/Series for compatibility
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_s = pd.Series(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_s, test_size=0.3, stratify=y, random_state=42
    )

    print(f"Dataset: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    print(f"Class distribution: {sum(y == 0)} benign, {sum(y == 1)} malware")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    # Use the integrated training function with evaluator
    print("\n1. Training and evaluating classifiers with integrated function:")
    results = train_and_evaluate_classifiers(
        X_train, X_test, y_train, y_test,
        use_evaluator=True
    )

    # Access results for a specific classifier
    print("\n2. Accessing detailed results from LogisticRegression:")
    lr_results = results.get("LogisticRegression", {})
    if 'evaluation' in lr_results:
        metrics = lr_results['evaluation']['metrics']
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")

def example_direct_evaluator_usage():
    """Example using ClassifierEvaluator directly."""
    print("\n\n" + "=" * 70)
    print("DIRECT EVALUATOR USAGE")
    print("=" * 70)

    # Simulate predictions from a classifier
    y_true = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 1])  # Mostly benign
    y_pred = np.array([0, 0, 0, 0, 0, 1, 0, 0, 1, 1])  # Some errors

    print(f"True labels:  {y_true}")
    print(f"Pred labels:  {y_pred}")

    # Create evaluator
    evaluator = ClassifierEvaluator(
        classifier_name="SampleMalwareDetector",
        y_true=y_true,
        y_pred=y_pred
    )

    # Get metrics
    metrics = evaluator.calculate_metrics()
    print(f"\nMetrics calculated:")
    for metric, value in metrics.items():
        print(f"  {metric:12s}: {value:.4f}")

    # Get confusion matrix
    cm = evaluator.get_confusion_matrix()
    print(f"\nConfusion Matrix:\n{cm}")

    # Get normalized confusion matrix
    cm_norm = evaluator.get_confusion_matrix(normalize=True)
    print(f"\nNormalized Confusion Matrix:\n{cm_norm}")

    # Comprehensive evaluation
    print(f"\nComprehensive evaluation report:")
    results = evaluator.evaluate(verbose=True)

def example_comparison_scenario():
    """Example comparing two classifiers for malware detection."""
    print("\n\n" + "=" * 70)
    print("CLASSIFIER COMPARISON SCENARIO")
    print("=" * 70)

    print("\nScenario: Comparing two malware detection algorithms")
    print("Malware detection is imbalanced (few malware samples)")
    print("Metrics in precedence order: Accuracy > F1 > Precision > Recall\n")

    # Create two evaluators with different performance profiles
    # Simulate predictions from two different classifiers

    # Classifier A: High accuracy, moderate F1
    y_true = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    y_pred_a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])  # Misses one malware
    y_pred_b = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1])  # One false positive

    evaluator_a = ClassifierEvaluator("ClassifierA", y_true, y_pred_a)
    evaluator_b = ClassifierEvaluator("ClassifierB", y_true, y_pred_b)

    print("Individual evaluations:")
    print("-" * 40)
    results_a = evaluator_a.evaluate(verbose=False)
    results_b = evaluator_b.evaluate(verbose=False)

    print(f"Classifier A:")
    for metric, value in results_a['metrics'].items():
        print(f"  {metric:12s}: {value:.4f}")

    print(f"\nClassifier B:")
    for metric, value in results_b['metrics'].items():
        print(f"  {metric:12s}: {value:.4f}")

    print(f"\nComparison (by precedence order):")
    comparison = evaluator_a.compare_with_other(evaluator_b, verbose=True)

    print(f"\nInterpretation:")
    print(f"Classifier A wins on: Accuracy (higher precedence)")
    print(f"Classifier B wins on: F1, Precision, Recall")
    print(f"Overall winner (by precedence): {comparison['overall_winner']}")

def example_quick_evaluation():
    """Example using the quick evaluation function."""
    print("\n\n" + "=" * 70)
    print("QUICK EVALUATION FUNCTION")
    print("=" * 70)

    # Simple test data
    y_true = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1])  # Some errors

    print(f"Running quick evaluation...")
    results = quick_evaluate_classifier(
        classifier_name="MalwareScannerV2",
        y_true=y_true,
        y_pred=y_pred,
        plot_cm=False,  # Set to True to plot confusion matrix
        normalize_cm=False
    )

    print(f"\nResults available in dictionary:")
    for key in results.keys():
        print(f"  {key}")

    if 'metrics' in results:
        print(f"\nMetrics:")
        for metric, value in results['metrics'].items():
            print(f"  {metric:12s}: {value:.4f}")

def example_for_malware_detection():
    """Specific example for malware detection context."""
    print("\n\n" + "=" * 70)
    print("MALWARE DETECTION CONTEXT")
    print("=" * 70)

    print("\nWhy metrics precedence matters for malware detection:")
    print("1. Accuracy alone can be misleading:")
    print("   - A model that always predicts 'benign' would have 90% accuracy")
    print("   - But it would miss all malware!")
    print("\n2. F1-score balances precision and recall:")
    print("   - Precision: Of predicted malware, how many are actually malware?")
    print("   - Recall: Of actual malware, how many did we detect?")
    print("\n3. Confusion matrix shows specific error types:")
    print("   - False positives: Benign apps flagged as malware (annoying)")
    print("   - False negatives: Malware apps missed (dangerous)")

    print("\nExample metrics interpretation:")
    print("- High accuracy + low F1: Model is biased toward majority class")
    print("- High precision: Few false positives (good for user experience)")
    print("- High recall: Most malware detected (good for security)")

def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("CLASSIFIER EVALUATOR EXAMPLES")
    print("For malware detection and imbalanced classification")
    print("=" * 70)

    example_basic_usage()
    example_direct_evaluator_usage()
    example_comparison_scenario()
    example_quick_evaluation()
    example_for_malware_detection()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nThe ClassifierEvaluator provides:")
    print("1. Automatic calculation of Accuracy, F1, Precision, Recall")
    print("2. Metrics in precedence order for decision making")
    print("3. Confusion matrix generation (raw and normalized)")
    print("4. Classifier comparison with tie-breaking by precedence")
    print("5. Integration with existing training workflows")
    print("\nUsage patterns:")
    print("- Use train_and_evaluate_classifiers() for integrated training")
    print("- Use ClassifierEvaluator directly for custom evaluation")
    print("- Use quick_evaluate_classifier() for simple cases")
    print("- Use compare_with_other() to choose between classifiers")

if __name__ == "__main__":
    main()