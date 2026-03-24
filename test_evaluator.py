#!/usr/bin/env python3
"""
Test script for the ClassifierEvaluator class.
Demonstrates usage and shows the metrics in order of precedence.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Import our new evaluator
from models import ClassifierEvaluator, train_and_evaluate_classifiers


def generate_imbalanced_data(n_samples=1000, n_features=20):
    """Generate imbalanced classification data similar to malware detection."""
    # Create imbalanced data (90% class 0, 10% class 1)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=2,
        weights=[0.9, 0.1],  # Imbalanced classes
        random_state=42
    )

    return X, y


def demo_single_evaluator():
    """Demonstrate basic usage of ClassifierEvaluator."""
    print("=" * 60)
    print("DEMONSTRATION 1: SINGLE CLASSIFIER EVALUATION")
    print("=" * 60)

    # Generate sample data
    X, y = generate_imbalanced_data(n_samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train a simple classifier
    print("\nTraining Logistic Regression classifier...")
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Create evaluator
    evaluator = ClassifierEvaluator(
        classifier_name="LogisticRegression Demo",
        y_true=y_test,
        y_pred=y_pred
    )

    # Evaluate with confusion matrix
    print("\nEvaluating classifier performance...")
    results = evaluator.evaluate(
        verbose=True,
        include_confusion_matrix=True,
        plot_confusion_matrix=False  # Set to True to see the plot
    )

    # Demonstrate confusion matrix methods
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX DEMONSTRATION")
    print("=" * 60)

    cm = evaluator.get_confusion_matrix()
    print(f"\nRaw confusion matrix:\n{cm}")

    cm_normalized = evaluator.get_confusion_matrix(normalize=True)
    print(f"\nNormalized confusion matrix:\n{cm_normalized}")

    return evaluator, results


def demo_comparison():
    """Demonstrate comparison of two classifiers."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION 2: CLASSIFIER COMPARISON")
    print("=" * 60)

    # Generate sample data
    X, y = generate_imbalanced_data(n_samples=1500)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train two different classifiers
    print("\nTraining classifiers for comparison...")

    # Classifier 1: Logistic Regression
    lr_clf = LogisticRegression(max_iter=1000, random_state=42)
    lr_clf.fit(X_train, y_train)
    lr_pred = lr_clf.predict(X_test)

    # Classifier 2: Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    rf_pred = rf_clf.predict(X_test)

    # Create evaluators
    lr_evaluator = ClassifierEvaluator("LogisticRegression", y_test, lr_pred)
    rf_evaluator = ClassifierEvaluator("RandomForest", y_test, rf_pred)

    # Compare classifiers
    print("\nComparing classifiers based on precedence order...")
    comparison = lr_evaluator.compare_with_other(rf_evaluator, verbose=True)

    return lr_evaluator, rf_evaluator, comparison


def demo_integrated_training():
    """Demonstrate integrated training with evaluator."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION 3: INTEGRATED TRAINING FUNCTION")
    print("=" * 60)

    # Generate sample data
    X, y = generate_imbalanced_data(n_samples=2000)

    # Convert to pandas DataFrame/Series for compatibility
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.3, random_state=42, stratify=y
    )

    # Use the integrated training function
    print("\nUsing train_and_evaluate_classifiers with evaluator...")
    results = train_and_evaluate_classifiers(
        X_train, X_test, y_train, y_test,
        use_evaluator=True  # This enables the new ClassifierEvaluator
    )

    print("\nResults structure:")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  - Has model: {result['model'].__class__.__name__}")
        print(f"  - Has evaluator: {'evaluator' in result}")
        if 'evaluation' in result:
            metrics = result['evaluation']['metrics']
            print(f"  - Accuracy: {metrics['accuracy']:.4f}")
            print(f"  - F1-Score: {metrics['f1_score']:.4f}")

    return results


def quick_evaluation_example():
    """Show quick evaluation example."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION 4: QUICK EVALUATION")
    print("=" * 60)

    # Create some simple test data
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 1, 1, 1])  # Some errors

    print("\nTrue labels: ", y_true)
    print("Pred labels: ", y_pred)

    # Import the quick_evaluate_classifier function
    from models import quick_evaluate_classifier

    print("\nRunning quick evaluation...")
    results = quick_evaluate_classifier(
        classifier_name="TestClassifier",
        y_true=y_true,
        y_pred=y_pred,
        plot_cm=False,
        normalize_cm=True
    )

    return results


def demo_manual_metric_calculation():
    """Demonstrate manual metric calculation and precedence."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION 5: MANUAL METRIC CALCULATION")
    print("=" * 60)

    # Simple example to show precedence order
    print("\nExample: Two classifiers with different metric profiles")
    print("-" * 40)

    # Simulated metrics for two classifiers
    metrics_a = {
        'accuracy': 0.92,
        'f1_score': 0.85,
        'precision': 0.88,
        'recall': 0.82
    }

    metrics_b = {
        'accuracy': 0.91,  # Lower accuracy
        'f1_score': 0.90,  # Higher F1
        'precision': 0.95,  # Higher precision
        'recall': 0.95    # Higher recall
    }

    print("\nClassifier A:")
    print(f"  Accuracy:  {metrics_a['accuracy']:.4f}")
    print(f"  F1-Score:  {metrics_a['f1_score']:.4f}")
    print(f"  Precision: {metrics_a['precision']:.4f}")
    print(f"  Recall:    {metrics_a['recall']:.4f}")

    print("\nClassifier B:")
    print(f"  Accuracy:  {metrics_b['accuracy']:.4f}")
    print(f"  F1-Score:  {metrics_b['f1_score']:.4f}")
    print(f"  Precision: {metrics_b['precision']:.4f}")
    print(f"  Recall:    {metrics_b['recall']:.4f}")

    print("\nPrecedence analysis:")
    print("1. Accuracy: A wins (0.92 > 0.91)")
    print("2. If tied, check F1-Score")
    print("3. If tied, check Precision")
    print("4. If tied, check Recall")
    print("\nResult: Classifier A wins due to higher accuracy")

    print("\nThis demonstrates why precedence matters:")
    print("- Classifier B has better F1, precision, and recall")
    print("- But accuracy has highest precedence")
    print("- For malware detection: F1 might be more important")
    print("- You can adjust precedence based on your use case")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("CLASSIFIER EVALUATOR DEMONSTRATION")
    print("=" * 60)
    print("\nThis demonstrates the new ClassifierEvaluator class for")
    print("evaluating classifier performance with metrics in precedence order:")
    print("Accuracy > F1-Score > Precision > Recall")
    print("\n(Important for imbalanced problems like malware detection)")

    try:
        # Demo 1: Single evaluator
        evaluator1, results1 = demo_single_evaluator()

        # Demo 2: Comparison
        evaluator2, evaluator3, comparison = demo_comparison()

        # Demo 3: Integrated training
        training_results = demo_integrated_training()

        # Demo 4: Quick evaluation
        quick_results = quick_evaluation_example()

        # Demo 5: Manual metric calculation
        demo_manual_metric_calculation()

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("\nThe ClassifierEvaluator provides:")
        print("1. Comprehensive metrics calculation (Accuracy, F1, Precision, Recall)")
        print("2. Metrics in order of precedence for imbalanced problems")
        print("3. Confusion matrix generation and visualization")
        print("4. Classifier comparison with tie-breaking rules")
        print("5. Integration with existing training workflow")

        print("\nKey points for malware detection:")
        print("- Accuracy alone can be misleading for imbalanced data")
        print("- F1-score balances precision and recall")
        print("- Precedence order helps choose best classifier systematically")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("\nYou may need to install missing dependencies:")
        print("pip install scikit-learn pandas numpy matplotlib seaborn")


if __name__ == "__main__":
    main()