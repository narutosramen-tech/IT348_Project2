from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
# Optional imports for visualization (only needed if plotting confusion matrices)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    plt = None
    sns = None


class ClassifierEvaluator:
    """
    A class to evaluate classifier performance with metrics in order of precedence:
    Accuracy > F1-score > Precision > Recall

    Useful for imbalanced classification problems like malware detection.
    """

    def __init__(self, classifier_name: str, y_true, y_pred):
        """
        Initialize evaluator with ground truth and predictions.

        Args:
            classifier_name: Name of the classifier being evaluated
            y_true: Ground truth labels (numpy array, pandas Series, or list)
            y_pred: Predicted labels (numpy array, pandas Series, or list)
        """
        self.classifier_name = classifier_name

        # Convert to numpy arrays for consistent handling
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.

        Returns:
            Dictionary with accuracy, f1, precision, recall scores
        """
        metrics = {}

        # Calculate accuracy
        metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)

        # Calculate F1-score, precision, recall (macro-averaged for imbalanced datasets)
        metrics['f1_score'] = f1_score(self.y_true, self.y_pred, average='macro')
        metrics['precision'] = precision_score(self.y_true, self.y_pred, average='macro')
        metrics['recall'] = recall_score(self.y_true, self.y_pred, average='macro')

        return metrics

    def get_confusion_matrix(self, normalize: bool = False) -> np.ndarray:
        """
        Generate confusion matrix.

        Args:
            normalize: If True, return normalized confusion matrix

        Returns:
            Confusion matrix as numpy array
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        return cm

    def plot_confusion_matrix(self, save_path: Optional[str] = None,
                              normalize: bool = False, title_suffix: str = "") -> None:
        """
        Plot confusion matrix as a heatmap.

        Args:
            save_path: Optional path to save the plot image
            normalize: If True, plot normalized confusion matrix
            title_suffix: Additional text to append to title
        """
        if not VISUALIZATION_AVAILABLE:
            print("Visualization libraries (matplotlib, seaborn) not available.")
            print("Install with: pip install matplotlib seaborn")
            return

        cm = self.get_confusion_matrix(normalize=normalize)

        plt.figure(figsize=(8, 6))

        if normalize:
            display_cm = np.round(cm, 2)
            title = f"Normalized Confusion Matrix - {self.classifier_name} {title_suffix}"
        else:
            display_cm = cm
            title = f"Confusion Matrix - {self.classifier_name} {title_suffix}"

        sns.heatmap(display_cm, annot=True, fmt='.2f' if normalize else 'd',
                    cmap='Blues', linewidths=0.5, linecolor='gray')

        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

        plt.show()

    def evaluate(self, verbose: bool = True,
                 include_confusion_matrix: bool = False,
                 plot_confusion_matrix: bool = False) -> Dict[str, Any]:
        """
        Comprehensive evaluation of classifier performance.

        Args:
            verbose: If True, print detailed results
            include_confusion_matrix: If True, include confusion matrix in results
            plot_confusion_matrix: If True, plot confusion matrix

        Returns:
            Dictionary with all evaluation metrics and optionally confusion matrix
        """
        # Calculate metrics
        metrics = self.calculate_metrics()

        if verbose:
            self._print_evaluation_report(metrics)

        # Get confusion matrix if requested
        results = {'metrics': metrics, 'classifier_name': self.classifier_name}

        if include_confusion_matrix or plot_confusion_matrix:
            cm = self.get_confusion_matrix()
            results['confusion_matrix'] = cm

        if plot_confusion_matrix:
            self.plot_confusion_matrix()

        return results

    def _print_evaluation_report(self, metrics: Dict[str, float]) -> None:
        """
        Print formatted evaluation report.
        """
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS - {self.classifier_name}")
        print(f"{'='*60}")

        # Print metrics in order of precedence
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")

        print(f"{'='*60}")

    def compare_with_other(self, other_evaluator: 'ClassifierEvaluator',
                           verbose: bool = True) -> Dict[str, str]:
        """
        Compare this classifier with another based on precedence order.

        Args:
            other_evaluator: Another ClassifierEvaluator instance to compare with
            verbose: If True, print comparison results

        Returns:
            Dictionary with comparison results and winner for each metric
        """
        metrics_self = self.calculate_metrics()
        metrics_other = other_evaluator.calculate_metrics()

        comparison = {}

        # Compare in order of precedence: Accuracy > F1 > Precision > Recall
        if metrics_self['accuracy'] > metrics_other['accuracy']:
            comparison['accuracy_winner'] = self.classifier_name
        elif metrics_self['accuracy'] < metrics_other['accuracy']:
            comparison['accuracy_winner'] = other_evaluator.classifier_name
        else:
            comparison['accuracy_winner'] = "Tie"

        if metrics_self['f1_score'] > metrics_other['f1_score']:
            comparison['f1_winner'] = self.classifier_name
        elif metrics_self['f1_score'] < metrics_other['f1_score']:
            comparison['f1_winner'] = other_evaluator.classifier_name
        else:
            comparison['f1_winner'] = "Tie"

        if metrics_self['precision'] > metrics_other['precision']:
            comparison['precision_winner'] = self.classifier_name
        elif metrics_self['precision'] < metrics_other['precision']:
            comparison['precision_winner'] = other_evaluator.classifier_name
        else:
            comparison['precision_winner'] = "Tie"

        if metrics_self['recall'] > metrics_other['recall']:
            comparison['recall_winner'] = self.classifier_name
        elif metrics_self['recall'] < metrics_other['recall']:
            comparison['recall_winner'] = other_evaluator.classifier_name
        else:
            comparison['recall_winner'] = "Tie"

        # Determine overall winner based on precedence
        overall_winner = self._determine_overall_winner(comparison)
        comparison['overall_winner'] = overall_winner

        if verbose:
            self._print_comparison_report(comparison, metrics_self, metrics_other,
                                         other_evaluator.classifier_name)

        return comparison

    def _determine_overall_winner(self, comparison: Dict[str, str]) -> str:
        """
        Determine overall winner based on precedence order.
        """
        # Check in order: Accuracy > F1 > Precision > Recall
        if comparison['accuracy_winner'] != "Tie":
            return comparison['accuracy_winner']
        elif comparison['f1_winner'] != "Tie":
            return comparison['f1_winner']
        elif comparison['precision_winner'] != "Tie":
            return comparison['precision_winner']
        elif comparison['recall_winner'] != "Tie":
            return comparison['recall_winner']
        else:
            return "Tie (All metrics equal)"

    def _print_comparison_report(self, comparison: Dict[str, str],
                                metrics_self: Dict[str, float],
                                metrics_other: Dict[str, float],
                                other_name: str) -> None:
        """
        Print formatted comparison report.
        """
        print(f"\n{'='*60}")
        print(f"CLASSIFIER COMPARISON")
        print(f"{'='*60}")
        print(f"{self.classifier_name} vs {other_name}")
        print(f"{'='*60}")

        print(f"\nAccuracy Comparison:")
        print(f"  {self.classifier_name}: {metrics_self['accuracy']:.4f}")
        print(f"  {other_name}: {metrics_other['accuracy']:.4f}")
        print(f"  Winner: {comparison['accuracy_winner']}")

        print(f"\nF1-Score Comparison:")
        print(f"  {self.classifier_name}: {metrics_self['f1_score']:.4f}")
        print(f"  {other_name}: {metrics_other['f1_score']:.4f}")
        print(f"  Winner: {comparison['f1_winner']}")

        print(f"\nPrecision Comparison:")
        print(f"  {self.classifier_name}: {metrics_self['precision']:.4f}")
        print(f"  {other_name}: {metrics_other['precision']:.4f}")
        print(f"  Winner: {comparison['precision_winner']}")

        print(f"\nRecall Comparison:")
        print(f"  {self.classifier_name}: {metrics_self['recall']:.4f}")
        print(f"  {other_name}: {metrics_other['recall']:.4f}")
        print(f"  Winner: {comparison['recall_winner']}")

        print(f"\n{'='*60}")
        print(f"OVERALL WINNER (by precedence): {comparison['overall_winner']}")
        print(f"{'='*60}")


def train_and_evaluate_classifiers(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                   y_train: pd.Series, y_test: pd.Series,
                                   use_evaluator: bool = True):
    """
    Train multiple classifiers and evaluate performance.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        use_evaluator: If True, use ClassifierEvaluator for comprehensive evaluation

    Returns:
        Dictionary with model results and evaluation metrics
    """

    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    evaluators = {}

    for name, clf in classifiers.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")

        # Train classifier
        clf.fit(X_train, y_train)

        # Make predictions
        y_pred = clf.predict(X_test)

        if use_evaluator:
            # Use ClassifierEvaluator for comprehensive evaluation
            evaluator = ClassifierEvaluator(name, y_test.values, y_pred)
            evaluation = evaluator.evaluate(
                verbose=True,
                include_confusion_matrix=True,
                plot_confusion_matrix=False  # Set to True to automatically plot confusion matrices
            )

            evaluators[name] = evaluator

            results[name] = {
                "model": clf,
                "predictions": y_pred,
                "evaluator": evaluator,
                "evaluation": evaluation
            }
        else:
            # Original evaluation method (backward compatibility)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, digits=4)
            cm = confusion_matrix(y_test, y_pred)

            print(f"{name} Accuracy: {acc:.4f}")
            print(f"{name} Classification Report:\n{report}")
            print(f"{name} Confusion Matrix:\n{cm}")

            results[name] = {
                "model": clf,
                "accuracy": acc,
                "report": report,
                "confusion_matrix": cm
            }

    # Compare classifiers if using evaluator and we have at least 2
    if use_evaluator and len(evaluators) >= 2:
        print(f"\n{'='*60}")
        print("COMPARING CLASSIFIERS")
        print(f"{'='*60}")

        # Get the two evaluators for comparison
        evaluator_list = list(evaluators.values())
        evaluator_list[0].compare_with_other(evaluator_list[1])

    return results


def quick_evaluate_classifier(classifier_name: str, y_true: np.ndarray, y_pred: np.ndarray,
                             plot_cm: bool = False, normalize_cm: bool = False) -> Dict[str, Any]:
    """
    Quick evaluation of a single classifier using the ClassifierEvaluator.

    Args:
        classifier_name: Name of the classifier
        y_true: Ground truth labels
        y_pred: Predicted labels
        plot_cm: If True, plot confusion matrix
        normalize_cm: If True, normalize confusion matrix

    Returns:
        Evaluation results
    """
    evaluator = ClassifierEvaluator(classifier_name, y_true, y_pred)

    if plot_cm:
        evaluator.plot_confusion_matrix(normalize=normalize_cm)

    return evaluator.evaluate(verbose=True, include_confusion_matrix=True)


class SecurityFirstEnsemble:
    """
    Security-first voting ensemble for malware detection.
    Uses 3 models with security-conservative tie-breaking.
    """

    def __init__(self, tie_breaker: str = "malware",  # "malware", "reject", or "confidence"
                 voting_type: str = "hard"):  # "hard", "soft", or "stacked"
        """
        Initialize ensemble.

        Args:
            tie_breaker: What to do when models disagree
                - "malware": Default to malware (security-first)
                - "reject": Flag for human review
                - "confidence": Use highest confidence prediction
            voting_type: Type of ensemble
                - "hard": Majority voting
                - "soft": Probability-weighted voting
                - "stacked": Stacked ensemble with meta-classifier
        """
        self.tie_breaker = tie_breaker
        self.voting_type = voting_type
        self.models = self._create_base_models()
        self.ensemble = None
        self.is_fitted = False

    def _create_base_models(self) -> Dict[str, Any]:
        """Create the 3-model ensemble."""
        models = {
            "LogisticRegression": LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'  # Important for imbalanced malware data
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced'
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        }
        return models

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train all models in the ensemble."""
        print(f"\n{'='*70}")
        print("TRAINING 3-MODEL ENSEMBLE FOR MALWARE DETECTION")
        print(f"{'='*70}")

        # Train individual models
        self.individual_predictions = {}
        self.individual_models = {}

        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X, y)
            self.individual_models[name] = model

            # Make predictions for ensemble voting
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)[:, 1]  # Probability of malware class
                self.individual_predictions[name] = {
                    'predictions': model.predict(X),
                    'probabilities': probs
                }
            else:
                self.individual_predictions[name] = {
                    'predictions': model.predict(X),
                    'probabilities': None
                }

            # Evaluate individual performance
            evaluator = ClassifierEvaluator(name, y, model.predict(X))
            results = evaluator.evaluate(verbose=False)
            print(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
            print(f"  F1-Score: {results['metrics']['f1_score']:.4f}")

        # Create and train the ensemble based on voting type
        if self.voting_type == "hard":
            self._create_hard_voting_ensemble(X, y)
        elif self.voting_type == "soft":
            self._create_soft_voting_ensemble(X, y)
        elif self.voting_type == "stacked":
            self._create_stacked_ensemble(X, y)

        self.is_fitted = True
        print(f"\nEnsemble training complete. Tie-breaking: {self.tie_breaker}")

    def _create_hard_voting_ensemble(self, X: pd.DataFrame, y: pd.Series):
        """Create hard voting ensemble (majority vote)."""
        estimators = [(name, model) for name, model in self.individual_models.items()]
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='hard',
            n_jobs=-1
        )
        self.ensemble.fit(X, y)

    def _create_soft_voting_ensemble(self, X: pd.DataFrame, y: pd.Series):
        """Create soft voting ensemble (probability weighted)."""
        estimators = [(name, model) for name, model in self.individual_models.items()]
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        self.ensemble.fit(X, y)

    def _create_stacked_ensemble(self, X: pd.DataFrame, y: pd.Series):
        """Create stacked ensemble with meta-classifier."""
        estimators = [(name, model) for name, model in self.individual_models.items()]
        self.ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=5,
            n_jobs=-1
        )
        self.ensemble.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with security-first tie-breaking."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before predicting")

        if self.ensemble is not None and self.voting_type in ["hard", "soft", "stacked"]:
            # Use sklearn's voting classifier if available
            predictions = self.ensemble.predict(X)
        else:
            # Custom voting with security-first tie-breaking
            predictions = self._security_first_vote(X)

        return np.array(predictions)

    def _security_first_vote(self, X: pd.DataFrame) -> np.ndarray:
        """
        Custom voting with security-first tie-breaking.
        When models disagree, default to malware prediction.
        """
        # Collect predictions from all models
        all_predictions = []
        all_probabilities = []

        for name, model in self.individual_models.items():
            preds = model.predict(X)
            all_predictions.append(preds)

            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)[:, 1]
                all_probabilities.append(probs)

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)  # Shape: (n_models, n_samples)

        # Initialize final predictions
        final_predictions = np.zeros(X.shape[0], dtype=int)

        for i in range(X.shape[0]):
            model_votes = all_predictions[:, i]

            # Count votes
            malware_votes = np.sum(model_votes == 1)
            benign_votes = np.sum(model_votes == 0)

            if malware_votes > benign_votes:
                # Majority says malware
                final_predictions[i] = 1
            elif benign_votes > malware_votes:
                # Majority says benign
                final_predictions[i] = 0
            else:
                # Tie - use tie-breaking strategy
                final_predictions[i] = self._break_tie(i, all_predictions, all_probabilities)

        return final_predictions

    def _break_tie(self, sample_idx: int, all_predictions: np.ndarray,
                  all_probabilities: list) -> int:
        """Implement tie-breaking strategies."""
        if self.tie_breaker == "malware":
            # Security-first: default to malware
            return 1
        elif self.tie_breaker == "reject":
            # Flag for review (treat as malware for now)
            return 1
        elif self.tie_breaker == "confidence" and all_probabilities:
            # Use highest confidence
            confidences = []
            for probs in all_probabilities:
                if probs is not None:
                    # Convert to confidence: if prediction is 1, use prob; if 0, use 1-prob
                    pred = all_predictions[len(confidences), sample_idx]
                    conf = probs[sample_idx] if pred == 1 else 1 - probs[sample_idx]
                    confidences.append(conf)

            if confidences:
                # Use prediction from most confident model
                most_confident_idx = np.argmax(confidences)
                return all_predictions[most_confident_idx, sample_idx]
            else:
                # No probabilities available, default to malware
                return 1
        else:
            # Default to security-first
            return 1

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before predicting")

        if self.ensemble is not None and hasattr(self.ensemble, 'predict_proba'):
            return self.ensemble.predict_proba(X)
        else:
            # If ensemble doesn't have predict_proba, use average of base models
            all_probs = []
            for model in self.individual_models.values():
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X)
                    all_probs.append(probs)

            if all_probs:
                return np.mean(all_probs, axis=0)
            else:
                # Fallback: create dummy probabilities
                preds = self.predict(X)
                probs = np.zeros((len(preds), 2))
                probs[preds == 0, 0] = 1.0
                probs[preds == 1, 1] = 1.0
                return probs

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series,
                 verbose: bool = True) -> Dict[str, Any]:
        """Comprehensive evaluation of the ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before evaluating")

        print(f"\n{'='*70}")
        print("ENSEMBLE EVALUATION")
        print(f"{'='*70}")

        # Get ensemble predictions
        y_pred = self.predict(X_test)

        # Use ClassifierEvaluator
        evaluator = ClassifierEvaluator("SecurityFirstEnsemble", y_test, y_pred)
        results = evaluator.evaluate(verbose=verbose, include_confusion_matrix=True)

        # Add ensemble-specific metrics
        results['ensemble_type'] = self.voting_type
        results['tie_breaker'] = self.tie_breaker
        results['num_models'] = len(self.models)

        # Calculate model agreement
        if hasattr(self, 'individual_predictions'):
            agreement_rate = self._calculate_model_agreement(X_test, y_pred)
            results['model_agreement_rate'] = agreement_rate
            print(f"\nModel Agreement: {agreement_rate:.1%} of samples")

        # Show tie-breaking statistics
        if self.tie_breaker == "malware":
            print(f"Tie-breaking: Default to MALWARE (security-first)")

        return results

    def _calculate_model_agreement(self, X: pd.DataFrame, ensemble_preds: np.ndarray) -> float:
        """Calculate how often all models agree with each other."""
        if not hasattr(self, 'individual_models'):
            return 0.0

        n_samples = X.shape[0]
        agreement_count = 0

        for i in range(n_samples):
            model_predictions = []
            for name, model in self.individual_models.items():
                pred = model.predict(X.iloc[[i]])[0]
                model_predictions.append(pred)

            # Check if all models agree
            if len(set(model_predictions)) == 1:
                agreement_count += 1

        return agreement_count / n_samples


def train_and_evaluate_ensemble(X_train: pd.DataFrame, X_test: pd.DataFrame,
                               y_train: pd.Series, y_test: pd.Series,
                               voting_type: str = "hard",
                               tie_breaker: str = "malware") -> Dict[str, Any]:
    """
    Train and evaluate the 3-model ensemble.

    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test labels
        voting_type: "hard", "soft", or "stacked"
        tie_breaker: "malware", "reject", or "confidence"

    Returns:
        Dictionary with ensemble results
    """
    print(f"\n{'='*70}")
    print(f"3-MODEL ENSEMBLE WITH {voting_type.upper()} VOTING")
    print(f"{'='*70}")

    # Create and train ensemble
    ensemble = SecurityFirstEnsemble(
        tie_breaker=tie_breaker,
        voting_type=voting_type
    )

    ensemble.fit(X_train, y_train)

    # Evaluate
    results = ensemble.evaluate(X_test, y_test, verbose=True)

    # Compare with individual models
    print(f"\n{'='*70}")
    print("COMPARISON WITH INDIVIDUAL MODELS")
    print(f"{'='*70}")

    individual_results = train_and_evaluate_classifiers(
        X_train, X_test, y_train, y_test,
        use_evaluator=True
    )

    # Extract best F1 from individual models
    best_individual_f1 = 0
    best_individual_model = None

    for name, result in individual_results.items():
        if 'evaluation' in result:
            f1 = result['evaluation']['metrics']['f1_score']
            if f1 > best_individual_f1:
                best_individual_f1 = f1
                best_individual_model = name

    ensemble_f1 = results['metrics']['f1_score']

    print(f"\nBest Individual Model ({best_individual_model}): F1 = {best_individual_f1:.4f}")
    print(f"Ensemble Model: F1 = {ensemble_f1:.4f}")

    if ensemble_f1 > best_individual_f1:
        improvement = (ensemble_f1 - best_individual_f1) / best_individual_f1 * 100
        print(f"Ensemble IMPROVEMENT: +{improvement:.1f}%")
    else:
        print("Ensemble not better than best individual model")

    return {
        'ensemble': ensemble,
        'ensemble_results': results,
        'individual_results': individual_results,
        'improvement': ensemble_f1 - best_individual_f1
    }