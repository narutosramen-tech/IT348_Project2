from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

def train_and_evaluate_classifiers(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
    """
    Train multiple classifiers and evaluate performance.
    """

    classifier = {
        "LogisticRegression": LogisticRegression(max_iter = 1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    results = {}

    for name, clf in classifier.items():
        print(f"\nTraining {name}...")

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

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
    
    return results