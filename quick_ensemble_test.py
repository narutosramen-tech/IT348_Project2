#!/usr/bin/env python3
"""
Quick test of the ensemble system.
"""

import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from models import SecurityFirstEnsemble

# Simple realistic test
X, y = make_classification(
    n_samples=1000,
    n_features=30,
    n_informative=20,
    n_redundant=5,
    n_repeated=5,
    weights=[0.85, 0.15],  # 85% benign, 15% malware
    flip_y=0.1,  # 10% noise
    class_sep=0.8,
    random_state=42
)

X_df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
y_s = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_s, test_size=0.3, random_state=42, stratify=y_s
)

print(f"Training data: {X_train.shape}")
print(f"Test data: {X_test.shape}")
print(f"Malware rate: {y_s.mean():.1%}")

# Test ensemble
ensemble = SecurityFirstEnsemble(
    tie_breaker="malware",
    voting_type="hard"
)

ensemble.fit(X_train, y_train)
results = ensemble.evaluate(X_test, y_test, verbose=True)

print("\nEnsemble components:")
for name, model in ensemble.individual_models.items():
    preds = model.predict(X_test)
    accuracy = (preds == y_test).mean()
    print(f"  {name}: accuracy = {accuracy:.3f}")