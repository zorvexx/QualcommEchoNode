import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

def compute_permutation_importance(X, y):
    """
    Computes permutation importance scores.
    """
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    result = permutation_importance(model, X, y, n_repeats=5, random_state=42)
    return dict(zip(X.columns, result.importances_mean))
