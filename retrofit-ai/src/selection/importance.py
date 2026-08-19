import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

def compute_mutual_info(X, y):
    """
    Computes mutual information scores for feature matrix X and target y.
    """
    mi_scores = mutual_info_classif(X, y, random_state=42)
    return dict(zip(X.columns, mi_scores))

def compute_rf_importance(X, y):
    """
    Computes Random Forest Gini feature importances.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return dict(zip(X.columns, rf.feature_importances_))
