import numpy as np
import pandas as pd
from src.selection.importance import compute_mutual_info, compute_rf_importance
from src.selection.permutation import compute_permutation_importance

def remove_redundant_features(X, threshold=0.92):
    """
    Removes highly correlated features above correlation threshold.
    """
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return X.drop(columns=to_drop), to_drop

def select_top_features(X, y=None, top_n=30):
    """
    Selects top N features combining variance, correlation removal, and feature importance.
    """
    # 1. Drop low variance & physically unsupported hardware features
    from src.hardware.capabilities import HardwareCapabilityRegistry
    registry = HardwareCapabilityRegistry()
    
    variances = X.var()
    valid_cols = [c for c in variances[variances > 1e-6].index if registry.is_feature_supported(c)]
    X_filtered = X[valid_cols]
    
    # 2. Correlation filtering
    X_uncorr, dropped_corr = remove_redundant_features(X_filtered, threshold=0.92)
    
    # 3. If y exists (unsupervised or proxy target), rank by variance or RF
    if y is not None and len(np.unique(y)) > 1:
        mi_scores = compute_mutual_info(X_uncorr, y)
        rf_scores = compute_rf_importance(X_uncorr, y)
        perm_scores = compute_permutation_importance(X_uncorr, y)
        
        ranking_df = pd.DataFrame({
            'feature': list(X_uncorr.columns),
            'mutual_info': [mi_scores.get(c, 0) for c in X_uncorr.columns],
            'rf_importance': [rf_scores.get(c, 0) for c in X_uncorr.columns],
            'permutation': [perm_scores.get(c, 0) for c in X_uncorr.columns]
        })
        
        # Normalize and aggregate rank
        for col in ['mutual_info', 'rf_importance', 'permutation']:
            max_v = ranking_df[col].max()
            if max_v > 0:
                ranking_df[col] = ranking_df[col] / max_v
                
        ranking_df['score'] = ranking_df[['mutual_info', 'rf_importance', 'permutation']].mean(axis=1)
        ranking_df = ranking_df.sort_values(by='score', ascending=False)
        selected_cols = ranking_df.head(top_n)['feature'].tolist()
    else:
        # Fallback: rank by variance ratio
        var_series = X_uncorr.var() / (X_uncorr.abs().mean() + 1e-9)
        selected_cols = var_series.nlargest(top_n).index.tolist()
        ranking_df = pd.DataFrame({'feature': var_series.index, 'score': var_series.values}).sort_values(by='score', ascending=False)
        
    return selected_cols, ranking_df
