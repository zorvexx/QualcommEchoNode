import time
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from src.models.isolation_forest import IsolationForestAnomalyDetector
from src.models.mahalanobis import MahalanobisAnomalyDetector
from src.models.one_class_svm import OneClassSVMAnomalyDetector
from src.models.autoencoder import AutoencoderAnomalyDetector

def compare_anomaly_models(X_train, X_test, y_test):
    """
    Trains and evaluates Isolation Forest, Mahalanobis, One-Class SVM, and Autoencoder.
    Returns comparison DataFrame and best model object.
    y_test: 0 for healthy (normal), 1 for anomaly.
    """
    candidates = {
        'IsolationForest': IsolationForestAnomalyDetector(contamination=0.05),
        'Mahalanobis': MahalanobisAnomalyDetector(),
        'OneClassSVM': OneClassSVMAnomalyDetector(nu=0.05),
        'Autoencoder': AutoencoderAnomalyDetector(latent_dim=8, epochs=30, lr=1e-3)
    }
    
    results = []
    trained_models = {}
    
    for name, model in candidates.items():
        t0 = time.time()
        model.fit(X_train)
        train_time = time.time() - t0
        
        t0 = time.time()
        scores = model.predict_score(X_test)
        inference_time_ms = ((time.time() - t0) / len(X_test)) * 1000.0
        
        # Calculate threshold as 99th percentile of train score
        train_scores = model.predict_score(X_train)
        thresh = np.percentile(train_scores, 99)
        preds = (scores > thresh).astype(int)
        
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, scores)
        except Exception:
            auc = 0.5
            
        fpr = float(np.sum((preds == 1) & (y_test == 0)) / (np.sum(y_test == 0) + 1e-9))
        model_size_kb = sys.getsizeof(model) / 1024.0
        
        results.append({
            'model': name,
            'f1': round(f1, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'roc_auc': round(auc, 4),
            'false_positive_rate': round(fpr, 4),
            'inference_ms': round(inference_time_ms, 3),
            'model_size_kb': round(model_size_kb, 2)
        })
        trained_models[name] = (model, thresh)
        
    df_res = pd.DataFrame(results).sort_values(by='f1', ascending=False)
    best_name = df_res.iloc[0]['model']
    best_model, best_thresh = trained_models[best_name]
    
    return df_res, best_name, best_model, best_thresh
