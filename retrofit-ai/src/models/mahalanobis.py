import numpy as np

class MahalanobisAnomalyDetector:
    def __init__(self):
        self.mean = None
        self.inv_cov = None
        
    def fit(self, X):
        X_arr = np.asarray(X)
        self.mean = np.mean(X_arr, axis=0)
        cov = np.cov(X_arr, rowvar=False) + 1e-6 * np.eye(X_arr.shape[1])
        self.inv_cov = np.linalg.inv(cov)
        return self
        
    def predict_score(self, X):
        X_arr = np.asarray(X)
        delta = X_arr - self.mean
        # Mahalanobis distance D^2 = delta * inv_cov * delta^T
        m_dist = np.sqrt(np.sum(np.dot(delta, self.inv_cov) * delta, axis=1))
        return m_dist
