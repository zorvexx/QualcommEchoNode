import numpy as np
from sklearn.svm import OneClassSVM

class OneClassSVMAnomalyDetector:
    def __init__(self, nu=0.05, kernel='rbf', gamma='scale'):
        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        
    def fit(self, X):
        self.model.fit(X)
        return self
        
    def predict_score(self, X):
        scores = -self.model.decision_function(X)
        return scores
