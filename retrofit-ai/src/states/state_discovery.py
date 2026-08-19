import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

class OperatingStateDiscoverer:
    """
    Discovers normal behavioral clusters using PCA + Clustering (GMM or KMeans).
    Clusters represent data-driven behavioral groupings unless experimentally validated.
    """
    def __init__(self, method='gmm', min_clusters=2, max_clusters=5, n_components=4):
        self.method = method
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=42)
        self.model = None
        self.best_k = min_clusters
        self.cluster_metrics = {}

    def fit(self, X):
        X_pca = self.pca.fit_transform(X)
        best_score = -1.0
        
        for k in range(self.min_clusters, self.max_clusters + 1):
            if self.method == 'kmeans':
                clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = clusterer.fit_predict(X_pca)
            else:
                clusterer = GaussianMixture(n_components=k, random_state=42)
                labels = clusterer.fit_predict(X_pca)
                
            if len(np.unique(labels)) > 1:
                sil = silhouette_score(X_pca, labels)
                db = davies_bouldin_score(X_pca, labels)
                self.cluster_metrics[k] = {'silhouette': sil, 'davies_bouldin': db}
                
                if sil > best_score:
                    best_score = sil
                    self.best_k = k
                    self.model = clusterer
                    
        if self.model is None:
            self.best_k = self.min_clusters
            self.model = KMeans(n_clusters=self.best_k, random_state=42).fit(X_pca)
            
        return self

    def predict(self, X):
        X_pca = self.pca.transform(X)
        if self.method == 'kmeans':
            return self.model.predict(X_pca)
        else:
            return self.model.predict(X_pca)
