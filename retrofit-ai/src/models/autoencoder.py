import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class AutoencoderModule(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super(AutoencoderModule, self).__init__()
        hidden_dim = max(16, input_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

class AutoencoderAnomalyDetector:
    def __init__(self, latent_dim=8, epochs=50, lr=1e-3, batch_size=32):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = None
        self.input_dim = None

    def fit(self, X):
        X_arr = np.asarray(X, dtype=np.float32)
        self.input_dim = X_arr.shape[1]
        self.model = AutoencoderModule(self.input_dim, self.latent_dim)
        
        dataset = torch.utils.data.TensorDataset(torch.tensor(X_arr))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch in dataloader:
                inputs = batch[0]
                optimizer.zero_grad()
                reconstructed, _ = self.model(inputs)
                loss = criterion(reconstructed, inputs)
                loss.backward()
                optimizer.step()
                
        return self

    def predict_score(self, X):
        X_arr = np.asarray(X, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X_arr)
            reconstructed, _ = self.model(inputs)
            residuals = (inputs - reconstructed).numpy() ** 2
            mse_scores = np.mean(residuals, axis=1)
        return mse_scores

    def get_latent_embeddings(self, X):
        X_arr = np.asarray(X, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X_arr)
            _, latent = self.model(inputs)
        return latent.numpy()

    def get_feature_residuals(self, X):
        X_arr = np.asarray(X, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X_arr)
            reconstructed, _ = self.model(inputs)
            residuals = (inputs - reconstructed).numpy() ** 2
        return residuals
