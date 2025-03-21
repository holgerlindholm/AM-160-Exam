import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from Ex2.ConditionalConvVAE import *
from Ex2.generating import *
from Ex2.Custom_data_loader import *

def main():
    # Model parameters
    n_past = 5
    latent_dim = 128
    hidden_dim = 64
    beta = 0.01
    batch_size = 64
    learning_rate = 1e-4
    num_epochs = 100
    n_past = 5
    n_future = 25
    ensemble_size = 10
    
    # Create timestamp for logs and checkpoints
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_path = "Ex2/vae_model_20250319_013258.pth"
    data_path = "data/"

    # Load data and create data loaders
    train_loader, test_loader,_,test_dataset = get_dataloaders(data_path, n_past=n_past, batch_size=batch_size)

    # Load model
    model = ConditionalConvVAE(input_channels=2, hidden_dim=hidden_dim, latent_dim=latent_dim, n_past=n_past)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))["model_state_dict"])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate predictions
    print(f"Generating {ensemble_size} ensemble predictions for {n_future} days...")
    results = generate_autoregressive_predictions(
        model, 
        test_dataset, 
        n_future=n_future, 
        device=device, 
        ensemble_size=ensemble_size
    )
    
    # Get normalization stats for denormalization
    #data_stats = test_dataset.get_stats()
    
    # Plot spatial patterns
    print("Plotting spatial patterns...")
    plot_spatial_patterns_with_ensemble(
        results['true_sequence'],
        results['ensemble_predictions'],
        plot_timesteps=[5, 10, 15, 25],  # 5 is the first predicted frame
        data_stats=None
    )
    
    # Plot MSE
    print("Plotting MSE errors...")
    plot_ensemble_mse_over_time(
        results['true_sequence'],
        results['ensemble_predictions'],
        data_stats=None,
        n_past=n_past
    )

if __name__ == '__main__':
    main()
