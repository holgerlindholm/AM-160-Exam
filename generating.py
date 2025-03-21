import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm

from .ConditionalConvVAE import ConditionalConvVAE 

def generate_autoregressive_predictions(model, test_dataset, n_future, device, ensemble_size=10):
    """
    Generate autoregressive predictions with ensemble.
    
    Args:
        model: Trained conditional VAE model
        test_dataset: Test dataset containing the initial data
        n_future: Number of days to predict
        device: Device to run prediction on
        ensemble_size: Number of ensemble members
        
    Returns:
        Dictionary containing true sequence and ensemble predictions
    """
    model.eval()
    
    # Get the initial sequence (first n_past days)
    initial_data = test_dataset.clean_data[:model.n_past]
    
    # Get true sequence for comparison (initial + future days)
    true_sequence = test_dataset.clean_data[:model.n_past + n_future]
    
    # Generate ensemble predictions
    ensemble_predictions = []
    
    for e in range(ensemble_size):
        print(f"Generating ensemble member {e+1}/{ensemble_size}")
        
        # Start with the same initial sequence
        current_sequence = initial_data.clone().unsqueeze(0).to(device)  # (1, n_past, C, H, W)
        
        # Store predictions, starting with initial sequence
        predictions = [frame.cpu().numpy() for frame in initial_data]
        
        # Predict autoregressively
        for t in tqdm(range(n_future), desc=f"Ensemble {e+1}"):
            # Add small noise for ensemble diversity (except first member)
            if e > 0:
                noise_scale = 0.01  # Small noise
                noise = noise_scale * torch.randn_like(current_sequence)
                current_sequence = current_sequence + noise
            
            # Predict next frame
            next_frame = model.predict_next(current_sequence)  # (1, C, H, W)
            
            # Store prediction
            predictions.append(next_frame.cpu().squeeze(0).numpy())
            
            # Update current sequence by removing oldest frame and adding the prediction
            current_sequence = torch.cat([
                current_sequence[:, 1:],  # Remove oldest frame
                next_frame.unsqueeze(1)   # Add predicted frame with sequence dim
            ], dim=1)
        
        # Stack predictions into a sequence
        pred_sequence = np.stack(predictions, axis=0)  # (n_past + n_future, C, H, W)
        ensemble_predictions.append(pred_sequence)
    
    # Convert to numpy array [ensemble_size, n_past + n_future, C, H, W]
    ensemble_array = np.array(ensemble_predictions)
    
    return {
        'true_sequence': true_sequence.cpu().numpy(),
        'ensemble_predictions': ensemble_array
    }

def plot_spatial_patterns_with_ensemble(true_seq, ensemble_preds, plot_timesteps=None, data_stats=None):
    """
    Plot spatial patterns for specific timesteps with ensemble mean and uncertainty.
    
    Args:
        true_seq: True sequence [n_past + n_future, C, H, W]
        ensemble_preds: Ensemble predictions [ensemble_size, n_past + n_future, C, H, W]
        plot_timesteps: List of timesteps to plot (None for default selection)
        data_stats: Statistics for denormalization (optional)
    """
    total_timesteps = true_seq.shape[0]
    ensemble_size = ensemble_preds.shape[0]
    n_channels = true_seq.shape[1]
    
    # Default timesteps if not specified - focus on the future predictions
    n_past = 5  # Assuming 5 past frames were used
    if plot_timesteps is None:
        plot_timesteps = [n_past, n_past + 5, n_past + 10, n_past + 20]
    plot_timesteps = [t for t in plot_timesteps if t < total_timesteps]
    
    num_timesteps = len(plot_timesteps)
    
    # Calculate ensemble mean and std
    ensemble_mean = np.mean(ensemble_preds, axis=0)
    ensemble_std = np.std(ensemble_preds, axis=0)
    
    # Denormalize if statistics are provided
    if data_stats is not None:
        # Convert NumPy arrays
        mean = data_stats["mean"]
        std = data_stats["std"]
        
        true_seq = true_seq * std + mean
        ensemble_mean = ensemble_mean * std + mean
    
    # Create a figure for each channel
    for c in range(n_channels):
        fig, axes = plt.subplots(3, num_timesteps, figsize=(num_timesteps * 4, 12))
        
        # Define common min/max values for consistent color mapping
        vmin = min(true_seq[:, c].min(), ensemble_mean[:, c].min())
        vmax = max(true_seq[:, c].max(), ensemble_mean[:, c].max())
        
        for i, t in enumerate(plot_timesteps):
            # True pattern
            ax = axes[0, i]
            im = ax.imshow(true_seq[t, c], cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f'True t={t} (Ch {c})')
            ax.axis('off')
            plt.colorbar(im, ax=ax)
            
            # Predicted pattern (ensemble mean)
            ax = axes[1, i]
            im = ax.imshow(ensemble_mean[t, c], cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f'Pred t={t} (Ch {c})')
            ax.axis('off')
            plt.colorbar(im, ax=ax)
            
            # Uncertainty (ensemble std)
            ax = axes[2, i]
            im = ax.imshow(ensemble_std[t, c], cmap='hot')
            ax.set_title(f'Std t={t} (Ch {c})')
            ax.axis('off')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(f'spatial_patterns_channel_{c}.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Spatial patterns for channel {c} saved as 'spatial_patterns_channel_{c}.png'")

def plot_ensemble_mse_over_time(true_seq, ensemble_preds, data_stats=None, n_past=5):
    """
    Plot MSE errors over time for each channel with ensemble statistics.
    
    Args:
        true_seq: True sequence [n_past + n_future, C, H, W]
        ensemble_preds: Ensemble predictions [ensemble_size, n_past + n_future, C, H, W]
        data_stats: Statistics for denormalization (optional)
        n_past: Number of past timesteps used for conditioning
    """
    n_timesteps = ensemble_preds.shape[1]
    ensemble_size = ensemble_preds.shape[0]
    n_channels = true_seq.shape[1]
    
    # Denormalize if statistics are provided
    if data_stats is not None:
        mean = data_stats["mean"]
        std = data_stats["std"]
        
        true_seq = true_seq * std + mean
        ensemble_preds = ensemble_preds * std + mean
    
    # Calculate MSE for each ensemble member and each channel
    # Only consider the future predictions (not the initial conditioning sequence)
    mse_all = np.zeros((ensemble_size, n_timesteps - n_past, n_channels))
    
    for e in range(ensemble_size):
        for t in range(n_past, n_timesteps):
            for c in range(n_channels):
                # Calculate MSE
                mse_all[e, t - n_past, c] = mean_squared_error(
                    true_seq[t, c].flatten(), 
                    ensemble_preds[e, t, c].flatten()
                )
    
    # Calculate mean and std of MSE across ensemble
    mse_mean = np.mean(mse_all, axis=0)  # [n_timesteps - n_past, n_channels]
    mse_std = np.std(mse_all, axis=0)    # [n_timesteps - n_past, n_channels]
    
    # Create MSE plot for each channel
    plt.figure(figsize=(12, 8))
    
    timesteps = np.arange(n_past, n_timesteps)  # Adjust timestep indices
    
    for c in range(n_channels):
        plt.plot(timesteps, mse_mean[:, c], label=f'Channel {c}', linewidth=2)
        plt.fill_between(
            timesteps, 
            mse_mean[:, c] - mse_std[:, c], 
            mse_mean[:, c] + mse_std[:, c], 
            alpha=0.3
        )
    
    plt.xlabel('Days from Jan 1, 1985', fontsize=14)
    plt.ylabel('Mean Squared Error', fontsize=14)
    plt.title('Prediction Error Over Time (1985)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('mse_over_time.png', dpi=200)
    plt.close()
    print("MSE error plot saved as 'mse_over_time.png'")
