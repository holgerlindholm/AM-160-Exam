# Data loader
import os
import numpy as np
import torch
import netCDF4 as nc
from torch.utils.data import Dataset, DataLoader

def load_nc_data(file_path):
    """Loads NetCDF data from the given file path."""
    dataset = nc.Dataset(file_path, mode='r')
    lon = dataset.variables["longitude"][:].tolist()
    lat = dataset.variables["latitude"][:].tolist()
    return dataset.variables['z'][:], lon, lat

def normalize_per_channel(data):
    """
    Normalization per channel for data with shape (T, C, H, W).
    Each channel is normalized across all frames and spatial dimensions to have mean 0 and standard deviation 1.
    """
    data = data.astype(np.float32)
    mean = data.mean(axis=(0, 2, 3), keepdims=True)
    std = data.std(axis=(0, 2, 3), keepdims=True)
    return (data - mean) / (std + 1e-8)

def combine_data(files):
    """
    Loads and combines data from multiple NetCDF files into one torch tensor.
    """
    all_data = []
    lon_list = []
    lat_list = []
    for file in files:
        data, lon, lat = load_nc_data(file)
        all_data.append(data)
        lon_list = lon
        lat_list = lat
    
    combined_data = np.concatenate(all_data, axis=0)
    combined_data = normalize_per_channel(combined_data)
    return torch.tensor(combined_data, dtype=torch.float32), lon_list, lat_list

class GeopotentialDataset(Dataset):
    """
    Dataset for autoregressive prediction with variable-length input sequences.
    """
    def __init__(self, files, n_past=1):
        self.clean_data, self.lon, self.lat = combine_data(files)
        self.n_past = n_past  

    def __len__(self):
        return self.clean_data.shape[0] - self.n_past

    def __getitem__(self, idx):
        x_input = self.clean_data[idx : idx + self.n_past]  # (n_past, C, H, W)
        y_target = self.clean_data[idx + self.n_past]  # (C, H, W)
        return x_input, y_target  

def get_dataloaders(dir_path, n_past=1, batch_size=16, shuffle=True):
    """
    Creates train and test DataLoaders with flexible past time steps.
    """
    files = [f for f in os.listdir(dir_path) if f.endswith('.nc')]
    
    train_files = [os.path.join(dir_path, file) for file in files if file != "z1985.nc"]
    test_files = [os.path.join(dir_path, "z1985.nc")]

    train_dataset = GeopotentialDataset(train_files, n_past=n_past)
    test_dataset = GeopotentialDataset(test_files, n_past=n_past)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Finished loading data")

    return train_loader, test_loader, train_dataset, test_dataset