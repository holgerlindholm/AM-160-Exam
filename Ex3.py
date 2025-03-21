# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# beta = 0.1
# x0 = 2.0
# max_t = 100

# # Compute KL divergence for each t
# t_values = np.arange(1, max_t + 1)
# kl_divergences = []

# for t in t_values:
#     mu_t = x0 * ((1 - beta) ** (t / 2))
#     sigma2_t = 1 - (1 - beta) ** t
#     kl = 0.5 * (sigma2_t + mu_t**2 - 1 - np.log(sigma2_t))
#     kl_divergences.append(kl)

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(t_values, kl_divergences, label='KL Divergence')
# plt.xlabel('t')
# plt.ylabel('KL Divergence')
# plt.title('KL Divergence between q(x(t)|x(0)) and N(0,1)')
# plt.legend()
# plt.grid(True)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from tqdm import tqdm

# Parameters
beta = 0.1
x0 = 2.0
max_t = 50
n_samples = 10000  # Number of trajectories to simulate
grid_points = 1000  # Resolution for numerical KL computation

# Precompute all samples for all timesteps
samples = np.zeros((n_samples, max_t + 1))
samples[:, 0] = x0  # Initial condition x(0) = x0

for t in range(1, max_t + 1):
    noise = np.random.randn(n_samples)
    samples[:, t] = np.sqrt(1 - beta) * samples[:, t - 1] + np.sqrt(beta) * noise

# Compute KL divergence numerically for each t
t_values = np.arange(1, max_t + 1)
kl_divergences = []

# Define integration grid
x_grid = np.linspace(-5, 5, grid_points)
dx = x_grid[1] - x_grid[0]
true_pdf = norm.pdf(x_grid, 0, 1)  # N(0,1) PDF

for t in tqdm(t_values):
    x_t = samples[:, t]
    
    # Estimate PDF of x(t) using KDE
    kde = gaussian_kde(x_t)
    q = kde.evaluate(x_grid)
    q = np.clip(q, 1e-10, None)  # Avoid log(0)
    
    # Compute KL divergence: ∫ q(x) * (log(q(x)) - log(p(x))) dx
    integrand = q * (np.log(q) - np.log(true_pdf))
    kl = np.sum(integrand) * dx
    kl_divergences.append(kl)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t_values, kl_divergences, marker='o', linestyle='--', label='KL Divergence (Numerical)')
plt.xlabel('t')
plt.ylabel('KL Divergence')
plt.title('KL Divergence Between q(x(t)|x(0)) and N(0,1) (Numerical)')
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

T1 = [0,3,6,9,12,15,18,21,24]
T2 = [6,12,18,24]
T3 = [3,6,15,18,22]

def y(T):
    return ((np.log(T)+np.random.randn()*0.2)/3)+np.random.randn()*0.2

for i in range(3):
    for T in [T1,T2,T3]:
        plt.scatter(T,y(T),color="gray")
        plt.plot(T,y(T),color="gray",alpha=0.1)
plt.show()
