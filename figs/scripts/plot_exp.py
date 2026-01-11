import torch
import matplotlib.pyplot as plt

# Given parameters
num_steps = 50
base_shift = 0.5
max_shift = 1.15
image_seq_len = 4096

# ==== Helpers ====
def get_lin_function(x1=256, y1=base_shift, x2=4096, y2=max_shift):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

def time_shift(mu, sigma, t):
    mu_t = torch.tensor(mu, dtype=torch.float32)
    return torch.exp(mu_t) / (torch.exp(mu_t) + (1 / t - 1)**sigma)

# ==== Baseline schedule ====
timesteps = torch.linspace(1, 0, num_steps + 1)
mu_default = get_lin_function()(image_seq_len)
schedule_default = time_shift(mu_default, 1.0, timesteps)

plt.figure(figsize=(8, 6))
plt.plot(range(num_steps + 1), schedule_default, label=f"mu={mu_default:.4f}")
plt.xlabel("Step")
plt.ylabel("Shifted timestep")
plt.title("FLUX Sampling Schedule (baseline)")
plt.grid(True)
plt.legend()
plt.savefig("schedule_baseline.png")


# ==== Compare different mu ====
mu_list = [0.0, 0.6, 1.15, 1.4, 1.8]  # you can change these
plt.figure(figsize=(8, 6))
for mu in mu_list:
    s = time_shift(mu, 1.0, timesteps)
    plt.plot(range(num_steps + 1), s, label=f"mu={mu}")

plt.xlabel("Step")
plt.ylabel("Shifted timestep")
plt.title("Effect of mu on FLUX time_shift")
plt.grid(True)
plt.legend()
plt.savefig("schedule_compare_mu.png")


# # ==== Compare different sigma ====
# sigma_list = [0.5, 1.0, 2.0, 4.0, 8.0]  # you can change these
# plt.figure(figsize=(8, 6))
# for sigma in sigma_list:
#     s = time_shift(mu_default, sigma, timesteps)
#     plt.plot(range(num_steps + 1), s, label=f"sigma={sigma}")

# plt.xlabel("Step")
# plt.ylabel("Shifted timestep")
# plt.title("Effect of sigma on FLUX time_shift")
# plt.grid(True)
# plt.legend()
# plt.savefig("schedule_compare_sigma.png")

# print(f"Default mu = {mu_default}")
# print("Saved plots: schedule_baseline.png, schedule_compare_mu.png, schedule_compare_sigma.png")
