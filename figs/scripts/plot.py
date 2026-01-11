import torch
import matplotlib.pyplot as plt

# given parameters
num_steps = 50
base_shift = 0.5
max_shift = 1.15
image_seq_len = 4096

# define helpers
def get_lin_function(x1=256, y1=base_shift, x2=4096, y2=max_shift):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

def time_shift(mu, sigma, t):
    return torch.exp(torch.tensor(mu)) / (torch.exp(torch.tensor(mu)) + (1 / t - 1) ** sigma)

# compute schedule
timesteps = torch.linspace(1, 0, num_steps + 1)
mu = get_lin_function()(image_seq_len)
print(f"Computed mu: {mu}")
schedule = time_shift(mu, 1.0, timesteps)

# plot
plt.figure(figsize=(8,6))
plt.plot(range(num_steps+1), schedule)
plt.xlabel("Step")
plt.ylabel("Shifted timestep")
plt.title("FLUX Sampling Schedule")
plt.grid(True)

plt.savefig("sampling_schedule.png")
