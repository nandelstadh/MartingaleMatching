import torch
from matplotlib import pyplot as plt
from torch.cuda import is_available

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

from distributions import *
from paths import *
from plotting import *
from simulation import *
from testfuncs import *
from training import *

PARAMS = {
    "scale": 15.0,
    "target_scale": 10.0,
    "target_std": 1.0,
}

# p_data = GaussianMixture.symmetric_2D(
#     nmodes=5, std=PARAMS["target_std"], scale=PARAMS["target_scale"]
# ).to(device)
p_data = CheckerboardSampleable(device, grid_size=4)

num_epochs = 5000
batch_size = 10000
steps = 1000
dim = 2
tfunc = Polynomial()
sigma = 0
dt = 1 / (steps - 1)

# Construct conditional probability path
path = GaussianConditionalProbabilityPath(
    p_data=p_data,
    alpha=LinearAlpha(),
    beta=SquareRootBeta(),
    dt=dt,
    sigma=sigma,
).to(device)

model = MLPDrift(dim=2, hiddens=[64, 64, 64, 64])

# Construct trainer
trainer = MartingaleMatchingTrainer(
    path,
    model,
    steps,
    dim,
    tfunc,
    sigma,
)

losses = trainer.train(
    num_epochs=num_epochs, device=device, lr=1e-3, batch_size=batch_size
)


#######################
# Change these values #
#######################
num_samples = 1000
num_timesteps = 300
num_marginals = 3
sigma_plot = 0.5

##########################
# Play around With These #
##########################
num_samples = 50000
num_marginals = 5

##############
# Setup Plots #
##############

fig, axes = plt.subplots(2, num_marginals, figsize=(6 * num_marginals, 6 * 2))
axes = axes.reshape(2, num_marginals)
scale = 6.0

###########################
# Graph Ground-Truth Marginals #
###########################
ts = torch.linspace(0.0, 1.0, num_marginals).to(device)
for idx, t in enumerate(ts):
    tt = t.view(1, 1).expand(num_samples, 1)
    xts = path.sample_marginal_path(tt)
    hist2d_samples(
        samples=xts.cpu(),
        ax=axes[0, idx],
        bins=200,
        scale=scale,
        percentile=99,
        alpha=1.0,
    )
    axes[0, idx].set_xlim(-scale, scale)
    axes[0, idx].set_ylim(-scale, scale)
    axes[0, idx].set_xticks([])
    axes[0, idx].set_yticks([])
    axes[0, idx].set_title(f"$t={t.item():.2f}$", fontsize=15)
axes[0, 0].set_ylabel("Ground Truth", fontsize=20)

###############################################
# Graph Marginals of Learned Vector Field #
###############################################
sde = MartingaleLossSDE(model, sigma_plot)
simulator = EulerMaruyamaSimulator(sde)
ts = torch.linspace(0, 1, 100).to(device)
record_every_idxs = record_every(len(ts), len(ts) // (num_marginals - 1))
x0 = path.p_simple.sample(num_samples)
xts = simulator.simulate_with_trajectory(
    x0, ts.view(1, -1, 1).expand(num_samples, -1, 1)
)
xts = xts[:, record_every_idxs, :]
for idx in range(xts.shape[1]):
    xx = xts[:, idx, :]
    hist2d_samples(
        samples=xx.cpu(),
        ax=axes[1, idx],
        bins=200,
        scale=scale,
        percentile=99,
        alpha=1.0,
    )
    axes[1, idx].set_xlim(-scale, scale)
    axes[1, idx].set_ylim(-scale, scale)
    axes[1, idx].set_xticks([])
    axes[1, idx].set_yticks([])
    tt = ts[record_every_idxs[idx]]
    axes[1, idx].set_title(f"$t={tt.item():.2f}$", fontsize=15)
axes[1, 0].set_ylabel("Learned", fontsize=20)

plt.show()
