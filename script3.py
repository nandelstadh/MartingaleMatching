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

num_epochs = 5000
batch_size = 8
steps = 1000
dim = 2
tfunc = Polynomial()
sigma = 0
dt = 1 / (steps - 1)

p_data = DigitsSampleable(batch_size=batch_size, device=device)

# Construct conditional probability path
path = GaussianConditionalProbabilityPath(
    p_data=p_data,
    alpha=LinearAlpha(),
    beta=SquareRootBeta(),
    dt=dt,
    sigma=sigma,
).to(device)

model = MLPDrift(dim=784, hiddens=[64, 64, 64, 64])

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


##########################
# Play around With These #
##########################
num_samples = 50000
num_marginals = 5
