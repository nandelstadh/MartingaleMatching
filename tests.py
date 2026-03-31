import torch
from tqdm import tqdm

from distributions import *
from ppaths import *


def build_mlp(dims: List[int], activation: Type[torch.nn.Module] = torch.nn.SiLU):
    mlp = []
    for idx in range(len(dims) - 1):
        mlp.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            mlp.append(activation())
    return torch.nn.Sequential(*mlp)


class MLPDrift(torch.nn.Module):
    """
    MLP-parameterization of the learned drift
    """

    def __init__(self, dim: int, hiddens: List[int]):
        super().__init__()
        self.dim = dim
        self.net = build_mlp([dim + 1] + hiddens + [dim])

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - s_t^theta(x): (bs, dim)
        """
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)


path = GaussianConditionalProbabilityPath(
    p_data=GaussianMixture.symmetric_2D(
        nmodes=5, std=PARAMS["target_std"], scale=PARAMS["target_scale"]
    ).to(device),
    alpha=LinearAlpha(),
    beta=SquareRootBeta(),
).to(device)

model = MLPDrift(dim=2, hiddens=[64, 64, 64, 64])

batch_size = 256
steps = 100

z = path.p_data.sample(batch_size)
z = z.unsqueeze(1).repeat(1, steps, 1)
t = torch.linspace(0, 1, steps=steps).to(z)
t = t.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1)
x = path.sample_conditional_path(z, t)
dt = 1 / steps

print(z.shape)
print(t.shape)
print(x.shape)

x_now = x[:, :-1, :]
x_next = x[:, 1:, :]
t_now = t[:, :-1, :]

b_theta = model(x_now, t_now)
residual = x_next - x_now - b_theta * dt
mse = torch.mean(residual) ** 2
print(mse)
