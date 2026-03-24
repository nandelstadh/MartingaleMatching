import torch
import torch.distributions as D
from matplotlib import pyplot as plt
from matplotlib.axes._axes import Axes
from sklearn.datasets import make_circles, make_moons
from torch.func import jacrev, vmap
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from distributions import *
from ppaths import *
from simulation import *

# ### Problem 3.1 Flow Matching with Gaussian Conditional Probability Paths


def build_mlp(dims: List[int], activation: Type[torch.nn.Module] = torch.nn.SiLU):
    mlp = []
    for idx in range(len(dims) - 1):
        mlp.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            mlp.append(activation())
    return torch.nn.Sequential(*mlp)


class MLPVectorField(torch.nn.Module):
    """
    MLP-parameterization of the learned vector field u_t^theta(x)
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
        - u_t^theta(x): (bs, dim)
        """
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)


class Trainer(ABC):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, **kwargs) -> torch.Tensor:
        pass

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(
        self, num_epochs: int, device: torch.device, lr: float = 1e-3, **kwargs
    ) -> torch.Tensor:
        # Start
        self.model.to(device)
        opt = self.get_optimizer(lr)
        self.model.train()

        # Train loop
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(**kwargs)
            loss.backward()
            opt.step()
            pbar.set_description(f"Epoch {idx}, loss: {loss.item()}")

        # Finish
        self.model.eval()


class ConditionalFlowMatchingTrainer(Trainer):
    def __init__(
        self, path: ConditionalProbabilityPath, model: MLPVectorField, **kwargs
    ):
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z = self.path.p_data.sample(batch_size)  # (bs, dim)
        t = torch.rand(batch_size, 1).to(z)  # (bs, 1)
        x = self.path.sample_conditional_path(z, t)  # (bs, dim)

        ut_theta = self.model(x, t)  # (bs, dim)
        ut_ref = self.path.conditional_vector_field(x, z, t)  # (bs, dim)
        error = torch.sum(torch.square(ut_theta - ut_ref), dim=-1)  # (bs,)
        return torch.mean(error)


class LearnedVectorFieldODE(ODE):
    def __init__(self, net: MLPVectorField):
        self.net = net

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: (bs, dim)
            - t: (bs, dim)
        Returns:
            - u_t: (bs, dim)
        """
        return self.net(x, t)


# ### Problem 3.2: Score Matching with Gaussian Conditional Probability Paths


class MLPScore(torch.nn.Module):
    """
    MLP-parameterization of the learned score field
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


class ConditionalScoreMatchingTrainer(Trainer):
    def __init__(self, path: ConditionalProbabilityPath, model: MLPScore, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        z = self.path.p_data.sample(batch_size)  # (bs, dim)
        t = torch.rand(batch_size, 1).to(z)  # (bs, 1)
        x = self.path.sample_conditional_path(z, t)  # (bs, dim)

        s_theta = self.model(x, t)  # (bs, dim)
        s_ref = self.path.conditional_score(x, z, t)  # (bs, dim)
        mse = torch.sum(torch.square(s_theta - s_ref), dim=-1)  # (bs,)
        return torch.mean(mse)


class LangevinFlowSDE(SDE):
    def __init__(self, flow_model: MLPVectorField, score_model: MLPScore, sigma: float):
        """
        Args:
        - path: the ConditionalProbabilityPath object to which this vector field corresponds
        - z: the conditioning variable, (1, dim)
        """
        super().__init__()
        self.flow_model = flow_model
        self.score_model = score_model
        self.sigma = sigma

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        return self.flow_model(x, t) + 0.5 * self.sigma**2 * self.score_model(x, t)

    def diffusion_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        return self.sigma
