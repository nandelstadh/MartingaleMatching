import torch
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

from torch.distributions.multivariate_normal import MultivariateNormal

from distributions import *
from paths import *
from simulation import *
from testfuncs import *

# ### Problem 3.1 Flow Matching with Gaussian Conditional Probability Paths


def build_mlp(dims: List[int], activation: Type[torch.nn.Module] = torch.nn.SiLU):
    mlp = []
    for idx in range(len(dims) - 1):
        mlp.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            mlp.append(activation())
    return torch.nn.Sequential(*mlp)


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


# Maringale matching


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


class MartingaleMatchingTrainer(Trainer):
    def __init__(
        self,
        path: ConditionalProbabilityPath,
        model: MLPDrift,
        steps: int,
        dim: int,
        tfunc: Polynomial,
        sigma: float,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.steps = steps
        self.dim = dim
        self.path = path
        self.tfunc = tfunc
        self.sigma = sigma

    def get_train_loss(self, batch_size: int) -> torch.Tensor:

        dt = 1 / (self.steps - 1)
        z = self.path.p_data.sample(batch_size)
        t = torch.rand(batch_size, 1).to(z) * (1 - dt)  # (bs, 1)
        x = self.path.sample_conditional_path(z, t)

        b_theta = self.model(x, t)
        mean = torch.zeros(self.dim)
        covariance = torch.eye(self.dim)
        mult_norm = MultivariateNormal(mean, covariance)
        epsilon = mult_norm.sample((batch_size,))

        # x_next = x + dt * b_theta + self.sigma * math.sqrt(dt) * epsilon
        x_next = self.path.sample_conditional_path(z, t + dt)

        grad, trace = self.tfunc.grad_and_trace(x)

        drift = (grad * b_theta.unsqueeze(1)).sum(-1)
        diffusion = self.sigma * self.sigma * trace
        generator = drift + 0.5 * diffusion

        f_now = self.tfunc.func(x)
        f_next = self.tfunc.func(x_next)

        residual = f_next - f_now - dt * generator
        mse = torch.mean(residual**2)
        return mse


class MartingaleLossSDE(SDE):
    def __init__(self, drift: MLPDrift, sigma: float):
        """
        Args:
        - path: the ConditionalProbabilityPath object to which this vector field corresponds
        - z: the conditioning variable, (1, dim)
        """
        super().__init__()
        self.drift = drift
        self.sigma = sigma

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        return self.drift(x, t)

    def diffusion_coefficient(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            - x: state at time t, shape (bs, dim)
            - t: time, shape (bs,.)
        Returns:
            - u_t(x|z): shape (batch_size, dim)
        """
        return self.sigma
