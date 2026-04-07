import torch
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

from distributions import *
from paths import *
from simulation import *

# My implemenation of a TestFunction abstract class. This should make it easier to try out different test functions


class TestFunction(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def func(self):
        pass

    @abstractmethod
    def grad_and_trace(self):
        pass


class Polynomial(TestFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def func(self, x: torch.Tensor):
        """
        Args:
        - x: (bs, t, dim)
        Returns:
        - f(x): (bs, t, dim + dim^2)
        """
        _, dim = x.shape

        linear = x

        quad = []
        for i in range(dim):
            for j in range(dim):
                quad.append((x[:, i] * x[:, j]).unsqueeze(-1))
        quad = torch.cat(quad, dim=-1)

        return torch.cat([linear, quad], dim=-1)

    def grad_and_trace(self, x: torch.Tensor):
        """
        Args:
        - x: (bs, t, dim)
        Returns:
        - grad: (bs, t, K, dim)
        - trace: (bs, t, K)
        """

        batch_size, dim = x.shape
        device = x.device

        # Output dimension
        K = dim + dim * dim

        grad = torch.zeros(batch_size, K, dim, device=device)
        trace = torch.zeros(batch_size, K, device=device)

        # Linear terms
        eye = torch.eye(dim, device=device)
        grad[:, :dim, :] = eye.view(1, dim, dim)

        # Nonlinear terms
        # indices
        i_idx = torch.arange(dim, device=device).repeat_interleave(dim)
        j_idx = torch.arange(dim, device=device).repeat(dim)

        # gradient
        eye = torch.eye(dim, device=device)
        e_i = eye[i_idx]  # (dim^2, dim)
        e_j = eye[j_idx]  # (dim^2, dim)
        grad[:, dim:, :] = x[:, j_idx].unsqueeze(-1) * e_i.unsqueeze(0) + x[
            :, i_idx
        ].unsqueeze(-1) * e_j.unsqueeze(0)

        # trace (Laplacian) of x_i x_j is 2 for i=j and 0 otherwise.
        diag_mask = i_idx == j_idx
        trace[:, dim:] = (2.0 * diag_mask).view(1, -1)

        return grad, trace
