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
        - x: (bs, dim)
        Returns:
        - f(x): (bs, dim + dim^2)
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
        bs: batch size
        dim: dimension of data
        K: number of test funcs
        Args:
        - x: (bs, dim)
        Returns:
        - grad: (bs, K, dim)
        - trace: (bs, K)
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


class Hermite(TestFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def func(self, x: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        Returns:
        - f(x): (bs, 4*dim)
        """
        H_zero = torch.ones_like(x)
        H_one = x
        H_two = x**2 + 1
        H_three = x**3 - 3 * x
        return torch.cat([H_zero, H_one, H_two, H_three], dim=-1)

    def grad_and_trace(self, x: torch.Tensor):
        """
        bs: batch size
        dim: dimension of data
        K: number of test funcs
        Args:
        - x: (bs, dim)
        Returns:
        - grad: (bs, 4*dim, dim)
        - trace: (bs, 4*dim)
        """

        batch_size, dim = x.shape
        device = x.device

        grad = torch.zeros(batch_size, 4 * dim, dim, device=device)
        trace = torch.zeros(batch_size, 4 * dim, device=device)

        # grad[:, :dim, :] = torch.zeros(batch_size, dim, dim)
        grad[:, dim : 2 * dim, :] = torch.eye(dim).unsqueeze(0)
        grad[:, 2 * dim : 3 * dim, :] = 2 * torch.diag_embed(x)
        grad[:, 3 * dim :, :] = 3 * (torch.diag_embed(x) ** 2) - 3

        # trace[:,:2*dim] = torch.zeros(batch_size, 2*dim)
        trace[:, 2 * dim : 3 * dim] = 2
        trace[:, 3 * dim :] = 9 * x

        return grad, trace
