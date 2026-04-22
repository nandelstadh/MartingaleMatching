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
        - f(x): (bs, 5*dim)
        """
        H_zero = torch.ones_like(x)
        H_one = x
        H_two = x**2 - 1
        H_three = x**3 - 3 * x
        H_four = x**4 - 6 * x**2 + 3
        return torch.cat([H_zero, H_one, H_two, H_three, H_four], dim=-1)

    def grad_and_trace(self, x: torch.Tensor):
        """
        bs: batch size
        dim: dimension of data
        K: number of test funcs
        Args:
        - x: (bs, dim)
        Returns:
        - grad: (bs, 5*dim, dim)
        - trace: (bs, 5*dim)
        """

        batch_size, dim = x.shape
        device = x.device

        grad = torch.zeros(batch_size, 5 * dim, dim, device=device)
        trace = torch.zeros(batch_size, 5 * dim, device=device)

        # grad[:, :dim, :] = torch.zeros(batch_size, dim, dim)
        grad[:, dim : 2 * dim, :] = torch.eye(dim).unsqueeze(0)
        grad[:, 2 * dim : 3 * dim, :] = 2 * torch.diag_embed(x)
        grad[:, 3 * dim : 4 * dim, :] = 3 * (torch.diag_embed(x) ** 2) - 3 * (
            torch.eye(dim).unsqueeze(0)
        )
        grad[:, 4 * dim : 5 * dim, :] = 4 * (torch.diag_embed(x) ** 3) - 12 * (
            torch.diag_embed(x)
        )

        # trace[:,:2*dim] = torch.zeros(batch_size, 2*dim)
        trace[:, 2 * dim : 3 * dim] = 2
        trace[:, 3 * dim : 4 * dim] = 6 * x
        trace[:, 4 * dim : 5 * dim] = 12 * x**2 - 12

        return grad, trace


class FourierFeatures(TestFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _phase(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if w.ndim != 2 or w.shape[0] != x.shape[0]:
            raise ValueError("w must have shape (batch_size, num_frequencies)")
        _, dim = x.shape
        if w.shape[1] % dim != 0:
            raise ValueError("w.shape[1] must be a multiple of x.shape[1]")
        return w * x.repeat(1, w.shape[1] // dim)

    def func(self, x: torch.Tensor, w: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        - w: (bs, 50*dim)
        Returns:
        - f(x): (bs, 100*dim)
        """
        phase = self._phase(x, w)
        cos = torch.cos(phase)
        sin = torch.sin(phase)
        return torch.cat([cos, sin], dim=-1)

    def grad_and_trace(self, x: torch.Tensor, w: torch.Tensor):
        """
        Args:
        - x: (bs, dim)
        - w: (bs, 50*dim)
        Returns:
        - grad: (bs, 100*dim, dim)
        - trace: (bs, 100*dim)
        """
        batch_size, dim = x.shape
        device = x.device
        phase = self._phase(x, w)
        num_freq = w.shape[1]

        grad = torch.zeros(batch_size, 2 * num_freq, dim, device=device, dtype=x.dtype)
        trace = torch.zeros(batch_size, 2 * num_freq, device=device, dtype=x.dtype)

        freq_to_dim = (torch.arange(num_freq, device=device) % dim).view(1, num_freq, 1)
        freq_to_dim = freq_to_dim.expand(batch_size, -1, 1)

        grad[:, :num_freq, :].scatter_(
            2, freq_to_dim, (-torch.sin(phase) * w).unsqueeze(-1)
        )
        grad[:, num_freq:, :].scatter_(
            2, freq_to_dim, (torch.cos(phase) * w).unsqueeze(-1)
        )

        w_sq = w * w
        trace[:, :num_freq] = -w_sq * torch.cos(phase)
        trace[:, num_freq:] = -w_sq * torch.sin(phase)
        return grad, trace
