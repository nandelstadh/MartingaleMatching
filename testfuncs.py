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
    def grad_and_hess(self):
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
        _, _, dim = x.shape

        linear = x

        quad = []
        for i in range(dim):
            for j in range(dim):
                quad.append((x[:, :, i] * x[:, :, j]).unsqueeze(-1))
        quad = torch.cat(quad, dim=-1)

        return torch.cat([linear, quad], dim=-1)

    def grad_and_hess(self, x: torch.Tensor):
        """
        Args:
        - x: (bs, t, dim)
        Returns:
        - grad: (bs, t, K)
        - hess: (bs, t, K, K)
        """
        batch_size, steps, dim = x.shape

        # Output dimension
        K = dim + dim * dim

        grad = torch.zeros(batch_size, steps, K, dim)
        hess = torch.zeros(batch_size, steps, K, dim, dim)

        k = 0

        # Linear terms: f_i = x_i
        for i in range(dim):
            grad[:, k, i] = 1.0
            # Hessian = 0
            k += 1

        # Quadratic terms: f_ij = x_i x_j
        for i in range(dim):
            for j in range(dim):
                # gradient
                grad[:, k, i] = x[:, j]
                grad[:, k, j] = x[:, i]

                # Hessian
                hess[:, k, i, j] = 1.0
                hess[:, k, j, i] = 1.0

                k += 1

        return grad, hess
