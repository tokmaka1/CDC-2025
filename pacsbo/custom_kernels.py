import torch
import gpytorch
from gpytorch.kernels import Kernel

from gpytorch.lazy import NonLazyTensor

class RadialTemporalKernel(Kernel):
    def __init__(self, a=0.9/625, c=0.1, center=25.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.c = c
        self.center = center

    def forward(self, x1, x2, diag=False, **params):
        # Compute pairwise Euclidean distances
        dists = torch.cdist(x1, x2, p=2)

        # Apply the kernel function
        kernel_matrix = self.a * (dists - self.center) ** 2 + self.c

        # Return a LazyTensor representation
        return NonLazyTensor(kernel_matrix)
