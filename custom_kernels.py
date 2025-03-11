import torch
import gpytorch
from gpytorch.kernels import Kernel
import warnings
from gpytorch.lazy import NonLazyTensor, LazyTensor, DiagLazyTensor, MulLazyTensor, LazyEvaluatedKernelTensor
import numpy as np

class RadialTemporalKernel(Kernel):
    def __init__(self, a=0.9/625, c=0.1, center=25.0, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.c = c
        self.center = center

    def forward(self, x1, x2, diag=False, **params):
        # Compute pairwise Euclidean distances
        # dists = torch.cdist(x1, x2, p=2)
        dists = self.covar_dist(x1, x2, diag=diag, **params)
        # dists = torch.abs(x1 - x2)  # Why are they both the same now?

        # Apply the kernel function
        kernel_matrix = self.a * (dists - self.center) ** 2 + self.c

        if diag:
            warnings.warn('Returning DiagLazyTensor. What is this?')
            # If diagonal, return a DiagLazyTensor
            return DiagLazyTensor(kernel_matrix)
        else:
            # Else, return the full NonLazyTensor matrix
            return NonLazyTensor(kernel_matrix)


class A_Kernel(gpytorch.kernels.Kernel):
    """
    Custom kernel where a(t, t') = 1 - exp(-min(t, t') / 200)
    """
    # has_lengthscale = False  # This kernel does not have a trainable lengthscale
    def __init__(self, a_parameter=200, **kwargs):
        super().__init__(**kwargs)  # Ensure Kernel base class is initialized
        self.a_parameter = a_parameter

    def forward(self, x1, x2, **params):
        """ Compute the lazy kernel matrix for `a(t, t')` """
        t1 = x1[..., -1]  # Extract time components
        t2 = x2[..., -1]

        # Compute a(t, t')
        a_matrix = 1 - torch.exp(-torch.minimum(t1.unsqueeze(1), t2.unsqueeze(0)) / self.a_parameter)
        warnings.warn('Correct unsqueezing?')

        # Return as a LazyTensor for memory efficiency
        return gpytorch.lazy.NonLazyTensor(a_matrix)


class A_Neg_Kernel(gpytorch.kernels.Kernel):
    """
    Custom kernel where a_neg(t, t') = exp(-min(t, t') / 200)
    """
    # has_lengthscale = False
    def __init__(self, a_parameter=200, **kwargs):
        super().__init__(**kwargs)  # Ensure Kernel base class is initialized
        self.a_parameter = a_parameter

    def forward(self, x1, x2, **params):
        """ Compute the lazy kernel matrix for `a_neg(t, t')` """
        t1 = x1[..., -1]  # Extract time components
        t2 = x2[..., -1]

        # Compute a_neg(t, t')
        a_neg_matrix = torch.exp(-torch.minimum(t1.unsqueeze(1), t2.unsqueeze(0)) / self.a_parameter)

        # Return as a LazyTensor
        return gpytorch.lazy.NonLazyTensor(a_neg_matrix)


class Matern12_RBF_WeightedSumKernel(gpytorch.kernels.Kernel):
    is_stationary = False  # Since 'a' is input-dependent, the kernel is non-stationary

    def __init__(self, active_dims, a_parameter, lengthscale_temporal_RBF, lengthscale_temporal_Ma12, output_variance_RBF, output_variance_Ma12, **kwargs):
        super().__init__(**kwargs)
        self.a_parameter = a_parameter
        self.rbf = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(active_dims=active_dims),
            outputscale=output_variance_RBF
        )
        self.rbf.base_kernel.lengthscale = lengthscale_temporal_RBF

        self.matern12 = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(active_dims=active_dims),
            outputscale=output_variance_Ma12
        )
        self.matern12.base_kernel.lengthscale = lengthscale_temporal_Ma12
        self.a_kernel = A_Kernel(a_parameter)
        self.a_neg_kernel = A_Neg_Kernel(a_parameter)

        self.RBF_part = gpytorch.kernels.ProductKernel(self.rbf, self.a_kernel)
        self.Ma12_part = gpytorch.kernels.ProductKernel(self.matern12, self.a_neg_kernel)
        self.temporal_kernel = self.RBF_part + self.Ma12_part

    def forward(self, x1, x2, **params):
        return self.temporal_kernel(x1, x2)


def Matern12_RBF_weighted_sum(t, t_prime, ell_1, ell_2):  # Now the 
    T, T_prime = torch.meshgrid(t, t_prime)  # T is constant row, T_prime is constant column
    # nu = f(t)... Is that possible?
    # Let us have $\ell=5$ for all.
    RBF = torch.exp(-torch.abs(T-T_prime)**2/ell_1)
    Ma12 = 2*torch.exp(-torch.abs(T-T_prime)/ell_2)
    # a should be RBF but on its head
    a = 1 - torch.exp(-(torch.minimum(T, T_prime)-25)**2/200) # just a quite flat Gaussian as weight function
    return a*RBF + (1-a)*Ma12, a   #    # beginning, a  # 


if __name__ == '__main__':
    # This is to plot covariances
    iterations = 50
    lengthscale = 10**-10
    T, T_prime = np.meshgrid(np.linspace(1, iterations, 1000), np.linspace(1, iterations, 1000))  # we need linspace for plotting and not range because of interpolation
    ell_matrix = np.ones_like(T)*lengthscale  # like this is constant
    for i in range(ell_matrix.shape[0]):
        for j in range(ell_matrix.shape[1]):
            ell_matrix[i, j] = (0.5*max(T[i,j], T_prime[i,j]))  # 10**(-6+int(max(T[i,j], T_prime[i,j])))
    K = custom_kernel(T, T_prime, ell=ell_matrix)
    plt.figure()
    plt.contourf(T, T_prime, K, levels=50, cmap='viridis')
    plt.colorbar(label='$k(t, t^\prime)$')
    plt.xlabel('$t$')
    plt.ylabel('$t^\prime$')
    plt.title(f'Kernel $k(t, t^\prime)$ with time-varying $\ell$')
    plt.show()
    plt.savefig(f'custom_kernel_ell_timevarying_iterations_{iterations}.png')


    plt.figure()
    plt.plot(range(len(ell_matrix[:,0])), ell_matrix[:,0])
    plt.yscale('log')
    plt.xlabel('Iteration $t$')
    plt.ylabel('$\ell(t)$')
    plt.savefig('time_varying_lengthscale.png')
