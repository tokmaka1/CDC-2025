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


class MyPeriodicKernel(gpytorch.kernels.Kernel):
    def __init__(self, lengthscale=1.0, period=1.0, **kwargs):
        super().__init__(**kwargs)
        self.ell = lengthscale
        self.period = period

    def forward(self, t1, t2, **params):
        sine_squared = torch.sin(torch.pi * torch.abs(t1 - t2.T) / self.period)**2
        # Compute the kernel matrix
        periodic_kernel = torch.exp(-2*sine_squared/(2*self.ell**2))
        return gpytorch.lazy.NonLazyTensor(periodic_kernel)


class MyRQKernel(gpytorch.kernels.Kernel):
    def __init__(self, lengthscale=25.0, alpha=2.0, **kwargs):
        super().__init__(**kwargs)
        self.ell = lengthscale
        self.alpha = alpha

    def forward(self, t1, t2, **params):
        # Compute the kernel matrix
        value = (1 + torch.abs(t1-t2.T)/(2*self.alpha*self.ell**2))  ** (-self.alpha)
        return gpytorch.lazy.NonLazyTensor(value)


class MyPolynomialKernel(gpytorch.kernels.Kernel):
    def __init__(self, d=2.0, c=1.0, **kwargs):
        super().__init__(**kwargs)
        self.d = d
        self.c = c

    def forward(self, t1, t2, **params):
        K = (t1 - self.c) @ (t2.T - self.c)
        return gpytorch.lazy.NonLazyTensor(K)
    

class BrowianMotionKernel(gpytorch.kernels.Kernel):
    def __init__(self, active_dims, **kwargs):
        super().__init__(active_dims=active_dims, **kwargs)

    def forward(self, t1, t2, **params):
        K = torch.minimum(t1, t2.T)/50
        return gpytorch.lazy.NonLazyTensor(K)

class CustomBrowianMotionKernel(gpytorch.kernels.Kernel):
    def __init__(self, active_dims, **kwargs):
        super().__init__(active_dims=active_dims, **kwargs)

    def forward(self, t1, t2, **params):
        K = torch.minimum(50-t1, 50-t2.T)/50
        return gpytorch.lazy.NonLazyTensor(K)


class Matern12_RBF_WeightedSumKernel(gpytorch.kernels.Kernel):
    is_stationary = False  # Since 'a' is input-dependent, the kernel is non-stationary

    def __init__(self, active_dims, lengthscale_temporal_RBF, lengthscale_temporal_Ma12, output_variance_RBF, output_variance_Ma12, **kwargs):
        super().__init__(**kwargs)
        self.rbf = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(active_dims=active_dims)#,
            # outputscale=output_variance_RBF
        )
        self.rbf.base_kernel.lengthscale = lengthscale_temporal_RBF
        self.rbf.outputscale = output_variance_RBF  # Ensure this is correctly set
        self.rbf.base_kernel.raw_lengthscale.requires_grad = False
        self.rbf.raw_outputscale.requires_grad = False

        self.matern12 = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1/2, active_dims=active_dims),
        )
        self.matern12.base_kernel.lengthscale = lengthscale_temporal_Ma12
        self.matern12.outputscale = output_variance_Ma12
        self.matern12.base_kernel.raw_lengthscale.requires_grad = False
        self.matern12.raw_outputscale.requires_grad = False

        # self.a_kernel = A_Kernel(a_parameter)
        # self.a_neg_kernel = (a_parameter)
        # self.periodic_kernel = MyPeriodicKernel(period=30, lengthscale=0.5)
        # self.rq_kernel = MyRQKernel(lengthscale=25.0, alpha=2.0)
        self.bm_kernel = BrowianMotionKernel(active_dims=active_dims)
        self.cbm_kernel = CustomBrowianMotionKernel(active_dims=active_dims)

        self.bm_cbm_product = gpytorch.kernels.ProductKernel(self.cbm_kernel, self.bm_kernel)
        # self.RBF_part = gpytorch.kernels.ProductKernel(self.rbf, self.periodic_kernel)
        self.RBF_part = self.rbf  # gpytorch.kernels.ProductKernel(self.rbf, self.cbm_kernel)
        self.Ma12_part = gpytorch.kernels.ProductKernel(self.matern12, self.bm_kernel)
        self.Ma12_part = gpytorch.kernels.ProductKernel(self.Ma12_part, self.cbm_kernel)
        # self.temporal_kernel = self.RBF_part + self.Ma12_part
        # self.temporal_kernel = self.RBF_part + self.Ma12_part
        self.temporal_kernel = self.Ma12_part + self.Ma12_part  # +

    def forward(self, x1, x2, **params):
        return self.temporal_kernel(x1, x2)  # self.temporal_kernel(x1, x2)


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
