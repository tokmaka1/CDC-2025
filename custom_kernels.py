import torch
import gpytorch
from gpytorch.kernels import Kernel
import warnings
from gpytorch.lazy import NonLazyTensor, LazyTensor, DiagLazyTensor
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


def custom_kernel(t, t_prime, ell):  
    warnings.warn("To use as Pytorch kernel, we need different distance. But with 1D input, this is fine")        
    # But this will always be 1D input since this is time
    T, T_prime = torch.meshgrid(t, t_prime)
    if type(ell) == int:
        pass
    elif ell == 'varying':
        # ell = torch.ones_like(T)
        # for i in range(ell.shape[0]):
        #     for j in range(ell.shape[1]):
        #         ell[i, j] = torch.min(T[i, j], T_prime[i,j])
        ell = torch.minimum(T, T_prime)/5
    exp_weight = torch.exp(-torch.abs(T-T_prime)/ell)  # then also abs is fine
    return (1-exp_weight)*1/torch.maximum(T, T_prime) + exp_weight



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
