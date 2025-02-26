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
    exp_weight = np.exp(-np.abs(t-t_prime)/ell)
    return (1-exp_weight)*1/np.maximum(t, t_prime) + exp_weight





def generating_kernel_paths(kernel, RKHS_norm):  # Frequentist approach, only pre-RKHS
    X_c = torch.arange(1, 50).float()
    alpha = torch.rand(49)
    K = kernel(X_c, X_c)
    quadr_form_val = alpha @ K @ alpha
    warnings.warn('Covariance matrix of temporal kernel?')
    alpha /= torch.sqrt(quadr_form_val)*RKHS_norm
    Y_c = alpha @ K  # evaluation points = center points
    return Y_c


def generate_custom_kernel_paths(RKHS_norm):  # Frequentist approach, only pre-RKHS
    T = np.linspace(1, 50, 1000)
    X, Y = np.meshgrid(T, T)
    ell_matrix = np.ones_like(X)  # like this is constant
    for i in range(ell_matrix.shape[0]):
        for j in range(ell_matrix.shape[1]):
            # ell_matrix[i, j] = (1e-2*max(X[i,j], Y[i,j])**2)
            max_time = max(X[i,j], Y[i,j])
            if max_time < 5:
                ell_matrix[i,j] = 1e-8
            elif max_time < 10:
                ell_matrix[i,j] = 1e-7
            elif max_time < 15:
                ell_matrix[i,j] = 1e-6
            elif max_time < 20:
                ell_matrix[i,j] = 1e-5
            elif max_time < 25:
                ell_matrix[i,j] = 1e-4
            elif max_time < 30:
                ell_matrix[i,j] = 1e-3
            elif max_time < 35:
                ell_matrix[i,j] = 1e-2
            elif max_time < 40:
                ell_matrix[i,j] = 1e-1
            elif max_time < 45:
                ell_matrix[i,j] = 1
            elif max_time <= 50:
                ell_matrix[i,j] = 10

 
    K = custom_kernel(X, Y, ell=ell_matrix)  # This is the way to get covariance kernel with my implementation
    alpha = np.random.rand(1000)
    quadr_form_val = alpha @ K @ alpha
    alpha /= np.sqrt(quadr_form_val)*RKHS_norm
    Y_c = alpha @ K  # evaluation points = center points
    return Y_c




if __name__ == '__main__':
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