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
    T, T_prime = torch.meshgrid(t, t_prime)  # T is constant row, T_prime is constant column
    if type(ell) == int:
        pass
    elif ell == 'varying':
        ell = torch.ones_like(T)
        for i in range(ell.shape[0]):
            for j in range(ell.shape[1]):
                if min(T[i, j], T_prime[i, j]) < 40:
                    ell[i, j] = 1e-3
                # elif min(T[i, j], T_prime[i, j]) < 4:
                #     ell[i, j] = 1
                else:
                    ell[i, j] = 1
        # ell = torch.minimum(T, T_prime)/5

    exp_beginning = torch.exp(-torch.abs(T-T_prime)**2/ell)  # then also abs is fine
    K_beginning = 1/(torch.maximum(T, T_prime)) # we can also have this "constant and small" then this will go to 0; **0.1 if this diminishes too quickly; but still does not help
    # constant and large will remain at the initial value
    # for i in range(K_beginning.shape[0]):  # Go through all rows
    #     for j in range(K_beginning.shape[1]):  # go through all columns
    #         b = torch.rand(1)/torch.max(T[i,j], T_prime[i,j])
    #         K_beginning[i, j] = b.item()
    beginning = (1-exp_beginning)*K_beginning + exp_beginning
    # a = 1 - torch.exp(-torch.minimum(T, T_prime)/30)
    a = 1 - (torch.minimum(T, T_prime)/50)**2  # iteration_end; this is the most natural connection
    end = torch.exp(-torch.abs(T-T_prime)**2/1)  # make it smooth and put **2
    # this random 1 is good. But just a bit differently. Not THAT much roughness. Other weights. History dependence maye a bit more. Let's see.
    # Random walk? Random branching?
    # We can weight this with reward
    # Brownian motion? Kronecker delta? Make sure this itself is a valud kernel, then proving PD is easy.
    return a*beginning + (1-a)*end, a   #    # beginning, a  # 


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
