import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
from custom_kernels import Matern12_RBF_WeightedSumKernel, MyPeriodicKernel, MyPolynomialKernel
import gpytorch
from pacsbo.pacsbo_main import compute_X_plot
import random
import matplotlib.colors as mcolors
import tikzplotlib





def generating_kernel_paths(kernel, RKHS_norm, iterations_begin, iterations_end, num_X_center_min, num_X_center_max, X_plot):  # Frequentist approach, only pre-RKHS
    num_X_center = random.randint(num_X_center_min, num_X_center_max)
    X_c = iterations_begin + torch.rand(num_X_center)*(iterations_end - iterations_begin)
    X_c, _ = torch.sort(X_c)
    alpha = 2 * torch.rand(len(X_c)) - 1  # in [-1, 1]
    K_cc = kernel(X_c, X_c)
    quadr_form_val = alpha @ K_cc @ alpha
    # warnings.warn('Covariance matrix of temporal kernel?')
    alpha /= torch.sqrt(quadr_form_val)/RKHS_norm
    K_cp = kernel(X_c, X_plot)
    Y_c = alpha @ K_cp # evaluation points = center points
    return Y_c


if __name__ == '__main__':
    kernel = Matern12_RBF_WeightedSumKernel(active_dims=None, lengthscale_temporal_RBF=5, lengthscale_temporal_Ma12=0.5, output_variance_RBF=1, output_variance_Ma12=1)
    list_Y = []
    iterations_begin = 1
    iterations_end = 50
    num_X_plot = 1000  # not realistic because we evaluate on 50 integers
    X_plot = compute_X_plot(1, num_X_plot).flatten()
    X_plot = iterations_begin + X_plot*(iterations_end - iterations_begin)
    for _ in range(2):
        # Y_c = generating_kernel_paths(, RKHS_norm=1)
        Y_c = generating_kernel_paths(kernel=kernel, RKHS_norm=5, iterations_begin=iterations_begin,
                                      iterations_end=iterations_end, num_X_center_min=400, num_X_center_max=800, X_plot=X_plot)
        list_Y.append(Y_c)
    plt.figure()
    for Y_c in list_Y:
        plt.plot(range(1, len(list_Y[0])+1), Y_c.detach().numpy())  #  color='magenta', alpha=0.2)
    plt.xlabel('Iterations $t$')
    plt.xticks(ticks=[0, 200, 400, 600, 800, 1000], labels=[1, 10, 20, 30, 40, 50])
    plt.ylabel('$f(x)$')

    # Plot kernel
    # K = kernel(X_plot, X_plot).to_dense()
    # L = torch.linalg.cholesky(K)
    # a, b = torch.linalg.eigh(K)
    # plt.figure()
    # plt.plot(range(len(a)), a.detach().numpy())


    # weight = kernel.a_kernel(X_plot, X_plot).to_dense()
    # T, T_prime = torch.meshgrid(X_plot, X_plot)
    # plt.figure()
    # plt.contourf(T, T_prime, weight, levels=100, cmap='viridis')
    # plt.colorbar()
    # plt.xticks([])
    # plt.yticks([])
    # plt.xlabel('')
    # plt.ylabel('')
    # plt.title('')
    K = kernel.bm_kernel(X_plot, X_plot).to_dense()
    I = torch.eye(len(K))*1e-4

    L = torch.linalg.cholesky(K+I)  # heuristic check for positive definiteness
    a, b = torch.linalg.eigh(K+I)
    plt.figure()
    T, T_prime = torch.meshgrid(X_plot, X_plot)
    plt.contourf(T, T_prime, K, levels=50, cmap='viridis')
    plt.colorbar()
    # plt.xlabel('$t$')
    # plt.ylabel('$t^\prime$')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.title('')
    plt.savefig('bm_kernel_frac_50.pdf', dpi=1000, bbox_inches='tight', pad_inches=0)



    # plt.savefig('weight_a.pdf', dpi=1000, bbox_inches='tight', pad_inches=0)

    # 
    # plt.xlabel('$t$')
    # plt.ylabel('$t^\prime$')
    # plt.title('Weighting for kernels')

    weight = kernel.a_neg_kernel(X_plot, X_plot).to_dense()
    T, T_prime = torch.meshgrid(X_plot, X_plot)
    plt.figure()
    plt.contourf(T, T_prime, weight, levels=50, cmap='viridis')
    plt.colorbar(label='$a$')
    plt.xlabel('$t$')
    plt.ylabel('$t^\prime$')
    plt.title('Weighting for kernels')


