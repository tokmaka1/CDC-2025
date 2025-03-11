import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
from custom_kernels import Matern12_RBF_weighted_sum
import gpytorch
from pacsbo.pacsbo_main import compute_X_plot
import random
import matplotlib.colors as mcolors





def generating_kernel_paths(kernel, RKHS_norm, iterations_begin, iterations_end, num_X_center_min, num_X_center_max, X_plot, ell_1, ell_2):  # Frequentist approach, only pre-RKHS
    num_X_center = random.randint(num_X_center_min, num_X_center_max)
    X_c = iterations_begin + torch.rand(num_X_center)*(iterations_end - iterations_begin)
    X_c, _ = torch.sort(X_c)
    alpha = 2 * torch.rand(len(X_c)) - 1  # in [-1, 1]
    K_cc, _ = kernel(X_c, X_c, ell_1, ell_2)
    quadr_form_val = alpha @ K_cc @ alpha
    # warnings.warn('Covariance matrix of temporal kernel?')
    alpha /= torch.sqrt(quadr_form_val)/RKHS_norm
    K_cp, _ = kernel(X_c, X_plot, ell_1, ell_2)
    Y_c = alpha @ K_cp # evaluation points = center points
    return Y_c


if __name__ == '__main__':
    list_Y = []
    iterations_begin = 1
    iterations_end = 50
    num_X_plot = 1000  # not realistic because we evaluate on 50 integers
    X_plot = compute_X_plot(1, num_X_plot).flatten()
    X_plot = iterations_begin + X_plot*(iterations_end - iterations_begin)
    for _ in range(20):
        # Y_c = generating_kernel_paths(, RKHS_norm=1)
        Y_c = generating_kernel_paths(kernel=Matern12_RBF_weighted_sum, RKHS_norm=5, iterations_begin=iterations_begin,
                                      iterations_end=iterations_end, num_X_center_min=400, num_X_center_max=800, X_plot=X_plot, ell_1=5, ell_2=0.5)
        list_Y.append(Y_c)
    plt.figure()
    for Y_c in list_Y:
        plt.plot(range(1, len(list_Y[0])+1), Y_c.detach().numpy())  #  color='magenta', alpha=0.2)
    plt.xlabel('Iterations $t$')
    plt.xticks(ticks=[0, 200, 400, 600, 800, 1000], labels=[1, 10, 20, 30, 40, 50])
    plt.ylabel('$f(x)$')
    plt.title('Sample paths combined RBF and Matern12 kernel')

    # Plot kernel
    K, a = Matern12_RBF_weighted_sum(X_plot, X_plot, ell_1=5, ell_2=1)
    T, T_prime = torch.meshgrid(X_plot, X_plot)
    plt.figure()
    # plt.contourf(T, T_prime, K, levels=50, cmap='viridis')
    # plt.colorbar(label='$k(t, t^\prime)$')
    norm = mcolors.LogNorm(vmin=K.min().item(), vmax=K.max().item())  # Apply logarithmic scaling
    contour = plt.contourf(T, T_prime, K, levels=50, cmap='viridis', norm=norm)
    plt.colorbar(contour, label='$k(t, t^\prime)$')

    K, a = Matern12_RBF_weighted_sum(X_plot, X_plot, ell_1=5, ell_2=1)
    T, T_prime = torch.meshgrid(X_plot, X_plot)
    plt.figure()
    plt.contourf(T, T_prime, K, levels=50, cmap='viridis')
    plt.colorbar(label='$k(t, t^\prime)$')

    K, a = Matern12_RBF_weighted_sum(X_plot, X_plot, ell_1=5, ell_2=1)
    T, T_prime = torch.meshgrid(X_plot, X_plot)
    plt.figure()
    plt.contourf(T, T_prime, a, levels=50, cmap='viridis')
    plt.colorbar(label='$a$')
    plt.xlabel('$t$')
    plt.ylabel('$t^\prime$')
    plt.title('Weighting for kernels')

