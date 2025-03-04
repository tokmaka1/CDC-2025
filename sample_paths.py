import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
from custom_kernels import custom_kernel
import gpytorch
from pacsbo.pacsbo_main import compute_X_plot
import random




def generating_kernel_paths(kernel, RKHS_norm, iterations_begin, iterations_end, num_X_center_min, num_X_center_max, X_plot, ell):  # Frequentist approach, only pre-RKHS
    num_X_center = random.randint(num_X_center_min, num_X_center_max)
    X_c = iterations_begin + torch.rand(num_X_center)*(iterations_end - iterations_begin)
    alpha = torch.rand(len(X_c))
    quadr_form_val = alpha @ kernel(X_c, X_c, ell) @ alpha
    # warnings.warn('Covariance matrix of temporal kernel?')
    alpha /= torch.sqrt(quadr_form_val)*RKHS_norm
    Y_c = alpha @ kernel(X_c, X_plot, ell)  # evaluation points = center points
    return Y_c



if __name__ == '__main__':
    list_Y = []
    iterations_begin = 1
    iterations_end = 50
    num_X_plot = 1000
    X_plot = compute_X_plot(1, 1000).flatten()
    X_plot = iterations_begin + X_plot*(iterations_end - iterations_begin)
    for _ in range(50):
        # Y_c = generating_kernel_paths(, RKHS_norm=1)
        Y_c = generating_kernel_paths(kernel=custom_kernel, RKHS_norm=1, iterations_begin=iterations_begin,
                                      iterations_end=iterations_end, num_X_center_min=400, num_X_center_max=800, X_plot=X_plot, ell=5)
        list_Y.append(Y_c)
    plt.figure()
    for Y_c in list_Y:
        plt.plot(range(1, len(list_Y[0])+1), Y_c.detach().numpy())  # , color='magenta', alpha=0.2)
    plt.xlabel('Iterations $t$')
    plt.xticks(ticks=[0,200,400,600,800,1000], labels=[1,10,20,30,40,50])
    plt.ylabel('$y$')
    plt.title('Sample paths of RKHS of custom kernel')
