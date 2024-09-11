import numpy as np
import torch
import casadi
import gpytorch
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# General function definitions
class ground_truth():
    def __init__(self, num_center_points, X_plot, RKHS_norm):
        def fun(kernel, alpha):
            return lambda X: kernel(X.reshape(-1, self.X_center.shape[1]), self.X_center).detach().numpy() @ alpha
        # For ground truth
        self.X_plot = X_plot
        self.RKHS_norm = RKHS_norm
        random_indices_center = torch.randint(high=self.X_plot.shape[0], size=(num_center_points,))
        self.X_center = self.X_plot[random_indices_center]
        alpha = np.random.uniform(-1, 1, size=self.X_center.shape[0])
        self.kernel = gpytorch.kernels.MaternKernel(nu=3/2)
        self.kernel.lengthscale = 0.1
        RKHS_norm_squared = alpha.T @ self.kernel(self.X_center, self.X_center).detach().numpy() @ alpha
        alpha /= np.sqrt(RKHS_norm_squared)/RKHS_norm  # scale to RKHS norm
        self.f = fun(self.kernel, alpha)
        self.fX = torch.tensor(self.f(self.X_plot), dtype=torch.float32)
        self.safety_threshold = np.quantile(self.fX, 0.3)  # np.quantile(self.fX, np.random.uniform(low=0.15, high=0.5))


class GPRegressionModel(gpytorch.models.ExactGP):  # this model has to be build "new"
    def __init__(self, train_x, train_y, noise_std, n_devices=1, output_device=torch.device('cpu'), lengthscale=0.1):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = torch.tensor(noise_std**2)
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.kernel = gpytorch.kernels.MaternKernel(nu=2.5)
        # self.kernel = gpytorch.kernels.rbf_kernel.RBFKernel()
        self.kernel.lengthscale = lengthscale
        # self.base_kernel.lengthscale.requires_grad = False; somehow does not work
        if output_device.type != 'cpu':
            self.covar_module = gpytorch.kernels.MultiDeviceKernel(
                self.kernel, device_ids=range(n_devices), output_device=output_device)
        else:
            self.covar_module = self.kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def compute_grid(n_dimensions, points_per_axis):  # we do not want to use this anymore!
    X_per_domain = torch.linspace(0, 1, points_per_axis)
    X_per_domain_nd = [X_per_domain] * n_dimensions
    X_grid = torch.cartesian_prod(*X_per_domain_nd).reshape(-1, n_dimensions)
    return X_grid


def initial_safe_samples(gt, num_safe_points):
    fX = gt.fX
    num_safe_points = num_safe_points
    # sampling_logic = torch.logical_and(fX > np.quantile(fX, 0.8), fX < np.quantile(fX, 0.99))
    # sampling_logic = fX > gt.safety_threshold
    sampling_logic = torch.logical_and(fX > np.quantile(fX, 0.4), fX < np.quantile(fX, 0.50))
    random_indices_sample = torch.randint(high=X_plot[sampling_logic].shape[0], size=(num_safe_points,))
    X_sample = X_plot[sampling_logic][random_indices_sample]
    Y_sample = fX[sampling_logic][random_indices_sample] + torch.tensor(np.random.normal(loc=0, scale=noise_std, size=X_sample.shape[0]), dtype=torch.float32)
    return X_sample, Y_sample



# Create function
lengthscale = 0.1
noise_std = 1e-2
RKHS_norm = 1
X_plot = compute_grid(n_dimensions=1, points_per_axis=1000)
gt = ground_truth(num_center_points=250, X_plot=X_plot, RKHS_norm=RKHS_norm)  # cannot pickle this object
delta_confidence = 0.1  # 90% confidence

# Initialize safe BO
X_sample_init, Y_sample_init = initial_safe_samples(gt=gt, num_safe_points=1)
X_sample = X_sample_init.clone()
Y_sample = Y_sample_init.clone()


def kernel_distance(x, x_prime):  # inputs have to be tensors
    # wlog k(x,x)+k(x^\prime,x^\prime)-k(x,x^\prime)-k(x^\prime,x)=2-2k(x,x^\prime)
    return 2 - model.kernel(x, x_prime).evaluate()


def mean(x):  # x has to be torch.tensor
    x_tensor = torch.tensor(x, dtype=torch.float32)
    covariance_vector = model.kernel(X_sample, x_tensor).evaluate()
    m = covariance_vector.T @ K_inverse @ Y_sample
    return m.detach().numpy()


def covariance(x):  # x has to be torch.tensor
    # wlog k(x,x) = 1
    x_tensor = torch.tensor(x, dtype=torch.float32)
    covariance_vector = model.kernel(X_sample, x_tensor).evaluate()
    cov = (1 - covariance_vector.T @ K_inverse @ covariance_vector)
    return cov.detach().numpy()


def neg_GP_UCB(x):
    return -(mean(x) + beta*np.sqrt(covariance(x)))


def optimistic_safe_set(x):  # this is a constraint function
    pass


def pessimistic_safe_set(x):  # this is a constraint function
    pass


for i in range(50):
    model = GPRegressionModel(train_x=X_sample, train_y=Y_sample, noise_std=noise_std, lengthscale=lengthscale)  # no reusuable model etc... not yet important!
    K = model(X_sample).covariance_matrix
    K_inverse = torch.inverse(K + noise_std**2*torch.eye(X_sample.shape[0]))  # We can reuse the covariance matrix inverse
    # Compute beta
    inside_log = torch.det(torch.eye(X_sample.shape[0]) + (1/noise_std*K))
    beta_tensor = RKHS_norm + torch.sqrt(noise_std*torch.log(inside_log) - (2*noise_std*torch.log(torch.tensor(delta_confidence))))
    beta = beta_tensor.detach().numpy()
    # Implementation vs. theory; we will just use Q instead of C...

    # Let us do it without safety first; We can do GP-UCB, but we do a self-implementation
    x0 = torch.tensor([0])  # torch.tensor([np.random.uniform(0, 1)])
    x_next = torch.tensor(minimize(neg_GP_UCB, x0, method='L-BFGS-B', bounds=[(0, 1)]).x, dtype=torch.float32)
    y_new = torch.tensor(gt.f(x_next), dtype=torch.float32)  # let us do it noise-free
    X_sample = torch.cat((X_sample, x_next.unsqueeze(0)), dim=0)
    Y_sample = torch.cat((Y_sample, y_new), dim=0)

    if i % 10 == 0:
        plt.figure()
        model.eval()
        f_preds = model(X_plot)  # now this is where the discretization also kicks in; we can define the mean and covariance function ourselves! Did
        mu = f_preds.mean.detach().numpy()
        std = torch.sqrt(f_preds.variance).detach().numpy()
        plt.plot(X_plot, gt.f(X_plot), '-b', label='ground truth')
        plt.plot(X_plot, mu, '-k', label='GP mean')
        plt.fill_between(X_plot.flatten(), mu - beta*std, mu + beta*std, alpha=0.1, label='confidence')
        plt.scatter(X_sample, Y_sample, color='black')
        plt.scatter(X_sample[-1], Y_sample[-1], color='red', s=100, label='Last sample')
        plt.legend()
        plt.show()
print('Hallo')