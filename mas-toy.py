import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
import warnings


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

        # f(x) = (x1 + x2) - (y_1 + y_2); Then this is just 4-dimensional right? Yes? Yes.


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


class agent_class():
    def __init__(self, delta_confidence, noise_std, number, X_plot, own_actions, action_vector,
                Y_sample, exploration_threshold, gt):
        self.CCP
        self.gt = gt  # at least for toy experiments it works like this.
        # self.RKHS_norm 
        self.exploration_threshold = exploration_threshold
        self.delta_confidence = delta_confidence
        self.X_plot = X_plot
        self.noise_std = noise_std
        self.n_dimensions = X_plot.shape[1]
        self.safety_threshold = gt.safety_threshold
        self.number = number
        self.lambda_bar = max(self.noise_std, 1)
        self.own_actions = own_actions
        self.action_vector = action_vector
        self.global_reward = Y_sample
        self.discr_domain = X_plot

    def compute_actions_reward_model(self, gpr):
        self.actions_reward_model = gpr(train_x=self.action_vector, train_y=self.global_reward, noise_std=self.noise_std, lengthscale=0.1)

    def compute_predicting_others_model(self, gpr):
        self.predicting_others_model = gpr(train_x=self.own_actions, train_y=self.action_vector, noise_std=self.noise_std, lengthscale=0.1)

    def compute_mean_var(self):
        self.actions_reward_model.eval()
        self.f_preds = self.actions_reward_model(self.discr_domain)
        self.mean = self.f_preds.mean
        self.var = self.f_preds.variance

    def compute_confidence_intervals(self):
        warnings.warn("Somewhere RKHS norm")
        self.compute_beta()
        self.lcb = self.mean - self.beta*torch.sqrt(self.var)  # we have to use standard deviation instead of variance
        self.ucb = self.mean + self.beta*torch.sqrt(self.var)

    def compute_safe_set(self):
        self.S = self.lcb > self.safety_threshold
        warnings.warn("Computing safe set in the <pessimistic and lazy> way")

        # Auxiliary objects of potential maximizers M and potential expanders G
        self.G = self.S.clone()
        self.M = self.S.clone()

    def maximizer_routine(self, best_lower_bound_others):
        self.M[:] = False  # initialize
        self.max_M_var = 0  # initialize
        if not torch.any(self.S):  # no safe points
            return
        self.best_lower_bound_local = max(self.lcb[self.S])
        self.M[self.S] = self.ucb[self.S] >= max(best_lower_bound_others, self.best_lower_bound_local)
        self.M[self.M.clone()] = (self.ucb[self.M] - self.lcb[self.M]) > self.exploration_threshold
        if not torch.any(self.M):
            return
        self.max_M_var = torch.max(self.ucb[self.M] - self.lcb[self.M])
        self.max_M_ucb = torch.max(self.ucb[self.M])

    def expander_routine(self):
        self.G[:] = False  # initialize
        if not torch.any(self.S) or torch.all(self.S):  # no safe points or all of them are safe points
            return
        # no need to consider points in M
        if self.compute_all_sets:  # for visualization; introductory example
            s = self.S.clone()
        else:
            s = torch.logical_and(self.S, ~self.M)
            s[s.clone()] = (self.ucb[s] - self.lcb[s]) > self.max_M_var
            s[s.clone()] = (self.ucb[s] - self.lcb[s]) > self.exploration_threshold
        if not torch.any(s):
            return
        potential_expanders = self.discr_domain[s]
        unsafe_points = self.discr_domain[~self.S]
        kernel_distance = self.compute_kernel_distance(potential_expanders, unsafe_points)
        ucb_expanded = self.ucb[s].unsqueeze(1).expand(-1, kernel_distance.size(1))
        s[s.clone()] = torch.any(ucb_expanded - self.B*kernel_distance > self.safety_threshold, dim=1)
        self.G = s
        warnings.warn("Computing expanders in discrete way like Sui et al. 2015")

    def compute_beta(self):
        # Fiedler et al. 2024 Equation (7); based on Abbasi-Yadkori 2013
        inside_log = torch.det(torch.eye(len(self.global_rewards)) + (1/self.noise_std*self.K))
        inside_sqrt = self.noise_std*torch.log(inside_log) - (2*self.noise_std*torch.log(torch.tensor(self.delta_confidence)))
        self.beta = self.B + torch.sqrt(inside_sqrt)

    def compute_kernel_distance(self, x, x_prime):  # let us try whether it works without reshaped!
        '''
        k(x,x)+k(x^\prime,x^\prime)-k(x,x^\prime)-k(x^\prime,x)=2-2k(x,x^\prime)
        This holds for all radial kernels with output variance 1, i.e., k(x,x)\equiv 1.
        Both of which are true for our case.
        We have this setting and we exploit it.
        '''
        # print('Before kernel operation')
        if self.actions_reward_model.kernel.__class__.__name__ != 'MaternKernel' and self.actions_reward_model.kernel.__class__.__name__ != 'RBFKernel':
            raise Exception("Current implementation only works with radial kernels.")
        matrix_containing_kernel_values = self.actions_reward_model.kernel(x, x_prime).evaluate()  # here we can have problems with the size of the matrix
        # print('After kernel operation')
        return torch.sqrt(2-2*matrix_containing_kernel_values)

    # def update(self):
        # To update models(s), samples etc. No need to build the agent completely new
        # Not important right now. We will switch to BOTorch anyways so...

    def predict_action_space(self):
        self.compute_predicting_others_model(gpr=GPRegressionModel)
        self.compute_predicting_others_model.eval()
        self.f_preds_predicting_others_model = self.predicting_others_model(self.own_actions[-1])  # we give the own action(s), and get the others one action!
        last_action_space = self.f_preds_predicting_others_model.mean
        # Do some concatenation stuff here
        self.action_space = last_action_space  # combined with ...
        # return last_action_space

    def process_communication(self):
        return action_space

    def opt(self):
        if not torch.any(torch.logical_or(self.M, self.G)):
            raise Exception("No interesting point to evaluate!")
        x_next = self.discr_domain[torch.logical_or(self.M, self.G)][torch.argmax(self.var[torch.logical_or(self.M, self.G)])]
        return x_next


# Hyperparameters
noise_std = 0.01
delta_confidence = 0.1  # yields 90% confidence for safety proof.
num_safe_points = 1  # singleton safe set
num_iterations = 10  # number of total points at the end
exploration_threshold = 0.1  # let us start with that, maybe we should decrease to 0. But let's see
n_dimensions = 2  # 2 agents with 2 dimensions; but then we cannot really plot
points_per_axis = 500  # 30 for 4D, 1000 for 1D, 500 for 2D, 100 for 3D, 15 for 5D
X_plot = compute_grid(n_dimensions, points_per_axis)
# We also need one-dimensional X_plot for the other GP
RKHS_norm = 1  # RKHS norm of that one ground truth
gt = ground_truth(num_center_points=1000, X_plot=X_plot, RKHS_norm=RKHS_norm)
action_vector_init, global_reward_init = initial_safe_samples(gt=gt, num_safe_points=num_safe_points)  # this is wrong btw the ground truth needs 2 2-dimensional variables
action_vector = action_vector_init.clone()
global_reward = global_reward_init.clone()

agent0 = agent_class(delta_confidence, noise_std, 0, X_plot, action_vector[:, 0], action_vector, global_reward, exploration_threshold, gt)
agent1 = agent_class(delta_confidence, noise_std, 1, X_plot, action_vector[:, 1], action_vector, global_reward, exploration_threshold, gt)
list_agents = [agent0, agent1]

# self, delta_confidence, noise_std, number, X_plot, own_actions, action_vector

# But we need something to begin with... So we have the complete action space in the beginning

for i in range(num_iterations):  # 10 steps
    for agent in list_agents:
        if i != 0 or (bool(i % 2) and agent.number == 0) or \
           (not bool(i % 2) and agent.number == 1):
            agent.process_communication(communicated_action)  # updates self.action_vector
        elif i != 0:
            agent.predict_action_space()  # updates self.action_vector
        '''
        The action vector at this very moment is complete in the sense of it has all the action
        (communicated or predicted)
        '''
        agent.compute_actions_reward_model(gpr=GPRegressionModel)
        agent.compute_mean_var()
        agent.maximizer_routine()
        agent.expander_routine()

        # This is the information that will be processed by the next guy in the next iteration
        communicated_action = agent.own_actions[-1]

'''
The problem is also we need to optimize within each agent, then send the current idea to the global reward system and then receive an input
We need another GP to learn about the other's action. Or some other function that basically predicts what the other agent is doing. This can for 
now also be a black-box random function!!
'''
