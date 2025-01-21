import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
import warnings
import copy
from tqdm import tqdm


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

    def conduct_experiment(self, x, noise_std):
        return torch.tensor(self.f(x))  # noise-free in the beginning + #np.random.normal(loc=0, scale=noise_std, size=1), dtype=x.dtype)


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
    def __init__(self, delta_confidence, noise_std, number, X_plot, actions, exploration_threshold, gt):
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
        self.actions = actions
        self.rewards = []  # just initialize
        self.action_domain = X_plot

    def compute_actions_reward_model(self, gpr):
        self.actions_reward_model = gpr(train_x=self.action_vector, train_y=self.global_reward, noise_std=self.noise_std, lengthscale=0.1)

    def compute_predicting_others_model(self, gpr):
        self.predicting_others_model = gpr(train_x=self.own_actions, train_y=self.action_vector, noise_std=self.noise_std, lengthscale=0.1)

    def compute_mean_var_confidence_intervals_GP1(self, domain):
        self.GP1.eval()
        self.GP1.f_preds = self.GP1(domain)
        self.GP1.mean = self.GP1.f_preds.mean
        self.GP1.var = self.GP1.f_preds.variance
        # warnings.warn("Somewhere RKHS norm")
        self.GP1.beta = self.compute_beta()
        self.GP1.lcb = self.GP1.mean - self.GP1.beta*torch.sqrt(self.GP1.var)
        self.GP1.ucb = self.GP1.mean + self.GP1.beta*torch.sqrt(self.GP1.var)

    def compute_mean_var_confidence_intervals_GP2(self):
        self.GP2.eval()
        self.GP2.f_preds = self.GP2(self.action_domain)
        self.GP2.mean = self.GP2.f_preds.mean
        self.GP2.var = self.GP2.f_preds.variance
        # warnings.warn("Somewhere RKHS norm")
        self.GP2.beta = self.compute_beta()
        self.GP2.lcb = self.GP2.mean - self.GP2.beta*torch.sqrt(self.GP2.var)
        self.GP2.ucb = self.GP2.mean + self.GP2.beta*torch.sqrt(self.GP2.var)


    def compute_beta(self):
        # Fiedler et al. 2024 Equation (7); based on Abbasi-Yadkori 2013
        return 3
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


    def GPUCB(self, GP):
        # acquisition function
        max_index = torch.argmax(GP.ucb)
        return self.action_domain[max_index]  # action VECTOR!



# Hyperparameters
noise_std = 0.01
delta_confidence = 0.1  # yields 90% confidence for safety proof.
num_safe_points = 1  # singleton safe set
num_iterations = 10  # number of total points at the end
exploration_threshold = 0.1  # let us start with that, maybe we should decrease to 0. But let's see
n_dimensions = 2  # 2 agents with each 1 dimension
points_per_axis = 500  # 30 for 4D, 1000 for 1D, 500 for 2D, 100 for 3D, 15 for 5D
X_plot = compute_grid(n_dimensions, points_per_axis)
# We also need one-dimensional X_plot for the other GP
RKHS_norm = 1  # RKHS norm of that one ground truth
gt = ground_truth(num_center_points=1000, X_plot=X_plot, RKHS_norm=RKHS_norm)
action_vector_init, _ = initial_safe_samples(gt=gt, num_safe_points=num_safe_points)  # this is wrong btw the ground truth needs 2 2-dimensional variables
action_vector = action_vector_init.clone()
# global_reward = global_reward_init.clone()

agent0 = agent_class(delta_confidence, noise_std, 0, X_plot, action_vector[:, 0], exploration_threshold, gt)
agent1 = agent_class(delta_confidence, noise_std, 1, X_plot, action_vector[:, 1], exploration_threshold, gt)
list_agents = [agent0, agent1]

# self, delta_confidence, noise_std, number, X_plot, own_actions, action_vector

# But we need something to begin with... So we have the complete action space in the beginning; no

# Initialize
rewards = torch.tensor(gt.conduct_experiment(x=torch.tensor([agent0.actions, agent1.actions]), noise_std=0), dtype=torch.float32)  # also; initialize the list
GP1 = GPRegressionModel(train_x=torch.tensor([agent0.actions, agent1.actions]).unsqueeze(0), train_y=rewards, noise_std=noise_std)
agent0.GP1 = copy.deepcopy(GP1)
agent1.GP1 = copy.deepcopy(GP1)
agent0.GP2 = GPRegressionModel(torch.tensor([agent0.actions, rewards]).unsqueeze(0), agent1.actions, noise_std)
agent1.GP2 = GPRegressionModel(torch.tensor([agent1.actions, rewards]).unsqueeze(0), agent0.actions, noise_std)


for i in tqdm(range(1, num_iterations)):  # We do not start with 0; initialization is before. Does not make a big difference
    for agent in list_agents:
        # agent.compute_mean_var_confidence_intervals_GP1(domain=1)  # domain is questionable here; also not that important...
        agent.compute_mean_var_confidence_intervals_GP2()
        action_vector = agent.GPUCB(agent.GP2)
        agent.actions = torch.cat((agent.actions, action_vector[agent.number].unsqueeze(0)), dim=0)
        # We need both [in programming] because we need the reward
        # Predict whole action space GP2
        # Apply my action and receive reward
    # Conduct experiment
    reward = torch.tensor(gt.conduct_experiment(x=torch.tensor([agent0.actions[-1], agent1.actions[-1]]), noise_std=0), dtype=torch.float32)
    rewards = torch.cat((rewards, reward), dim=0)
    list_agents = list_agents[::1] if i%2 == list_agents[0].number else list_agents[::-1]  # permutate the list of agents such that the first entry in the list always communicates.
    for agent in list_agents:
        if i % 2 == agent.number:  # This is always the case
            communicated_action = agent.actions[-1]  # communicate last action
            # Determine action vector with now additional info of reward
            # First, determine the GP: Build it with training data and predict using this one new point!; already built usin this one training data point I feel
            agent.compute_mean_var_confidence_intervals_GP1(domain=torch.tensor([agent.actions[-1], reward]).unsqueeze(0))
            others_action = agent.GP1.mean  # just the mean prediction is our prediction of what the other person is doing
            others_actions = torch.cat((agent.GP1.train_inputs[0][0][1].view(1), others_action), dim=0) if len(agent.GP1.train_inputs[0][0].shape) == 1 else \
                                        torch.cat((agent.GP1.train_inputs[0][0][:, 1].view(-1,1), others_action), dim=0)
            # Update GP2 here and then remove last action because we do not want to use that for GP1??
        else:
            others_action = communicated_action.unsqueeze(0)
            # Update GP1
            # We now re-define the GP, which is fine. Adding point might be computationally cheaper and better, but not that important right now.
            others_actions = torch.cat((agent.GP1.train_inputs[0][0][1].view(1), others_action), dim=0) if len(agent.GP1.train_inputs[0][0].shape) == 1 else \
                                        torch.cat((agent.GP1.train_inputs[0][0][:, 1].view(-1, 1), others_action), dim=0) 
            train_x = torch.cat((agent.actions.view(-1, 1), others_actions.view(-1, 1)), dim=1)  # one row has the actions from iteration 1 from both agents
            agent.GP1 = GPRegressionModel(train_x=train_x.unsqueeze(0), train_y=rewards, noise_std=noise_std)
        # Update GP2
        agent.GP2 = GPRegressionModel(torch.stack([agent.actions, rewards], dim=1), others_actions, noise_std)  # correct dimension for the train_x?
print('Hello')
'''
The problem is also we need to optimize within each agent, then send the current idea to the global reward system and then receive an input
We need another GP to learn about the other's action. Or some other function that basically predicts what the other agent is doing. This can for 
now also be a black-box random function!!
'''
