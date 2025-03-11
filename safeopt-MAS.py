
import sys
import os
import torch
import warnings
import numpy as np
import gpytorch
from tqdm import tqdm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# Add the relative path to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Print to verify
print("Changed working directory to:", os.getcwd())


from safebo_MAS_plot import plot_2D_mean, plot_2D_UCB, plot_reward, plot_3D_sampled_space, plot_1D_sampled_space
from pacsbo.pacsbo_main import compute_X_plot, ground_truth, initial_safe_samples, PACSBO, GPRegressionModel



# sys.path.append(os.path.abspath("./pacsbo"))


def acquisition_function(noise_std, delta_confidence, exploration_threshold, B, X_plot, X_sample, Y_sample, t,
                         lengthscale_agent_spatio, a_parameter, lengthscale_temporal_RBF, lengthscale_temporal_Ma12,
                           output_variance_RBF, output_variance_Ma12, safety_threshold):
    def compute_sets(cube):
        cube.compute_safe_set()
        cube.maximizer_routine(best_lower_bound_others=-np.infty)
        cube.expander_routine()
    def update_model(cube):
        cube.compute_model(gpr=GPRegressionModel)  # compute confidence intervals?
        cube.compute_mean_var()
        cube.compute_confidence_intervals_evaluation(RKHS_norm_guessed=B)


    cube = PACSBO(delta_confidence=delta_confidence, noise_std=noise_std, tuple_ik=(-1, -1), X_plot=X_plot, X_sample=X_sample,
                    Y_sample=Y_sample, iteration=t, safety_threshold=safety_threshold, exploration_threshold=exploration_threshold, 
                    gt=gt, compute_all_sets=False, lengthscale_spatio=lengthscale_agent_spatio, lengthscale_agent_spatio=lengthscale_agent_spatio,
                    a_parameter=a_parameter, lengthscale_temporal_RBF=lengthscale_temporal_RBF, lengthscale_temporal_Ma12=lengthscale_temporal_Ma12,
                    output_variance_RBF=output_variance_RBF, output_variance_Ma12=output_variance_Ma12)  # all samples that we currently have
    # Building a new object in every iteration
    update_model(cube)
    compute_sets(cube)

    if cube.safety_threshold > -np.infty:
        # warnings.warn('Implement random maximizer of uncertainty!')
        if sum(torch.logical_or(cube.M, cube.G)) != 0:
                max_indices = torch.nonzero(cube.ucb[torch.logical_or(cube.M, cube.G)] == torch.max(cube.ucb[torch.logical_or(cube.M, cube.G)]), as_tuple=True)[0]
                random_max_index = max_indices[torch.randint(len(max_indices), (1,))].item()
                x_new = cube.discr_domain[random_max_index, :]
        else:
            warnings.warn('No new input found. Returning last point of X_sample')
            x_new = torch.cat((X_sample[-1], cube.iteration.flatten() + 1))  # .unsqueeze(0)
    else:
        if sum(cube.M) != 0:
            # max_indices = torch.nonzero(cube.var[cube.M] == torch.max(cube.var[cube.M]), as_tuple=True)[0]
            max_indices = torch.nonzero(cube.ucb == torch.max(cube.ucb), as_tuple=True)[0]
            random_max_index = max_indices[torch.randint(len(max_indices), (1,))].item()
            x_new = cube.discr_domain[random_max_index, :]
        else:
            warnings.warn('No new input found. Returning last point of X_sample')
            x_new = torch.cat((X_sample[-1], cube.iteration.flatten() + 1))

    return x_new, cube


if __name__ == '__main__':
    # Generate ground truth
    iterations = 50
    num_agents = 4
    random_expert = False
    sequential_expert = True
    agents = {}
    noise_std = 1e-1  # increase a little for numerical stability
    RKHS_norm = 1
    delta_confidence = 0.9
    exploration_threshold = 0
    dimension = num_agents

    '''
    Hyperparameters
    '''
    lengthscale_agent_spatio = 0.5  # 0.1
    a_parameter = 200  # weighting factor
    lengthscale_temporal_RBF = 5
    lengthscale_temporal_Ma12 = 0.2
    output_variance_RBF = 1
    output_variance_Ma12 = num_agents

    lengthscale_gt = num_agents/10
    gt = ground_truth(num_center_points=1000, dimension=dimension, RKHS_norm=RKHS_norm, lengthscale=lengthscale_gt)    
    safety_threshold = -np.infty  # torch.quantile(gt.fX, 0.001).item()
    print(f'The heuristic maximum of the function is {max(gt.fX)} and located at {gt.X_center[torch.argmax(gt.fX)]}.')
    print(f'The safety threshold is {safety_threshold}.')

    '''
    Radial temporal kernel
    plt.figure()
    plt.plot(X,Y)
    plt.xlabel('$\|t-t^\prime\|_2$')
    plt.ylabel('$k(t,t^\prime)=k(\|t-t^\prime\|_2$')
    plt.title('Radial temporal kernel')
    plt.savefig('radial_temporal_kernel.png')
    '''

    '''
    We have nearest neighbor communication, the easiest way possible.
    In a 3-agent setting, the undirected graph looks like: 1 - 2 - 3
    We are not in the setting where we want to see the time-series with A0 actions being constant.
    '''
    # Finding initial safe sample
    while True:
        X_sample_full = torch.rand(dimension).unsqueeze(0)  # just start here
        Y_sample = torch.tensor(gt.f(X_sample_full), dtype=torch.float32)
        if Y_sample > safety_threshold:
            break

    for j in range(num_agents):  # set-up agents
        n_dimensions = 2 if j==0 or j==num_agents-1 else 3
        # X_plot needs to be determined for every agent given their position in graph
        X_plot = compute_X_plot(n_dimensions=n_dimensions, points_per_axis=int(1e4**(1/n_dimensions)))  # even less points?
        # which indices are relevant for this agent? for agent 0 it is 0 and 1, for agent 1 it is 0,1,2 etc; see graph
        communication_indices_list = [j, j + 1] if j == 0 else [j - 1, j, j + 1] if 0 < j < num_agents - 1 else [num_agents - 2, num_agents - 1]
        agents[j] = [X_plot, communication_indices_list, None, None, None]
        # We can also just put our action always on index 1 but it makes most sense as is. Index 0 for j=0, index 1 for the rest.

    for t in tqdm(range(1, iterations)):
        for j in range(num_agents):  # this is parallelizable
            X_plot = agents[j][0]
            communication_indices_list = agents[j][1]
            X_sample = X_sample_full[:, communication_indices_list]
            x_new_neighbors, cube = acquisition_function(noise_std, delta_confidence, exploration_threshold, RKHS_norm, X_plot, X_sample, Y_sample, t, lengthscale_spatio=lengthscale_agent_spatio,
                                                             lengthscale_agent_spatio=lengthscale_agent_spatio, a_parameter=a_parameter, lengthscale_temporal_RBF=lengthscale_temporal_RBF,
                                                             lengthscale_temporal_Ma12=lengthscale_temporal_Ma12, output_variance_RBF=output_variance_RBF, output_variance_Ma12=output_variance_Ma12,
                                                             safety_threshold=safety_threshold)

            x_new = x_new_neighbors[1].unsqueeze(0) if j != 0 else x_new_neighbors[0].unsqueeze(0)  # in this easy tree structure. We can use anytrees later
            agents[j] = [X_plot, communication_indices_list, x_new, x_new_neighbors, cube]  # this contains the multi-dimensional x_new_neighbors and the single one x_new

        if t != iterations - 1:  # last iteration; do not add the new point; we just want the updated model
            if sequential_expert or random_expert:
                if sequential_expert:
                    expert_agent = (t - 1) % num_agents
                elif random_expert:
                    expert_agent = np.random.choice(range(num_agents))  # who is the expert this round?
                expert_x_new_neighbors = agents[expert_agent][3][:-1]  # exclude time. Wait time is never in?
                expert_communication_list = agents[expert_agent][1]
                x_new_full = torch.zeros(num_agents).unsqueeze(0)
                x_new_full[:, expert_communication_list] = expert_x_new_neighbors
                for jj in range(num_agents):
                    if jj not in expert_communication_list:
                        x_new_full[:, jj] = agents[jj][2]  # get their x_new prediction
            else:
                x_new_full = torch.cat([agents[j][2] for j in range(num_agents)]).unsqueeze(0)  # concatenate all x_new ("1D ones")
            if torch.any(torch.all(X_sample_full == x_new_full, dim=1)):
                message = f'Same same same; {x_new_full}'
                warnings.warn(message)
            y_new = torch.tensor(gt.f(x_new_full), dtype=torch.float32)  # this is the applied action!
            Y_sample = torch.cat((Y_sample, y_new), dim=0)
            X_sample_full = torch.cat((X_sample_full, x_new_full))  # , dim=0)  # cat all samples
    print('Hello')

    # Development of reward
    plot_reward(X_sample, Y_sample, safety_threshold, gt)

    warnings.warn("Beware the dimensions here!")
    K_T = cube.model.kernel_temporal(cube.iteration_x.float(), cube.iteration_x.float()).to_dense()
    K_S = cube.model.kernel_spatio(X_sample_full, X_sample_full).to_dense()
    X_iteration_full = torch.cat((X_sample_full, cube.iteration_x), dim=1)
    K_total = cube.model.kernel(X_iteration_full, X_iteration_full).to_dense()
    # Agents 0 and 3 Mean and UCB
    plot_2D_mean(cube=agents[0][-1], agent_number=0)
    plot_2D_mean(cube=agents[3][-1], agent_number=3)
    plot_2D_UCB(cube=agents[0][-1], agent_number=0)
    plot_2D_UCB(cube=agents[3][-1], agent_number=3)

    # Agents 1 and 2 3D explored domain
    plot_3D_sampled_space(cube=agents[1][-1], agent_number=1)
    plot_3D_sampled_space(cube=agents[2][-1], agent_number=2)

    # All agents 1D explored domain
    plot_1D_sampled_space(cube=agents[0][-1], agent_number=0)
    plot_1D_sampled_space(cube=agents[1][-1], agent_number=1)
    plot_1D_sampled_space(cube=agents[2][-1], agent_number=2)
    plot_1D_sampled_space(cube=agents[3][-1], agent_number=3)

    # How much did we explore? Convex hull volume
    hull = ConvexHull(X_sample_full.numpy())
    print(f'We explored about {hull.volume*100}% of the domain.')

    list_Y = []
    kernel = gpytorch.kernels.MaternKernel(nu=1.5)
    kernel.lengthscale = 10
    # kernel=cube.model.kernel_temporal
    for _ in range(20):
        # Y_c = generating_kernel_paths(, RKHS_norm=1)
        Y_c = generating_kernel_paths(kernel=kernel, RKHS_norm=1)
        list_Y.append(Y_c)
    plt.figure()
    for Y_c in list_Y:
        plt.plot(range(1, 50), Y_c.detach().numpy(), color='magenta', alpha=0.2)
    plt.xlabel('Iterations $t$')
    plt.ylabel('$y$')
    plt.title('Sample paths of RKHS of RadialTemporalKernel')
    # plt.savefig('Matern12kernel_ell_10.png')


    # Plot the kernel
    # Plot the 1D (radial) and 2D for all kernels
    kernel = gpytorch.kernels.MaternKernel(nu=1.5)
    kernel.lengthscale = 1
    X_c = torch.arange(1, 50).float()
    K = kernel(X_c, X_c).to_dense()
    plt.plot(X_c, K[0, :].detach().numpy())
    plt.xlabel('$\|t-t^\prime\|_2$')
    plt.ylabel('$k(\|t-t^\prime\|_2$')
    plt.title('Matern12 kernel ($\ell=1$)')

    kernel=cube.model.kernel_temporal
    K = kernel(X_iteration_full).to_dense()  # careful here; do we really need all of this? It depends. If we call the kernel from the model, we might need this because of the active dims
    # We just define the class here new and then we have the kernel
    kernel = RadialTemporalKernel()
    K = kernel(X_c, X_c).to_dense()
    plt.figure()
    plt.plot(X_c, K[0, :].detach().numpy())
    plt.xlabel('$\|t-t^\prime\|_2$')
    plt.ylabel('$k(\|t-t^\prime\|_2$')
    plt.title('RadialTemporalKernel')


    # 2D plots
    X_c = torch.arange(1, 50).float()
    X_c1, X_c2 = torch.meshgrid(X_c, X_c)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(X_c1, X_c2, K, cmap='viridis', edgecolor='none')
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Z value')
    ax.set_xlabel('$t_1$')
    ax.set_ylabel('$t_2$')
    ax.set_zlabel('$k(t_1,t_2)$')
    ax.set_title('RadialTemporalKernel')
    plt.savefig('Radialtemporalkernel_2D.png')




'''
    for j in range(1, num_agents):  # not the first because we keep this constant
        cube = agents[j][-1]
        plt.figure()
        plt.plot(np.asarray(X_sample_full[:, j]), np.asarray(Y_sample), 'ob', markersize=10)
        # plt.plot(X_plot, gt.f(X_plot), '-b')
        plt.fill_between(X_plot.flatten(), cube.lcb.detach().numpy(), cube.ucb.detach().numpy(), color='gray', alpha=0.25)
        if safety_threshold != -np.infty:
            plt.plot(X_plot.flatten(), torch.ones_like(X_plot.flatten())*safety_threshold, '-r')
        plt.xlabel('Action')
        plt.ylabel('Reward')
        plt.title(f'Agent {j}')
        # plt.savefig(f'../{num_agents}_agents_agent_{j}_safety.png')

    # Now plot Agent 0 (the one with constant)
    plt.figure()
    plt.plot(range(iterations), Y_sample, '*-')
    plt.title(f'Agent 0: Time series POV; {num_agents} total agents, safety threshold={round(safety_threshold,2)}')
    plt.xlabel('Iteration')
    plt.ylabel('Global reward')
    # plt.savefig(f'../{num_agents}_agents_agent_0_safety.png')
'''
