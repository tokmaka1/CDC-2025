
import sys
import os
import torch
import warnings
# warnings.filterwarnings("ignore")

import numpy as np
import gpytorch
from tqdm import tqdm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
import dill

random_seed_number = 107

np.random.seed(random_seed_number)

# Fix seed for PyTorch (CPU)
torch.manual_seed(random_seed_number)


# Add the relative path to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Print to verify
print("Changed working directory to:", os.getcwd())


from safebo_MAS_plot import plot_2D_mean, plot_reward, plot_3D_sampled_space, plot_1D_sampled_space, plot_2D_UCB
from pacsbo.pacsbo_main import compute_X_plot, ground_truth, initial_safe_samples, PACSBO, GPRegressionModel



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
                    Y_sample=Y_sample, iteration=t, safety_threshold=safety_threshold, exploration_threshold=exploration_threshold, compute_all_sets=False, lengthscale_agent_spatio=lengthscale_agent_spatio,
                    a_parameter=a_parameter, lengthscale_temporal_RBF=lengthscale_temporal_RBF, lengthscale_temporal_Ma12=lengthscale_temporal_Ma12,
                    output_variance_RBF=output_variance_RBF, output_variance_Ma12=output_variance_Ma12)  # all samples that we currently have
    # Building a new object in every iteration
    update_model(cube)
    compute_sets(cube)

    if cube.safety_threshold > -np.infty:
        if sum(torch.logical_or(cube.M, cube.G)) != 0:
                max_indices = torch.nonzero(cube.ucb[torch.logical_or(cube.M, cube.G)] == torch.max(cube.ucb[torch.logical_or(cube.M, cube.G)]), as_tuple=True)[0]
                random_max_index = max_indices[torch.randint(len(max_indices), (1,))].item()
                x_new = cube.discr_domain[torch.logical_or(cube.M, cube.G)][random_max_index, :]
        else:
            # warnings.warn('No new input found. Returning last point of X_sample')
            x_new = torch.cat((X_sample[-1], cube.iteration.flatten() + 1))  # .unsqueeze(0)
    else:
        if sum(cube.M) != 0:  # Why no G? Because there is no non-safe set. SafeOpt without safe set is GP-UCB
            # max_indices = torch.nonzero(cube.var[cube.M] == torch.max(cube.var[cube.M]), as_tuple=True)[0]
            max_indices = torch.nonzero(cube.ucb == torch.max(cube.ucb), as_tuple=True)[0]
            random_max_index = max_indices[torch.randint(len(max_indices), (1,))].item()
            x_new = cube.discr_domain[cube.M][random_max_index, :]
        else:
            # warnings.warn('No new input found. Returning last point of X_sample')
            x_new = torch.cat((X_sample[-1], cube.iteration.flatten() + 1))

    return x_new, cube


def process_agent(j, X_plot, communication_indices_list, X_sample_full, Y_sample, t, hyperparameters):
    noise_std, delta_confidence, exploration_threshold, RKHS_norm, lengthscale_agent_spatio, a_parameter, lengthscale_temporal_RBF, lengthscale_temporal_Ma12, output_variance_RBF, output_variance_Ma12, safety_threshold = hyperparameters
    X_sample = X_sample_full[:, communication_indices_list]

    x_new_neighbors, cube = acquisition_function(
        noise_std, delta_confidence, exploration_threshold, RKHS_norm,
        X_plot, X_sample, Y_sample, t, lengthscale_agent_spatio,
        a_parameter, lengthscale_temporal_RBF, lengthscale_temporal_Ma12,
        output_variance_RBF, output_variance_Ma12, safety_threshold
    )
    cube_dict = {}
    cube_dict['iteration'] = cube.iteration
    cube_dict['mean'] = cube.mean
    # cube_dict['discr_domain'] = cube.discr_domain
    cube_dict['x_sample'] = cube.x_sample
    cube_dict['var'] = cube.var  # maybe also not necessary, let's see
    cube_dict['y_sample'] = cube.y_sample
    cube_dict['safety_threshold'] = safety_threshold
    cube_dict['beta'] = cube.beta
    if communication:
        if full_communication:
            x_new = x_new_neighbors
        else:
            x_new = x_new_neighbors[1].unsqueeze(0) if j != 0 else x_new_neighbors[0].unsqueeze(0)  # I think this is fine for any communication

    else: 
        x_new = x_new_neighbors[0].unsqueeze(0)
    agents_j = [
        x_new.detach(),  # Detach tensor
        x_new_neighbors.detach(),  # Detach tensor
        cube_dict
    ]
    return j, agents_j


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)  # Ensure safe multiprocessing with PyTorch
    # Generate ground truth
    iterations = 50
    num_agents = 8
    random_expert = False
    sequential_expert = False
    agents = {}
    communication_list_dict = {}
    X_plot_dict = {}
    noise_std = 1e-2  # increase a little for numerical stability
    delta_confidence = 0.9
    exploration_threshold = 0.1
    dimension = num_agents
    initial_point_quantile = 0.5
    safety_quantile = 0.2
    communication = True
    full_communication = True
    time_latent_variable = False

    '''
    Hyperparameters
    '''
    RKHS_norm_spatio_temporal = 1  # 0.1
    lengthscale_agent_spatio = 0.3  # 0.5
    a_parameter = 50  # weighting factor for the brownian motion and reverse brownian motion kernel
    lengthscale_temporal_RBF = 20  # 20  # 5
    lengthscale_temporal_Ma12 = 5  # 5  # 1
    # Changing length scales did not directly influence stuff
    output_variance_RBF = 0.1  # 0.1 0.5  # 0.5
    output_variance_Ma12 = 0.1  # 0.1 2  # num_agents
    # Changing the output variance does influence stuff a lot

    lengthscale_gt = num_agents/10
    gt = ground_truth(num_center_points=1000, dimension=dimension, RKHS_norm=1, lengthscale=lengthscale_gt)    
    safety_threshold = torch.quantile(gt.fX, safety_quantile).item()  # -np.infty  # 
    print(f'The heuristic maximum of the function is {max(gt.fX)} and located at {gt.X_center[torch.argmax(gt.fX)]}.')
    print(f'The safety threshold is {safety_threshold}.')

    hyperparameters = [noise_std, delta_confidence, exploration_threshold, RKHS_norm_spatio_temporal, lengthscale_agent_spatio,
                    a_parameter, lengthscale_temporal_RBF, lengthscale_temporal_Ma12, output_variance_RBF, 
                    output_variance_Ma12, safety_threshold]
    agents['gt'] = gt
    # Finding initial safe sample
    while True:
        X_sample_full = torch.rand(dimension).unsqueeze(0)  # just start here
        # X_sample_full = gt.X_center[torch.argmax(gt.fX)].unsqueeze(0)  # start with highest point
        Y_sample = torch.tensor(gt.f(X_sample_full), dtype=torch.float32)
        if Y_sample > torch.quantile(gt.fX, initial_point_quantile).item():  #  safety_threshold:
            break

    for j in range(num_agents):  # set-up agents
        if communication:
            if full_communication:
                n_dimensions = num_agents
                communication_indices_list = [kk for kk in range(num_agents)]
            else:
                n_dimensions = 2 if j==0 or j==num_agents-1 else 3
                communication_indices_list = [j, j + 1] if j == 0 else [j - 1, j, j + 1] if 0 < j < num_agents - 1 else [num_agents - 2, num_agents - 1]
        else:
            n_dimensions = 1
            communication_indices_list = [j]
        communication_list_dict[j] = communication_indices_list
        # X_plot needs to be determined for every agent given their position in graph
        X_plot = compute_X_plot(n_dimensions=n_dimensions, points_per_axis=int(1e4**(1/n_dimensions)))  # more points?
        # which indices are relevant for this agent? for agent 0 it is 0 and 1, for agent 1 it is 0,1,2 etc; see graph
        X_plot_dict[j] = X_plot

    for t in tqdm(range(1, iterations)):
        for j in range(num_agents):  # this is parallelizable
            if time_latent_variable:
                j, agents_j = process_agent(j, X_plot_dict[j], communication_list_dict[j], X_sample_full, Y_sample, t, hyperparameters)
            else:
                j, agents_j = process_agent(j, X_plot_dict[j], communication_list_dict[j], X_sample_full, Y_sample, 0, hyperparameters)
            agents[j] = agents_j  # global dict.
        # with Pool(processes=num_agents) as pool:
        #     results = pool.starmap(process_agent, [(j, agents[j], X_sample_full, Y_sample, t, hyperparameters) for j in range(num_agents)])
        # for j, agents_j in results:
        #     agents[j] = agents_j


        if t != iterations - 1:  # last iteration; do not add the new point; we just want the updated model
            if sequential_expert or random_expert:
                if sequential_expert:
                    expert_agent = (t - 1) % num_agents
                elif random_expert:
                    expert_agent = np.random.choice(range(num_agents))  # who is the expert this round?
                expert_x_new_neighbors = agents[expert_agent][1][:-1]  # exclude time. Wait time is never in?
                expert_communication_list = communication_list_dict[expert_agent]
                x_new_full = torch.zeros(num_agents).unsqueeze(0)
                x_new_full[:, expert_communication_list] = expert_x_new_neighbors
                for jj in range(num_agents):
                    if jj not in expert_communication_list:
                        x_new_full[:, jj] = agents[jj][0]  # get their x_new prediction
            else:
                if full_communication:
                    x_new_full = agents[0][0][:-1].unsqueeze(0)  # we can take any agent; they are all the same because we model all and communicate everything
                    # -1 because we leave time domain out!
                else:
                    x_new_full = torch.cat([agents[j][0] for j in range(num_agents)]).unsqueeze(0)  # concatenate all x_new ("1D ones")
            print(f'We are sampling {x_new_full}')
            y_new = torch.tensor(gt.f(x_new_full), dtype=torch.float32)  # this is the applied action!
            Y_sample = torch.cat((Y_sample, y_new), dim=0)
            X_sample_full = torch.cat((X_sample_full, x_new_full))  # , dim=0)  # cat all samples
    print('Hello')
    # Just save the cube!  # also save beta
    with open('agents_8_50_107_full_communication.pickle', 'wb') as handle:
        dill.dump(agents, handle)


    # Development of reward
    plot_reward(cube=agents[0][-1])

    plot_2D_UCB(cube_dict=agents[0][-1], agent_number=0)
    plot_2D_UCB(cube_dict=agents[num_agents-1][-1], agent_number=num_agents-1)

    plot_2D_mean(cube_dict=agents[0][-1], agent_number=0)
    plot_2D_mean(cube_dict=agents[num_agents-1][-1], agent_number=num_agents-1)



    # Agents 1 and 2 3D explored domain
    for j in range(1, num_agents-1):  # not the first, not the last
        plot_3D_sampled_space(cube_dict=agents[j][-1], agent_number=j)

    # All agents 1D explored domain
    for j in range(1, num_agents-1):
        plot_1D_sampled_space(cube_dict=agents[j][-1], agent_number=j)

    # How much did we explore? Convex hull volume
    # hull = ConvexHull(X_sample_full.numpy())
    # print(f'We explored about {hull.volume*100}% of the domain.')






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
