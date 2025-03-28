
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
from vehicle_class import simulate, reward_function, vehicle, leading_vehicle

random_seed_number = 42

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
    num_agents = 4
    random_expert = False
    sequential_expert = True
    agents = {}
    communication_list_dict = {}
    X_plot_dict = {}
    noise_std = 1e-2  # increase a little for numerical stability
    delta_confidence = 0.9
    exploration_threshold = 0.1
    dimension = num_agents
    communication = True
    full_communication = False
    time_latent_variable = True

    '''
    Hyperparameters
    '''
    RKHS_norm_spatio_temporal = 5 # 0.1
    lengthscale_agent_spatio = 0.2  # 0.3
    a_parameter = 50  # weighting factor for the brownian motion and reverse brownian motion kernel
    lengthscale_temporal_RBF = 20  # 20  # 5
    lengthscale_temporal_Ma12 = 5  # 5  # 1
    # Changing length scales did not directly influence stuff
    output_variance_RBF = 1  # 0.5  # 0.5  # 0.1 0.5  # 0.5
    output_variance_Ma12 = 1  # 0.5  # 0.5  # 0.1 2  # num_agents



    # Set up the simulation
    s_init_list = [0, 300, 520, 700, 1000]
    num_vehicles = 5
    v_leader = 30
    T_simulation = 120 # Total simulation time in seconds
    dt_simulation = 0.1
    steps = int(T_simulation / dt_simulation) 
    # Determine goal distance
    d_ref = 100 # we want 100m between the LKWs
    time = np.linspace(0, T_simulation, steps)
    K_p_values = [0.4, 0.5, 0.4, 0.5]  # [0.5]*(num_vehicles-1)  # this is between 0 and 10, start with 5 for all
    K_i_values = [0.001]*(num_vehicles-1)
    hyperparameters_simulation = [num_vehicles, v_leader, d_ref, steps, dt_simulation, s_init_list]
    safety_threshold = -1  # -1  # -d_ref*(num_vehicles-1)*T_simulation/10000  # I guess quite non-smooth
    print(f'The safety threshold is {safety_threshold}.')

    hyperparameters = [noise_std, delta_confidence, exploration_threshold, RKHS_norm_spatio_temporal, lengthscale_agent_spatio,
                    a_parameter, lengthscale_temporal_RBF, lengthscale_temporal_Ma12, output_variance_RBF, 
                    output_variance_Ma12, safety_threshold]

    # Build cars
    list_vehicles = []
    for ii in range(num_vehicles):
        r = np.random.uniform(0.4, 0.6)   # 0.5   # Wheel radius (m)
        ga = 9.81      # Gravitational acceleration (m/s²); all on same planet
        alpha = 0  # 0.05   # Road grade (rad) ≈ 2.86°; all on same street
        cr = np.random.uniform(0.004, 0.008)  # 0.006     # Rolling resistance coefficient
        rho = 1.225    # Air density (kg/m³); all in the same air
        Ar = np.random.uniform(5, 7) # 10        # Cross-sectional area (m²)
        Cd = np.random.uniform(0.4, 0.8)  # 0.6       # Aerodynamic drag coefficient
        m = np.random.uniform(1950, 2050)  # 2000      # Mass (kg) ~15 tons
        if ii != num_vehicles - 1:
            list_vehicles.append(vehicle(r, ga, alpha, cr, rho, Ar, Cd, m, dt_simulation, s_init=s_init_list[ii]))
        else:
            list_vehicles.append(leading_vehicle(r, ga, alpha, cr, rho, Ar, Cd, m, dt_simulation, s_init=s_init_list[ii], v=v_leader))  # sinit 250
    positions, velocities, total_error_list, torques, distances_to_front_vehicle, abs_errors  = simulate(hyperparameters_simulation, K_p_values, K_i_values, list_vehicles)  # we will tune K_p values from our algoritm
    reward = reward_function(distances_to_front_vehicle, d_ref, num_vehicles, T_simulation)
    X_sample_full = torch.tensor(K_p_values, dtype=torch.float32).unsqueeze(0)
    Y_sample = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
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
            K_p_values = x_new_full.tolist()[0]
            positions, velocities, total_error_list, torques, distances_to_front_vehicle, abs_errors  = simulate(hyperparameters_simulation, K_p_values, K_i_values, list_vehicles)  # we will tune K_p values from our algoritm
            reward = reward_function(distances_to_front_vehicle, d_ref, num_vehicles, T_simulation)
            y_new = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
            print(f'We sampled {x_new_full} with value {y_new}')
            if y_new <= safety_threshold:
                warnings.warn("We have a safety violation!")
            Y_sample = torch.cat((Y_sample, y_new), dim=0)
            X_sample_full = torch.cat((X_sample_full, x_new_full))  # , dim=0)  # cat all samples
    agents['X_sample_full'] = X_sample_full
    print('Hello')
    # This is just the very last time step... We should have some other plot. To see what we save; but first. reward and safety threhold need to be well-defined
    # agents['positions'] = positions
    # agents['velocities'] = velocities
    # agents['total_error_list'] = total_error_list
    # agents['distances_to_front_vehicle'] = distances_to_front_vehicle
    # agents['abs_errors'] = abs_errors
    with open('vehicles_first_test.pickle', 'wb') as handle:
        dill.dump(agents, handle)
    plot_reward(cube=agents[0][-1])
