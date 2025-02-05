
import sys
import os
import torch
import warnings
import numpy as np
from tqdm import tqdm

# Add the relative path to the system path
sys.path.append(os.path.abspath("./pacsbo"))
from pacsbo.pacsbo_main import compute_X_plot, ground_truth, initial_safe_samples, PACSBO, GPRegressionModel
import matplotlib.pyplot as plt


def acquisition_function(noise_std, delta_confidence, exploration_threshold, B, X_plot, X_sample, Y_sample, lengthscale, safety_threshold):
    def compute_sets(cube):
        cube.compute_safe_set()
        cube.maximizer_routine(best_lower_bound_others=-np.infty)
        cube.expander_routine()
    def update_model(cube):
        cube.compute_model(gpr=GPRegressionModel)  # compute confidence intervals?
        cube.compute_mean_var()
        cube.compute_confidence_intervals_evaluation(RKHS_norm_guessed=B)


    cube = PACSBO(delta_confidence=delta_confidence, delta_cube=1, noise_std=noise_std, tuple_ik=(-1, -1), X_plot=X_plot, X_sample=X_sample,
                    Y_sample=Y_sample, safety_threshold=safety_threshold, exploration_threshold=exploration_threshold, gt=gt,
                    compute_local_X_plot=False, compute_all_sets=False, lengthscale=lengthscale)  # all samples that we currently have
    update_model(cube)
    compute_sets(cube)

    if sum(torch.logical_or(cube.M, cube.G)) != 0:
        x_new = cube.discr_domain[torch.logical_or(cube.M, cube.G)][torch.argmax(cube.var[torch.logical_or(cube.M, cube.G)])]
    else:
        warnings.warn('No new input found. Returning last point of X_sample')
        x_new = X_sample[-1].unsqueeze(0)
    return x_new, cube


if __name__ == '__main__':
    # Generate ground truth
    iterations = 75
    num_agents = 3
    agents = {}
    lengthscale_agent = 0.1
    lengthscale_gt = num_agents/10
    noise_std = 1e-1  # increase a little for numerical stability
    RKHS_norm = 1
    delta_confidence = 0.9
    exploration_threshold = 0
    dimension = num_agents
    communication = True
    gt = ground_truth(num_center_points=750, dimension=dimension, RKHS_norm=RKHS_norm, lengthscale=lengthscale_gt)
    safety_threshold = torch.quantile(gt.fX, 0.1).item()  # -np.infty  # based on X_center
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
        n_dimensions = 2 if j==0 or num_agents-1 else 3
        # X_plot needs to be determined for every agent given their position in graph
        X_plot = compute_X_plot(n_dimensions=n_dimensions, points_per_axis=int(1000/n_dimensions))
        agents[j] = [X_plot, None, None]
        # which indices are relevant for this agent? for agent 0 it is 0 and 1, for agent 1 it is 0,1,2 etc; see graph
        communication_indices = [j, j + 1] if j == 0 else [j - 1, j, j + 1] if 0 < j < num_agents - 1 else [num_agents - 2, num_agents - 1]
        # We can also just put our action always on index 1 but it makes most sense as is. Index 0 for j=0, index 1 for the rest.

    for i in tqdm(range(iterations)):
        for j in range(num_agents):  # this is parallelizable
            X_plot = agents[j][0]
            X_sample = X_sample_full[:, j+1]
            x_new, cube = acquisition_function(noise_std, delta_confidence, exploration_threshold, RKHS_norm, X_plot, X_sample, Y_sample, lengthscale=lengthscale_agent, safety_threshold=safety_threshold)
            agents[j] = [X_plot, x_new, cube]
        if i != iterations - 1:  # last iteration; do not add the new point; we just want the updated model
            x_new_full = torch.cat([agents[j][0] for j in range(num_agents)]).unsqueeze(0)  # concatenate all x_new
            y_new = torch.tensor(gt.f(x_new_full), dtype=torch.float32)
            Y_sample = torch.cat((Y_sample, y_new), dim=0)
            X_sample_full = torch.cat((X_sample_full, x_new_full))  # , dim=0)  # cat all samples
    print('Hello')

    for j in range(1, num_agents):  # not the first because we keep this constant
        cube = agents[j][1]
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
