import matplotlib.pyplot as plt
import torch
import numpy as np
from pacsbo.pacsbo_main import compute_X_plot


def plot_2D_mean(cube_dict, agent_number, save=False):
    t = cube_dict['iteration'].item()
    n_dimensions = 2 if agent_number==0 or agent_number==7 else 3  # only for 4 agents right now; fine
    discr_domain = compute_X_plot(n_dimensions=n_dimensions, points_per_axis=int(1e4**(1/n_dimensions)))
    plt.figure()
    m = cube_dict['mean'].detach().numpy()
    sc = plt.scatter(
        discr_domain[:, 0],
        discr_domain[:, 1],
        c=m,
        cmap='plasma'
    )
    plt.colorbar(sc, label="Mean Value")  # Add a colorbar to show the mapping
    plt.title(f'Iteration {t}; Agent {agent_number} mean')
    plt.scatter(cube_dict['x_sample'][:, 0], cube_dict['x_sample'][:, 1], label="Sampled Points", color='k')
    plt.xlabel('$a_1$')
    plt.ylabel('$a_2$')
    if not save:
        plt.show()
    else:
        plt.savefig(f'mean_value_agent_{agent_number}.png')


def plot_2D_UCB(cube_dict, agent_number, save=False):
    t = cube_dict['iteration'].item()
    n_dimensions = 2 if agent_number==0 or agent_number==3 else 3  # only for 4 agents right now; fine
    discr_domain = compute_X_plot(n_dimensions=n_dimensions, points_per_axis=int(1e4**(1/n_dimensions)))
    plt.figure()
    u = cube_dict['mean'].detach().numpy() + cube_dict['var'].detach().numpy()
    sc = plt.scatter(
        discr_domain[:, 0],
        discr_domain[:, 1],
        c=u,
        cmap='plasma'
    )
    plt.colorbar(sc, label="UCB Value")  # Add a colorbar to show the mapping
    plt.title(f'Iteration {t}; Agent {agent_number} UCB')
    plt.scatter(cube_dict['x_sample'][:, 0], cube_dict['x_sample'][:, 1], label="Sampled Points", color='k')
    plt.xlabel('$a_1$')
    plt.ylabel('$a_2$')
    if not save:
        plt.show()
    else:
        plt.savefig(f'ucb_value_agent_{agent_number}.png')


def plot_reward(cube, save=False):
    plt.figure()
    X_sample = cube['x_sample']
    Y_sample = cube['y_sample']
    safety_threshold = cube['safety_threshold']
    plt.plot(range(len(X_sample)), Y_sample.detach().numpy(), '-*', label='Samples')
    # plt.plot(range(len(X_sample)), torch.ones(len(X_sample))*max(gt.fX), '--g', label='Max')
    if safety_threshold > -np.infty:
        plt.plot(range(len(X_sample)), torch.ones(len(X_sample))*safety_threshold, '--r', label='Safety threshold')
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    # plt.legend()
    if not save:
        plt.show()
    else:
        plt.savefig('reward_development.png')


def plot_3D_sampled_space(cube_dict, agent_number, save=False):
    X_sample = cube_dict['x_sample'][:, :-1]
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 0.9, 0.9], projection='3d')
    ax.scatter(X_sample[:, 0], X_sample[:, 1], X_sample[:, 2], color='black', s=100)
    # Labels
    ax.set_xlabel('$a_1$')
    ax.set_ylabel('$a_2$')
    ax.set_zlabel('$a_3$')
    ax.set_title(f'Sampled space Agent {agent_number}')
    if not save:
        plt.show()
    else:
        plt.savefig(f'3D_sampled_space_agent_{agent_number}.png')


def plot_1D_sampled_space(cube_dict, agent_number, save=False):
    index = 1 if agent_number != 0 else 0
    X_sample = cube_dict['x_sample'][:, index]
    plt.figure()
    plt.plot()
    plt.scatter(X_sample, cube_dict['y_sample'])
    plt.xlabel(f'Parameter $a$ of Agent {agent_number}')
    plt.ylabel('Corresponding $y$ value')
    plt.title(f'1D explored domain Agent {agent_number}')
    plt.xlim([-0.1, 1.1])
    if not save:
        plt.show()
    else:
        plt.savefig(f'1D_sampled_space_agent_{agent_number}.png')