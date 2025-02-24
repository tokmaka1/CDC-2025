import matplotlib.pyplot as plt
import torch
import numpy as np


def plot_2D_mean(cube, agent_number, save=False):
    t = cube.iteration.item()
    plt.figure()
    m = cube.mean.detach().numpy()
    sc = plt.scatter(
        cube.discr_domain[:, 0],
        cube.discr_domain[:, 1],
        c=m,
        cmap='plasma'
    )
    plt.colorbar(sc, label="Mean Value")  # Add a colorbar to show the mapping
    plt.title(f'Iteration {t}; Agent {agent_number} mean')
    plt.scatter(cube.x_sample[:, 0], cube.x_sample[:, 1], label="Sampled Points", color='k')
    plt.xlabel('$a_1$')
    plt.ylabel('$a_2$')
    if not save:
        plt.show()
    else:
        plt.savefig(f'mean_value_agent_{agent_number}.png')


def plot_2D_UCB(cube, agent_number, save=False):
    t = cube.iteration.item()
    plt.figure()
    u = cube.mean.detach().numpy() + cube.var.detach().numpy()
    sc = plt.scatter(
        cube.discr_domain[:, 0],
        cube.discr_domain[:, 1],
        c=u,
        cmap='plasma'
    )
    plt.colorbar(sc, label="UCB Value")  # Add a colorbar to show the mapping
    plt.title(f'Iteration {t}; Agent {agent_number} UCB')
    plt.scatter(cube.x_sample[:, 0], cube.x_sample[:, 1], label="Sampled Points", color='k')
    plt.xlabel('$a_1$')
    plt.ylabel('$a_2$')
    if not save:
        plt.show()
    else:
        plt.savefig(f'ucb_value_agent_{agent_number}.png')


def plot_reward(X_sample, Y_sample, safety_threshold, gt, save=False):
    plt.figure()
    plt.plot(range(len(X_sample)), Y_sample.detach().numpy(), '-*', label='Samples')
    plt.plot(range(len(X_sample)), torch.ones(len(X_sample))*max(gt.fX), '--g', label='Max')
    if safety_threshold > -np.infty:
        plt.plot(range(len(X_sample)), torch.ones(len(X_sample))*safety_threshold, '--r', label='Safety threshold')
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    plt.legend()
    if not save:
        plt.show()
    else:
        plt.savefig('reward_development.png')


def plot_3D_sampled_space(cube, agent_number, save=False):
    X_sample = cube.x_sample[:, :-1]
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


def plot_1D_sampled_space(cube, agent_number, save=False):
    index = 1 if agent_number != 0 else 0
    X_sample = cube.x_sample[:, index]
    plt.figure()
    plt.plot()
    plt.scatter(X_sample, cube.y_sample)
    plt.xlabel(f'Parameter $a$ of Agent {agent_number}')
    plt.ylabel('Corresponding $y$ value')
    plt.title(f'1D explored domain Agent {agent_number}')
    if not save:
        plt.show()
    else:
        plt.savefig(f'1D_sampled_space_agent_{agent_number}.png')

