from functools import partial
import hydra
import cvxpy as cp
import numpy as np
from lah.launcher import Workspace
from scipy import sparse
import jax.numpy as jnp
from scipy.sparse import csc_matrix
import time
import matplotlib.pyplot as plt
import os
import scs
import logging
import yaml
from jax import vmap, jit
import pandas as pd
import matplotlib.colors as mc
import colorsys
from lah.algo_steps import get_scaled_vec_and_factor
from lah.examples.solve_script import setup_script
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset



plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 16,
    }
)
log = logging.getLogger(__name__)


def simulate(T, gamma, dt, sigma, p):
    A, B, C = robust_kalman_setup(gamma, dt)

    # generate random input and noise vectors
    w = np.random.randn(2, T)
    v = np.random.randn(2, T)

    x = np.zeros((4, T + 1))
    # x[:, 0] = [0, 0, 0, 0]
    y = np.zeros((2, T))

    # add outliers to v
    # np.random.seed(0)
    inds = np.random.rand(T) <= p
    v[:, inds] = sigma * np.random.randn(2, T)[:, inds]

    # simulate the system forward in time
    for t in range(T):
        y[:, t] = C.dot(x[:, t]) + v[:, t]
        x[:, t + 1] = A.dot(x[:, t]) + B.dot(w[:, t])

    x_true = x.copy()
    w_true = w.copy()
    return y, x_true, w_true, v


def shifted_sol(z_star, T, m, n):
    """
    variables
    (x_t, w_t, s_t, v_t,  u_t, z_t) in (nx + nu + no + 3)
    (nx,  nu,  1,   no,   no, 1, 1)
    min sum_{i=0}^{T-1} ||w_t||_2^2 + mu (u_t+rho*z_t^2)
        s.t. x_{t+1} = Ax_t + Bw_t  t=0,...,T-2 (dyn)
             y_t = Cx_t + v_t       t=0,...,T-1 (obs)
             u_t + z_t = s_t        t=0,...,T-1 (aux)
             z_t <= rho             t=0,...,T-1 (z ineq)
             u_t >= 0               t=0,...,T-1 (u ineq)
             ||v_t||_2 <= s_t       t=0,...,T-1 (socp)
    vars: 
        (x_0, ..., x_{T-1}) (nx)
        (w_0, ..., w_{T-2}) (nu)
        (v_0, ..., v_{T-1}) (no)
        (u_0, ..., u_{T-1}) (1)
        (s_0, ..., s_{T-1}) (1)
        (z_0, ..., z_{T-1}) (1)

    data: (y_0, ..., y_{T-1})
    """
    return z_star
    nx = 4
    nu = 2
    no = 2
    # shifted_z_star = jnp.zeros(z_star.size)

    x_star = z_star[:n]
    y_star = z_star[n:-1]

    shifted_x_star = jnp.zeros(n)
    shifted_y_star = jnp.zeros(m)

    # indices markers
    w_start = nx * T
    s_start = w_start + nu * T
    v_start = s_start + T
    u_start = v_start + no * T
    z_start = u_start + T
    # end_x = nx * T
    # end_w = end_x + nu * (T - 1)
    # end_v = end_w + 2 * (T)
    # assert end_v == n
    end_dyn_cons = nx * T
    end_state_cons = 2 * nx * T
    end = T * (2 * nx + nu)

    # get primal vars
    shifted_x = x_star[nx:w_start]
    shifted_w = x_star[w_start + nu: s_start]
    shifted_s = x_star[s_start + 1: v_start]
    shifted_v = x_star[v_start + no: u_start]
    shifted_u = x_star[u_start + 1: z_start]
    shifted_z = x_star[z_start + 1:]

    # insert into shifted x_star
    shifted_x_star = shifted_x_star.at[:w_start - nx].set(shifted_x)
    shifted_x_star = shifted_x_star.at[end_state:-nu].set(shifted_controls)

    # get dual vars
    shifted_dyn_cons = y_star[nx:end_dyn_cons]
    shifted_state_cons = y_star[end_dyn_cons + nx:end_state_cons]
    shifted_control_cons = y_star[end_state_cons + nu: end]

    # insert into shifted y_star
    shifted_y_star = shifted_y_star.at[:end_dyn_cons - nx].set(shifted_dyn_cons)
    shifted_y_star = shifted_y_star.at[end_dyn_cons:end_state_cons - nx].set(shifted_state_cons)
    # shifted_y_star = shifted_y_star.at[end_state_cons:-nu].set(shifted_control_cons)

    shifted_y_star = shifted_y_star.at[end_state_cons:end - nu].set(shifted_control_cons)

    # concatentate primal and dual
    shifted_z_star = jnp.concatenate([shifted_x_star, shifted_y_star])

    return shifted_z_star


def single_rollout_theta(rollout_length, T, gamma, dt, w_max, y_max):
    N = rollout_length
    w_traj = np.random.uniform(low=-w_max, high=w_max, size=(2, T + rollout_length))
    v = np.random.uniform(low=-y_max, high=y_max, size=(rollout_length, 2, T))
    # w_traj = np.random.rand(2, T + rollout_length)
    # w = w_noise_var * np.random.randn(rollout_length, 2, T)
    # v = np.random.rand(rollout_length, 2, T)

    # for j in range(N):
    #     inds = np.random.rand(T) <= p
    #     outlier_v = sigma * np.random.randn(2, T)[:, inds]
    #     # weird indexing to avoid transpose
    #     v[j:j+1, :, inds] = outlier_v

    # simulate_fwd_batch = vmap(simulate_x_fwd, in_axes=(0, 0, None, None, None, None), out_axes=(0, 0))

    # get the true X's to have shape (4, T + rollout_length)
    x_traj = simulate_x_fwd(w_traj, rollout_length + T, gamma, dt)

    # this translates to rollout_length problems
    # now get y_mat
    y_mat = np.zeros((rollout_length, 2, T))
    x_trues = np.zeros((rollout_length, 2, T))
    w_trues = np.zeros((rollout_length, 2, T))
    # pos_indices = np.array([0,])
    for i in range(rollout_length):
        curr_x_traj = x_traj[:2, i:i + T]
        y = curr_x_traj + v[i, :, :]
        y_mat[i, :, :] = y
        x_trues[i, :, :] = curr_x_traj

        curr_w_traj = w_traj[:, i:i + T]
        w_trues[i, :, :] = curr_w_traj

    # y_mat, x_trues = simulate_fwd_batch(w, v, T, gamma, dt, B_const)
    # y_mat, x_trues = simulate_fwd_batch(w, v, rollout_length, gamma, dt, B_const)

    # get the rotation angle
    # y_mat has shape (N, 2, T)
    find_rotation_angle_vmap = vmap(find_rotation_angle, in_axes=(0,), out_axes=(0))

    angles = find_rotation_angle_vmap(y_mat[:, :, -1])

    # rotation_vmap = vmap(rotate_vector, in_axes=(0, 0), out_axes=(0))
    clockwise = True
    y_mat_rotated = rotation_vmap(y_mat, angles, clockwise)

    thetas = jnp.zeros((N, 2*T))
    for i in range(N):
        # theta = jnp.ravel(y_mat_rotated[i, :, :].T)
        theta = jnp.ravel(y_mat[i, :, :].T)
        thetas = thetas.at[i, :].set(theta)

    # w_trues = w
    state_pos = jnp.array([0, 1])
    x_states = x_trues[:, state_pos, :]
    x_trues_rotated = rotation_vmap(x_states, angles, clockwise)
    w_trues_rotated = rotation_vmap(w_trues, angles, clockwise)

    return thetas, y_mat, x_trues, w_trues, y_mat_rotated, x_trues_rotated, w_trues_rotated, angles


def sample_theta(num_rollouts, rollout_length, T, sigma, p, gamma, dt, w_noise_var, y_noise_var, B_const=1):
    '''
    v is independent of the rollouts / control setup, but w isn't
    add outliers to v if we want to
        np.random.seed(0)
        inds = np.random.rand(T) <= p
        v[:, inds] = sigma * np.random.randn(2, T)[:, inds]
    '''
    N = num_rollouts * rollout_length

    # generate random input and noise vectors
    # w = w_noise_var * np.random.randn(num_rollouts, 2, T)
    w = 2*np.random.rand(num_rollouts, 2, T) - 1
    v = y_noise_var * np.random.randn(N, 2, T)

    for j in range(N):
        inds = np.random.rand(T) <= p
        outlier_v = sigma * np.random.randn(2, T)[:, inds]

        # weird indexing to avoid transpose
        v[j:j+1, :, inds] = outlier_v

    simulate_fwd_batch = vmap(simulate_fwd, in_axes=(0, 0, None, None, None, None), out_axes=(0, 0))

    y_mat, x_trues = simulate_fwd_batch(w, v, T, gamma, dt, B_const)

    # get the rotation angle
    # y_mat has shape (N, 2, T)
    find_rotation_angle_vmap = vmap(find_rotation_angle, in_axes=(0,), out_axes=(0))

    angles = find_rotation_angle_vmap(y_mat[:, :, -1])

    # rotation_vmap = vmap(rotate_vector, in_axes=(0, 0), out_axes=(0))
    clockwise = True
    y_mat_rotated = rotation_vmap(y_mat, angles, clockwise)

    thetas = jnp.zeros((N, 2*T))
    for i in range(N):
        theta = jnp.ravel(y_mat_rotated[i, :, :].T)
        thetas = thetas.at[i, :].set(theta)

    w_trues = w
    state_pos = jnp.array([0, 1])
    x_states = x_trues[:, state_pos, :]
    x_trues_rotated = rotation_vmap(x_states, angles, clockwise)
    w_trues_rotated = rotation_vmap(w_trues, angles, clockwise)

    return thetas, y_mat, x_trues, w_trues, y_mat_rotated, x_trues_rotated, w_trues_rotated, angles


def rotate_vector(position, angle, clockwise):
    if clockwise:
        angle = -angle
    cos, sin = jnp.cos(angle), jnp.sin(angle)
    rotation_matrix = jnp.array([[cos, -sin], [sin, cos]])
    new_position = rotation_matrix @ position
    return new_position


rotation_vmap = vmap(rotate_vector, in_axes=(0, 0, None), out_axes=(0))
rotation_single = vmap(rotate_vector, in_axes=(0, None, None), out_axes=(0))


def find_rotation_angle(y_T):
    """
    y_mat has shape (2, T)
    give (y_1, ..., y_T) where each entry is in R^2,
    returns the angle
    """
    # # only need the last entry, y_T
    # pos_angle = jnp.arctan(y_T[1] / y_T[0])

    # # if y_T[0] is positive, this is correct
    # # otherwise, angle is
    # if
    # angle = np.pi - pos_angle
    # return angle
    return jnp.arctan2(y_T[1], y_T[0])



def plot_state(t, actual, estimated=None, filename=None):
    '''
    plot position, speed, and acceleration in the x and y coordinates for
    the actual data, and optionally for the estimated data
    '''
    trajectories = [actual]
    if estimated is not None:
        trajectories.append(estimated)

    fig, ax = plt.subplots(3, 2, sharex='col', sharey='row', figsize=(8, 8))
    for x, w in trajectories:
        ax[0, 0].plot(t, x[0, :-1])
        ax[0, 1].plot(t, x[1, :-1])
        ax[1, 0].plot(t, x[2, :-1])
        ax[1, 1].plot(t, x[3, :-1])
        ax[2, 0].plot(t, w[0, :])
        ax[2, 1].plot(t, w[1, :])

    ax[0, 0].set_ylabel('x position')
    ax[1, 0].set_ylabel('x velocity')
    ax[2, 0].set_ylabel('x input')

    ax[0, 1].set_ylabel('y position')
    ax[1, 1].set_ylabel('y velocity')
    ax[2, 1].set_ylabel('y input')

    ax[0, 1].yaxis.tick_right()
    ax[1, 1].yaxis.tick_right()
    ax[2, 1].yaxis.tick_right()

    ax[0, 1].yaxis.set_label_position("right")
    ax[1, 1].yaxis.set_label_position("right")
    ax[2, 1].yaxis.set_label_position("right")

    ax[2, 0].set_xlabel('time')
    ax[2, 1].set_xlabel('time')
    if filename:
        fig.savefig(filename, bbox_inches='tight')
    else:
        plt.show()


def plot_positions(traj, labels, axis=None, filename=None):
    '''
    show point clouds for true, observed, and recovered positions
    '''
    # matplotlib.rcParams.update({'font.size': 14})
    n = len(traj)

    fig, ax = plt.subplots(1, n, sharex=True, sharey=True, figsize=(12, 5))
    if n == 1:
        ax = [ax]

    for i, x in enumerate(traj):
        ax[i].plot(x[0, :], x[1, :], 'ro', alpha=.1)
        ax[i].set_title(labels[i])
        if axis:
            ax[i].axis(axis)

    if filename:
        fig.savefig(filename, bbox_inches='tight')
    else:
        plt.show()


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """

    if color in mc.cnames.keys():
        c = mc.cnames[color]
    else:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_positions_overlay(traj, labels, num_dots=2, grayscales=[.8, .3, 1.0, 0.7, 1.0], axis=None, filename=None, legend=False, noise_only=False):
    '''
    show point clouds for true, observed, and recovered positions

    the first num_dots trajectories are given as scatter plots (dots)
    the rest of the trajectories are given as continuous lines
    '''
    n = len(traj)

    colors = ['green', 'red', 'blue', 'gray', 'orange']
    # cmap = plt.cm.Set1
    # colors = [cmap.colors[0], cmap.colors[1], cmap.colors[2], cmap.colors[3], cmap.colors[4]]
    # linestyles = ['o', 'o', '-.', ':', '--']
    linestyles = ['o', 'o', '-', '-', '-']

    fig = plt.figure(figsize=(6, 6))

    for i in range(n - 2):
        shade = (i + 1) / (n - 2)
        colors.append(lighten_color('blue', shade))

    for i, x in enumerate(traj):
        alpha = grayscales[i] #1 #grayscales[i]
        if i < num_dots:
            if noise_only:
                if i > 0:
                    plt.plot(x[0, :], x[1, :], 'o', color=colors[i], alpha=alpha, label=labels[i]) #, markersize=1)
            else:
                plt.plot(x[0, :], x[1, :], 'o', color=colors[i], alpha=alpha, label=labels[i]) #, markersize=1)
        else:
            plt.plot(x[0, :], x[1, :], color=colors[i], linestyle=linestyles[i], alpha=alpha, label=labels[i]) #, markersize=1)

    # save with legend
    if legend:
        plt.legend()

    plt.xticks([])
    plt.yticks([])
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
    plt.clf()


def plot_positions_overlay_genL2O(noisy, optimal, cold, num_dots=2, grayscales=[.8, .3, 1.0, 0.7, 1.0], axis=None, filename=None, legend=False, noise_only=False):
    '''
    show point clouds for true, observed, and recovered positions

    the first num_dots trajectories are given as scatter plots (dots)
    the rest of the trajectories are given as continuous lines
    '''
    # n = len(traj)

    colors = ['red', 'green', 'gray', 'orange', 'blue']
    # cmap = plt.cm.Set1
    # colors = [cmap.colors[0], cmap.colors[1], cmap.colors[2], cmap.colors[3], cmap.colors[4]]
    linestyles = ['o', 'o', '-.', ':', '--']

    # for i in range(n - 2):
    #     shade = (i + 1) / (n - 2)
    #     colors.append(lighten_color('blue', shade))

    # for i, x in enumerate(traj):
    # alpha = grayscales[i] #1 #grayscales[i]

    fig, ax = plt.subplots(figsize=(6, 6))

    markersize = 2.0

    # plot noisy
    ax.plot(noisy[0, :], noisy[1, :], 'o', color=colors[0], alpha=0.3, markersize=markersize)

    # plot optimal
    ax.plot(optimal[0, :], optimal[1, :], 'o', color=colors[1], alpha=1.0, markersize=markersize)

    # plot cold
    # ax.plot(cold[0, :], cold[1, :], 'o', color=colors[4], alpha=1.0, markersize=markersize)

    # plot shaded region
    # plt.plot(optimal[0, :], optimal[1, :], 'o', color=colors[i], alpha=1.0, label=labels[i], markersize=1)

    # fig, ax = plt.subplots()  # Create a figure and an axes
    radius = 0.1
    radius2 = 0.01

    from matplotlib.patches import Circle

    # Loop through the points and plot a circle at each point
    for i in range(optimal.shape[1]):
        circle = Circle((optimal[0, i], optimal[1, i]), radius, color=colors[3], alpha=0.2)
        ax.add_patch(circle)

        # circle = Circle((optimal[0, i], optimal[1, i]), radius2, color=colors[4], alpha=0.2)
        # ax.add_patch(circle)

    x_min, x_max, y_min, y_max = adjust_xy_min_max(noisy[0,:].min(), noisy[0,:].max(), noisy[1,:].min(), noisy[1,:].max())
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])


    axins = inset_axes(ax, width="40%", height="40%", loc='lower left',
                   bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
                   bbox_transform=ax.transAxes)

    # Optional: Set limits if you want to specify the bounds of your plot
    # ax.set_xlim([xmin, xmax])
    # ax.set_ylim([ymin, ymax])
    # Plot the same data on inset
    inset_markersize = 7
    axins.plot(noisy[0, :], noisy[1, :], 'o', color=colors[0], alpha=0.3, markersize=inset_markersize)
    axins.plot(optimal[0, :], optimal[1, :], 'o', color=colors[1], alpha=1.0, markersize=inset_markersize)
    # axins.plot(cold[0, :], cold[1, :], 'o', color=colors[4], alpha=1.0, markersize=inset_markersize)

    # Specify the region for zooming (can adjust as necessary)
    # x1, x2, y1, y2 = -0.2, 0.2, -0.2, 0.2  # Define zoom boundaries here
    # mid = 25
    center_x = optimal[0, 15]
    center_y = optimal[1, 15]
    width = 0.25
    x1, x2, y1, y2 = -width + center_x, width + center_x, -width + center_y, width + center_y
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    # Optionally, add circles in the zoomed region as well
    for i in range(optimal.shape[1]):
        if x1 < optimal[0, i] < x2 and y1 < optimal[1, i] < y2:
            circle = Circle((optimal[0, i], optimal[1, i]), radius, color=colors[3], alpha=0.2)
            axins.add_patch(circle)

            # circle = Circle((optimal[0, i], optimal[1, i]), radius2, color=colors[4], alpha=0.2)
            # axins.add_patch(circle)
    axins.set_xticks([])
    axins.set_yticks([])
    ax.set_xticks([])
    ax.set_yticks([])

    ax.indicate_inset_zoom(axins, edgecolor="black")
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    ax.set_aspect('equal')  # This ensures the circle is not oval in case the ax
    
    # is units are not equal

    


    # save with legend
    if legend:
        plt.legend()

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
    plt.clf()


def adjust_xy_min_max(x_min, x_max, y_min, y_max):
    # Calculate the centers of the x and y ranges
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0

    # Calculate the spans (ranges) of the x and y ranges
    x_span = x_max - x_min
    y_span = y_max - y_min

    # Determine the maximum span (range) between x and y
    max_span = max(x_span, y_span)
    # max_span = 6

    # Calculate new min and max for x and y to ensure equal ranges
    cushion = 0.2
    x_min_new = x_center - max_span / 2.0 - cushion
    x_max_new = x_center + max_span / 2.0 + cushion
    y_min_new = y_center - max_span / 2.0 - cushion
    y_max_new = y_center + max_span / 2.0 + cushion
    return x_min_new, x_max_new, y_min_new, y_max_new


partial(jit, static_argnums=(2, 3, 4))


def simulate_fwd(w_mat, v_mat, T, gamma, dt):
    A, B, C = robust_kalman_setup(gamma, dt)

    x = jnp.zeros((4, T + 1))
    y_mat = jnp.zeros((2, T))

    # simulate the system forward in time
    for t in range(T):
        y_mat = y_mat.at[:, t].set(C.dot(x[:, t]) + v_mat[:, t])
        x = x.at[:, t + 1].set(A.dot(x[:, t]) + B.dot(w_mat[:, t]))

    return y_mat, x


def simulate_x_fwd(w_mat, T, gamma, dt):
    A, B, C = robust_kalman_setup(gamma, dt)

    x_mat = jnp.zeros((4, T + 1))

    # simulate the system forward in time
    for t in range(T):
        x_mat = x_mat.at[:, t + 1].set(A.dot(x_mat[:, t]) + B.dot(w_mat[:, t]))

    return x_mat


def get_x_w_true(theta, T, gamma, dt):
    A, B, C = robust_kalman_setup(gamma, dt)
    nu, no = 2, 2

    # extract (w, v)

    # theta = (w_0,...,w_{T-1},v_0,...,v_{T-1})

    # get y
    w = theta[: T * nu]
    v = theta[T * nu:]
    w_mat = jnp.reshape(w, (nu, T))
    v_mat = jnp.reshape(v, (no, T))
    # y_mat = simulate_fwd(w_mat, v_mat, T, gamma, dt)
    x = jnp.zeros((4, T + 1))
    y_mat = jnp.zeros((2, T))

    # simulate the system forward in time
    for t in range(T):
        y_mat = y_mat.at[:, t].set(C.dot(x[:, t]) + v_mat[:, t])
        x = x.at[:, t + 1].set(A.dot(x[:, t]) + B.dot(w_mat[:, t]))
    # y = jnp.ravel(y_mat.T)
    return x, w_mat


# @functools.partial(jit, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8,))
# def single_q(theta, m, n, T, nx, nu, state_box, control_box, A_dynamics):
def single_q(theta, mu, rho, T, gamma, dt):
    """
    the observations, y_0,...,y_{T-1} are the parameters that change
    there are 6 blocks of constraints and the second one changes
    1. x_{t+1}=Ax_t+Bw_t
    2. y_t=Cx_t+v_t
    3. ...
    """
    nx, nu, no = 4, 2, 2
    single_len = nx + nu + no + 3
    nvars = single_len * T

    # extract (w, v)

    # theta = (w_0,...,w_{T-1},v_0,...,v_{T-1})

    # get y
    # w = theta[: T * nu]
    # v = theta[T * nu :]
    # w_mat = jnp.reshape(w, (nu, T))
    # v_mat = jnp.reshape(v, (no, T))
    # y_mat = simulate_fwd(w_mat, v_mat, T, gamma, dt)
    # y = jnp.ravel(y_mat.T)
    y = theta

    # c
    c = jnp.zeros(single_len * T)
    c = c.at[-2 * T: -T].set(2 * mu)

    # b
    b_dyn = jnp.zeros((T-1) * nx)
    # b_obs = jnp.zeros(T * no)
    b_obs = y

    # aux constraints
    n_aux = T
    b_aux = jnp.zeros(n_aux)

    # z_ineq constraints
    n_z_ineq = T
    b_z_ineq = rho * jnp.ones(n_z_ineq)

    # u_ineq constraints
    n_u_ineq = T
    b_u_ineq = jnp.zeros(n_u_ineq)

    # socp constraints
    n_socp = T * 3
    b_socp = jnp.zeros(n_socp)

    # get b
    b = jnp.hstack([b_dyn, b_obs, b_aux, b_z_ineq, b_u_ineq, b_socp])

    # q = jnp.zeros(n + m)
    # beq = jnp.zeros(T * nx)
    # beq = beq.at[:nx].set(A_dynamics @ theta)
    # b_upper = jnp.hstack([state_box * jnp.ones(T * nx), control_box * jnp.ones(T * nu)])
    # b_lower = jnp.hstack([state_box * jnp.ones(T * nx), control_box * jnp.ones(T * nu)])
    # b = jnp.hstack([beq, b_upper, b_lower])

    # q
    m = b.size
    q = jnp.zeros(m + nvars)
    q = q.at[:nvars].set(c)
    q = q.at[nvars:].set(b)
    # print('y', y)

    return q


def rkf_loss(z_next, z_star, T):
    x_mat = jnp.reshape(z_next[:2 * T], (T, 2))
    x_star_mat = jnp.reshape(z_star[:T * 2], (T, 2))
    norms = jnp.linalg.norm(x_mat - x_star_mat, axis=1)
    max_norm = jnp.max(norms)
    return max_norm


def run(run_cfg, lah=True):
    """
    retrieve data for this config
    theta is all of the following
    theta = (ret, pen_risk, pen_hold, pen_trade, w0)

    Sigma is constant

     just need (theta, factor, u_star), Pi
    """
    # todo: retrieve data and put into a nice form - OR - just save to nice form

    """
    create workspace
    needs to know the following somehow -- from the run_cfg
    1. nn cfg
    2. (theta, factor, u_star)_i=1^N
    3. Pi

    2. and 3. are stored in data files and the run_cfg holds the location

    it will create the l2a_model
    """
    datetime = run_cfg.data.datetime
    orig_cwd = hydra.utils.get_original_cwd()
    example = "robust_kalman"
    # folder = f"{orig_cwd}/outputs/{example}/aggregate_outputs/{datetime}"
    # data_yaml_filename = f"{folder}/data_setup_copied.yaml"
    data_yaml_filename = 'data_setup_copied.yaml'

    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            setup_cfg = {}
    A, B, C = robust_kalman_setup(setup_cfg['gamma'], setup_cfg['dt'])
    P = get_P(A, B, C, setup_cfg['T'], setup_cfg['mu'])
    
    n = P.shape[0]
    l = -np.ones(n) * setup_cfg['w_max']
    l[:4] = -np.inf
    u = np.ones(n) * setup_cfg['w_max']
    u[:4] = np.inf
    static_dict = dict(P=jnp.array(P), ista_step=.01, l=jnp.array(l), u=jnp.array(u))

    # we directly save q now
    static_flag = True
    # if model == 'lah':
    #     algo = 'lah_osqp'
    # elif model == 'l2ws':
    #     algo = 'osqp'
    # elif model == 'lm':
    #     algo = 'lm_osqp'
    algo = 'lah_accel_box_qp'
    
    # vis_fn = partial(custom_visualize_fn, figsize=img_size**2, deblur_or_denoise=deblur_or_denoise)
    # workspace = Workspace(algo, run_cfg, static_flag, static_dict, example, custom_visualize_fn=vis_fn)
    workspace = Workspace(algo, run_cfg, static_flag, static_dict, example)

    # run the workspace
    workspace.run()

    # non-identity DR scaling
    # rho_x = run_cfg.get('rho_x', 1)
    # scale = run_cfg.get('scale', 1)

    # static_dict = static_canon(
    #     setup_cfg['T'],
    #     setup_cfg['gamma'],
    #     setup_cfg['dt'],
    #     setup_cfg['mu'],
    #     setup_cfg['rho'],
    #     setup_cfg['B_const'],
    #     rho_x=rho_x,
    #     scale=scale
    # )

    # get_q = None

    """
    static_flag = True
    means that the matrices don't change across problems
    we only need to factor once
    """
    static_flag = True

    custom_visualize_fn_partial = partial(custom_visualize_fn, T=setup_cfg['T'])
    algo = 'lah_accel_scs' if lah else 'lm_scs'

    # A = static_dict['A_sparse']
    m, n = A.shape
    partial_shifted_sol_fn = partial(shifted_sol, T=setup_cfg['T'],  m=m, n=n)
    batch_shifted_sol_fn = vmap(partial_shifted_sol_fn, in_axes=(0), out_axes=(0))


    workspace = Workspace(algo, run_cfg, static_flag, static_dict, example,
                        #   custom_visualize_fn=custom_visualize_fn_partial,
                          shifted_sol_fn=batch_shifted_sol_fn,
                          traj_length=setup_cfg['rollout_length'])

    """
    run the workspace
    """
    workspace.run()


def l2ws_run(run_cfg):
    datetime = run_cfg.data.datetime
    orig_cwd = hydra.utils.get_original_cwd()
    example = "robust_kalman"
    # folder = f"{orig_cwd}/outputs/{example}/aggregate_outputs/{datetime}"
    # data_yaml_filename = f"{folder}/data_setup_copied.yaml"
    data_yaml_filename = 'data_setup_copied.yaml'

    # read the yaml file
    with open(data_yaml_filename, "r") as stream:
        try:
            setup_cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            setup_cfg = {}

    # non-identity DR scaling
    rho_x = run_cfg.get('rho_x', 1)
    scale = run_cfg.get('scale', 1)

    static_dict = static_canon(
        setup_cfg['T'],
        setup_cfg['gamma'],
        setup_cfg['dt'],
        setup_cfg['mu'],
        setup_cfg['rho'],
        setup_cfg['B_const'],
        rho_x=rho_x,
        scale=scale
    )

    get_q = None

    """
    static_flag = True
    means that the matrices don't change across problems
    we only need to factor once
    """
    static_flag = True

    custom_visualize_fn_partial = partial(custom_visualize_fn, T=setup_cfg['T'])
    algo = 'scs'

    A = static_dict['A_sparse']
    m, n = A.shape
    partial_shifted_sol_fn = partial(shifted_sol, T=setup_cfg['T'],  m=m, n=n)
    batch_shifted_sol_fn = vmap(partial_shifted_sol_fn, in_axes=(0), out_axes=(0))

    custom_loss = partial(rkf_loss, T=setup_cfg['T'])

    workspace = Workspace(algo, run_cfg, static_flag, static_dict, example,
                          custom_visualize_fn=custom_visualize_fn_partial,
                          custom_loss=custom_loss,
                          shifted_sol_fn=batch_shifted_sol_fn,
                          traj_length=setup_cfg['rollout_length'])

    """
    run the workspace
    """
    workspace.run()
    
    
def get_P(A, B, C, T, tau):
    """
    Construct QP matrices for a batch of problems with x0 as a decision variable:
        min_{x0, z} (1/2) z^T z + tau * sum_t ||C x_t(x0, z) - y_t||_2^2
        where x_t = A^t x0 + sum_{s=0}^{t-1} A^{t-1-s} B w_s

    Args:
        A (ndarray): (n x n) system dynamics matrix
        B (ndarray): (n x m) control matrix
        C (ndarray): (p x n) observation matrix
        y_tensor (ndarray): (N x p x T) batch of measurements
        tau (float): weight on measurement error

    Returns:
        P (ndarray): ((n + mT) x (n + mT)) shared quadratic matrix
        c_mat (ndarray): (N x (n + mT)) linear term for each problem
    """
    # N, p, T = y_tensor.shape
    n, m = B.shape

    dim = n + m * T  # total dimension of decision variable [x0; z]
    P = np.zeros((dim, dim))
    # c_mat = np.zeros((N, dim))

    # Precompute A powers
    A_powers = [np.linalg.matrix_power(A, i) for i in range(T + 1)]

    # Construct dynamic mappings: x_t = A^t x0 + M_t z
    M_list = []
    for t in range(T):
        M_t = np.zeros((n, m * T))
        for s in range(t):
            A_term = A_powers[t - 1 - s]
            M_t[:, m * s: m * (s + 1)] = A_term @ B
        M_list.append(M_t)

    # Quadratic term in z (control effort)
    P[n:, n:] = np.eye(m * T) * 1e-3

    # Add tau * ||C x_t(x0, z) - y_t||^2 terms
    for t in range(T):
        A_t = A_powers[t]
        M_t = M_list[t]
        C_A = C @ A_t
        C_M = C @ M_t

        # Update P (block structure)
        P[:n, :n] += tau * C_A.T @ C_A
        P[:n, n:] += tau * C_A.T @ C_M
        P[n:, :n] += tau * C_M.T @ C_A
        P[n:, n:] += tau * C_M.T @ C_M
    return P
        
    
def build_batched_qp(A, B, C, y_tensor, tau):
    """
    Construct QP matrices for a batch of problems with x0 as a decision variable:
        min_{x0, z} (1/2) z^T z + tau * sum_t ||C x_t(x0, z) - y_t||_2^2
        where x_t = A^t x0 + sum_{s=0}^{t-1} A^{t-1-s} B w_s

    Args:
        A (ndarray): (n x n) system dynamics matrix
        B (ndarray): (n x m) control matrix
        C (ndarray): (p x n) observation matrix
        y_tensor (ndarray): (N x p x T) batch of measurements
        tau (float): weight on measurement error

    Returns:
        P (ndarray): ((n + mT) x (n + mT)) shared quadratic matrix
        c_mat (ndarray): (N x (n + mT)) linear term for each problem
    """
    N, p, T = y_tensor.shape
    n, m = B.shape

    dim = n + m * T  # total dimension of decision variable [x0; z]
    P = np.zeros((dim, dim))
    c_mat = np.zeros((N, dim))

    # Precompute A powers
    A_powers = [np.linalg.matrix_power(A, i) for i in range(T + 1)]

    # Construct dynamic mappings: x_t = A^t x0 + M_t z
    M_list = []
    for t in range(T):
        M_t = np.zeros((n, m * T))
        for s in range(t):
            A_term = A_powers[t - 1 - s]
            M_t[:, m * s: m * (s + 1)] = A_term @ B
        M_list.append(M_t)

    # Quadratic term in z (control effort)
    # P[n:, n:] = tau * np.eye(m * T)
    # P[n:, n:] = np.eye(m * T) * 1

    # # Add tau * ||C x_t(x0, z) - y_t||^2 terms
    # for t in range(T):
    #     A_t = A_powers[t]
    #     M_t = M_list[t]
    #     C_A = C @ A_t
    #     C_M = C @ M_t

    #     # Update P (block structure)
    #     P[:n, :n] += tau * C_A.T @ C_A
    #     P[:n, n:] += tau * C_A.T @ C_M
    #     P[n:, :n] += tau * C_M.T @ C_A
    #     P[n:, n:] += tau * C_M.T @ C_M
    P = get_P(A, B, C, T, tau)


    # Compute c_mat (one row per problem)
    for i in range(N):
        c_i = np.zeros(dim)
        for t in range(T):
            y_t = y_tensor[i, :, t]
            A_t = A_powers[t]
            M_t = M_list[t]

            # c_i[:n] += -2 * C_A.T @ y_t
            # c_i[n:] += -2 * C_M.T @ y_t
            C_A = C @ A_t
            C_M = C @ M_t
            c_i[:n] += -1 * tau * C_A.T @ y_t
            c_i[n:] += -1 * tau * C_M.T @ y_t
        c_mat[i] = c_i

    return P, c_mat


def recover_states(z, A, B):
    """
    Recover full state trajectory from z = (x0, w0, ..., w_{T-1}).

    Args:
        z (ndarray): (n + m*T,) flattened vector of initial state and control inputs
        A (ndarray): (n x n) state transition matrix
        B (ndarray): (n x m) control input matrix

    Returns:
        x_mat (ndarray): (n x (T+1)) matrix of state trajectory
    """
    n, m = B.shape
    T = (z.shape[0] - n) // m
    x_mat = np.zeros((n, T + 1))

    # Extract x0 and control sequence
    x_mat[:, 0] = z[:n]
    for t in range(T):
        w_t = z[n + m * t : n + m * (t + 1)]
        x_mat[:, t + 1] = A @ x_mat[:, t] + B @ w_t

    return x_mat


def setup_probs(setup_cfg):
    print("entered robust kalman setup", flush=True)
    cfg = setup_cfg
    # N_train, N_test = cfg.N_train, cfg.N_test
    # N = N_train + N_test
    N = cfg.num_rollouts * cfg.rollout_length
    
    """
    sample theta and get y for each problem
    """
    outs = []
    for i in range(cfg.num_rollouts):
        out = single_rollout_theta(cfg.rollout_length, cfg.T, cfg.gamma, cfg.dt, cfg.w_max, cfg.y_max)
        outs.append(out)
    theta_mat, y_mat, x_trues, w_trues, y_mat_rotated, x_trues_rotated, w_trues_rot, angles = compile_outs(outs)
    
    # y_mat = y_mat - y_mat.mean()
    off_center = y_mat.mean(axis=2, keepdims=True)
    y_mat = y_mat - off_center
    x_trues = x_trues - off_center
    
    # create P, q_mat, l, u
    A, B, C = robust_kalman_setup(cfg.gamma, cfg.dt)
    P, q_mat = build_batched_qp(A, B, C, y_mat, cfg.mu)
    n = P.shape[0]
    l = -np.ones(n) * cfg.w_max
    # l = l.at[:4].set(-np.inf)
    l[:4] = -np.inf
    u = np.ones(n) * cfg.w_max
    # u = u.at[:4].set(np.inf)
    u[:4] = np.inf
    
    c_param = cp.Parameter(n)
    z = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(.5 * cp.quad_form(z, P) + c_param @ z), [l <= z, z <= u])
    z_stars = np.zeros((N, n))
    for i in range(N):
        c_param.value = q_mat[i,:]
        prob.solve(verbose=True)
        z_stars[i,:] = z.value
    
    # transform z = (x_0, w_0, ..., w_{T-1}) to estimated states (x_0, ..., x_{T-1})
    
    # save the results
    # save the data
    output_filename = f"{os.getcwd()}/data_setup"
    log.info("final saving final data...")
    t0 = time.time()
    jnp.savez(
        output_filename,
        thetas=theta_mat,
        z_stars=z_stars,
        q_mat=q_mat
    )
    for j in range(3):
        start = j * cfg.rollout_length
        for i in range(10):
            x_mat = recover_states(z_stars[i+start,:], A, B)
            plt.scatter(y_mat[i+start][0,:], y_mat[i+start][1,:])
            plt.scatter(x_trues[i+start][0,:], x_trues[i+start][1,:])
            plt.scatter(x_mat[0,:], x_mat[1,:])
            plt.savefig(f'plot_{j}_{i}.pdf')
            plt.clf()
    import pdb
    pdb.set_trace()
    return

    """
    - canonicalize according to whether we have states or not
    - extract information dependent on the setup
    """
    log.info("creating static canonicalization...")
    t0 = time.time()
    out_dict = static_canon(
        cfg.T,
        cfg.gamma,
        cfg.dt,
        cfg.mu,
        cfg.rho,
        cfg.B_const
    )

    t1 = time.time()
    log.info(f"finished static canonicalization - took {t1-t0} seconds")

    cones_dict = out_dict["cones_dict"]
    A_sparse, P_sparse = out_dict["A_sparse"], out_dict["P_sparse"]

    b, c = out_dict["b"], out_dict["c"]

    m, n = A_sparse.shape

    """
    save output to output_filename
    """
    output_filename = f"{os.getcwd()}/data_setup"
    
    
    data = dict(P=P_sparse, A=A_sparse, b=b, c=c)
    tol_abs = cfg.solve_acc_abs
    tol_rel = cfg.solve_acc_rel
    solver = scs.SCS(data,
                     cones_dict,
                     normalize=False,
                     scale=1,
                     adaptive_scale=False,
                     rho_x=1,
                     alpha=1,
                     acceleration_lookback=0,
                     eps_abs=tol_abs,
                     eps_rel=tol_rel)
    # solve_times = np.zeros(N)
    # x_stars = jnp.zeros((N, n))
    # y_stars = jnp.zeros((N, m))
    # s_stars = jnp.zeros((N, m))
    # q_mat = jnp.zeros((N, m + n))

    
    
    # out = sample_theta(N, cfg.T, cfg.sigma, cfg.p, cfg.gamma, cfg.dt,
    #                    cfg.w_noise_var, cfg.y_noise_var, cfg.B_const)
    # thetas_np, y_mat, x_trues, w_trues, y_mat_rotated, x_trues_rotated, w_trues_rot, angles = out
    thetas = jnp.array(thetas_np)

    batch_q = vmap(single_q, in_axes=(0, None, None, None, None, None), out_axes=(0))
    q_mat = batch_q(thetas, cfg.mu, cfg.rho, cfg.T, cfg.gamma, cfg.dt)

    # scs_instances = []


    x_stars, y_stars, s_stars = setup_script(q_mat, thetas, solver, data, cones_dict, output_filename, solve=True)

    time_limit = cfg.dt * cfg.T
    ts, delt = np.linspace(0, time_limit, cfg.T-1, endpoint=True, retstep=True)

    os.mkdir("states_plots")
    os.mkdir("positions_plots")
    for i in range(25):
        x_state = x_stars[i, :cfg.T * 4]
        x1_kalman = x_state[0::4]
        x2_kalman = x_state[1::4]
        x_kalman = jnp.stack([x1_kalman, x2_kalman])

        # plot original
        # rotate back the output, x_kalman
        clockwise = False

        x_kalman_rotated_transpose = rotation_single(x_kalman.T, angles[i], clockwise)
        x_kalman_rotated = x_kalman_rotated_transpose.T

        # plot original
        plot_positions_overlay([x_kalman_rotated, y_mat[i, :, :]],
                               ['True', 'KF recovery', 'Noisy'],
                               filename=f"positions_plots/positions_{i}_noise.pdf",
                               noise_only=True)
        plot_positions_overlay([x_kalman_rotated, y_mat[i, :, :]],
                               ['True', 'KF recovery', 'Noisy'],
                               filename=f"positions_plots/positions_{i}.pdf")

        plot_positions_overlay([x_kalman, y_mat_rotated[i, :, :]],
                               ['True', 'KF recovery', 'Noisy'],
                               grayscales=[1.0, 1.0, 1.0, 1.0, 1.0],
                               filename=f"positions_plots/positions_{i}_rotated.pdf")
        plot_positions_overlay_genL2O(y_mat[i, :, :], x_kalman_rotated,
                                      filename=f"positions_plots/positions_{i}_genL2O.pdf")
        # plot_positions_overlay([x_trues[i, :, :-1], x_kalman_rotated, y_mat[i, :, :]],
        #                        ['True', 'KF recovery', 'Noisy'],
        #                        filename=f"positions_plots/positions_{i}.pdf")

        # plot_positions_overlay([x_trues_rotated[i, :, :-1], x_kalman, y_mat_rotated[i, :, :]],
        #                        ['True', 'KF recovery', 'Noisy'],
        #                        filename=f"positions_plots/positions_{i}_rotated.pdf")
        
    

def setup_probs2(setup_cfg):
    print("entered robust kalman setup", flush=True)
    cfg = setup_cfg
    # N_train, N_test = cfg.N_train, cfg.N_test
    # N = N_train + N_test
    N = cfg.num_rollouts * cfg.rollout_length

    """
    - canonicalize according to whether we have states or not
    - extract information dependent on the setup
    """
    log.info("creating static canonicalization...")
    t0 = time.time()
    out_dict = static_canon(
        cfg.T,
        cfg.gamma,
        cfg.dt,
        cfg.mu,
        cfg.rho,
        cfg.B_const
    )

    t1 = time.time()
    log.info(f"finished static canonicalization - took {t1-t0} seconds")

    cones_dict = out_dict["cones_dict"]
    A_sparse, P_sparse = out_dict["A_sparse"], out_dict["P_sparse"]

    b, c = out_dict["b"], out_dict["c"]

    m, n = A_sparse.shape

    """
    save output to output_filename
    """
    output_filename = f"{os.getcwd()}/data_setup"
    """
    create scs solver object
    we can cache the factorization if we do it like this
    """

    data = dict(P=P_sparse, A=A_sparse, b=b, c=c)
    tol_abs = cfg.solve_acc_abs
    tol_rel = cfg.solve_acc_rel
    solver = scs.SCS(data,
                     cones_dict,
                     normalize=False,
                     scale=1,
                     adaptive_scale=False,
                     rho_x=1,
                     alpha=1,
                     acceleration_lookback=0,
                     eps_abs=tol_abs,
                     eps_rel=tol_rel)
    # solve_times = np.zeros(N)
    # x_stars = jnp.zeros((N, n))
    # y_stars = jnp.zeros((N, m))
    # s_stars = jnp.zeros((N, m))
    # q_mat = jnp.zeros((N, m + n))

    """
    sample theta and get y for each problem
    """
    outs = []
    for i in range(cfg.num_rollouts):
        out = single_rollout_theta(cfg.rollout_length, cfg.T, cfg.sigma, cfg.p, 
                                cfg.gamma, cfg.dt, cfg.w_noise_var, cfg.y_noise_var, cfg.B_const)
        outs.append(out)
    thetas_np, y_mat, x_trues, w_trues, y_mat_rotated, x_trues_rotated, w_trues_rot, angles = compile_outs(outs)
    
    # out = sample_theta(N, cfg.T, cfg.sigma, cfg.p, cfg.gamma, cfg.dt,
    #                    cfg.w_noise_var, cfg.y_noise_var, cfg.B_const)
    # thetas_np, y_mat, x_trues, w_trues, y_mat_rotated, x_trues_rotated, w_trues_rot, angles = out
    thetas = jnp.array(thetas_np)

    batch_q = vmap(single_q, in_axes=(0, None, None, None, None, None), out_axes=(0))
    q_mat = batch_q(thetas, cfg.mu, cfg.rho, cfg.T, cfg.gamma, cfg.dt)

    # scs_instances = []


    x_stars, y_stars, s_stars = setup_script(q_mat, thetas, solver, data, cones_dict, output_filename, solve=True)

    time_limit = cfg.dt * cfg.T
    ts, delt = np.linspace(0, time_limit, cfg.T-1, endpoint=True, retstep=True)

    os.mkdir("states_plots")
    os.mkdir("positions_plots")
    for i in range(25):
        x_state = x_stars[i, :cfg.T * 4]
        x1_kalman = x_state[0::4]
        x2_kalman = x_state[1::4]
        x_kalman = jnp.stack([x1_kalman, x2_kalman])

        # plot original
        # rotate back the output, x_kalman
        clockwise = False

        x_kalman_rotated_transpose = rotation_single(x_kalman.T, angles[i], clockwise)
        x_kalman_rotated = x_kalman_rotated_transpose.T

        # plot original
        plot_positions_overlay([x_kalman_rotated, y_mat[i, :, :]],
                               ['True', 'KF recovery', 'Noisy'],
                               filename=f"positions_plots/positions_{i}_noise.pdf",
                               noise_only=True)
        plot_positions_overlay([x_kalman_rotated, y_mat[i, :, :]],
                               ['True', 'KF recovery', 'Noisy'],
                               filename=f"positions_plots/positions_{i}.pdf")

        plot_positions_overlay([x_kalman, y_mat_rotated[i, :, :]],
                               ['True', 'KF recovery', 'Noisy'],
                               grayscales=[1.0, 1.0, 1.0, 1.0, 1.0],
                               filename=f"positions_plots/positions_{i}_rotated.pdf")
        plot_positions_overlay_genL2O(y_mat[i, :, :], x_kalman_rotated,
                                      filename=f"positions_plots/positions_{i}_genL2O.pdf")
        # plot_positions_overlay([x_trues[i, :, :-1], x_kalman_rotated, y_mat[i, :, :]],
        #                        ['True', 'KF recovery', 'Noisy'],
        #                        filename=f"positions_plots/positions_{i}.pdf")

        # plot_positions_overlay([x_trues_rotated[i, :, :-1], x_kalman, y_mat_rotated[i, :, :]],
        #                        ['True', 'KF recovery', 'Noisy'],
        #                        filename=f"positions_plots/positions_{i}_rotated.pdf")
        
def compile_outs(outs):
    thetas_np = np.stack([item for rollout_result in outs for item in rollout_result[0]])
    y_mat = np.stack([item for rollout_result in outs for item in rollout_result[1]])
    x_trues = np.stack([item for rollout_result in outs for item in rollout_result[2]])
    w_trues = np.stack([item for rollout_result in outs for item in rollout_result[3]])
    y_mat_rotated = np.stack([item for rollout_result in outs for item in rollout_result[4]])
    x_trues_rotated = np.stack([item for rollout_result in outs for item in rollout_result[5]])
    w_trues_rot = np.stack([item for rollout_result in outs for item in rollout_result[6]])
    angles = np.stack([item for rollout_result in outs for item in rollout_result[7]])

    return thetas_np, y_mat, x_trues, w_trues, y_mat_rotated, x_trues_rotated, w_trues_rot, angles


def custom_visualize_fn(x_primals, x_stars, x_prev_sol, x_nn, thetas, iterates, visual_path, T, num=50):
    """
    assume len(iterates) == 1 for now
        point is to compare no-learning vs learned for 20 iterations
    """
    assert len(iterates) == 1
    num = np.min([x_stars.shape[0], num])

    y_mat_rotated = jnp.reshape(thetas[:num, :], (num, T, 2))


    for i in range(num):
        titles = ['optimal solution', 'noisy trajectory']
        x_true_kalman = get_x_kalman_from_x_primal(x_stars[i, :], T)
        traj = [x_true_kalman, y_mat_rotated[i, :].T]

        for j in range(len(iterates)):
            iter = iterates[j]
    #         x_prev_sol_kalman = get_x_kalman_from_x_primal(x_prev_sol[i, iter, :], T)
            x_hat_kalman = get_x_kalman_from_x_primal(x_primals[i, iter, :], T)
            x_nn_kalman = get_x_kalman_from_x_primal(x_nn[i, iter, :], T)
            # traj.append(x_prev_sol_kalman)
            traj.append(x_nn_kalman)
            traj.append(x_hat_kalman)
            titles.append(f"nearest neighbor: ${iter}$ iters")
            titles.append(f"learned: ${iter}$ iters")

            # df.to_csv(f"{visual_path}/positions_{i}_rotated.pdf")


        plot_positions_overlay(traj, titles, filename=f"{visual_path}/positions_{i}_rotated_legend.pdf", legend=True)
        plot_positions_overlay(traj, titles, filename=f"{visual_path}/positions_{i}_rotated.pdf", legend=False)
    
    # saving to df
    df_x_stars = pd.DataFrame(x_stars)
    df_x_stars.to_csv(f"{visual_path}/x_stars.csv")

    df_thetas = pd.DataFrame(thetas)
    df_thetas.to_csv(f"{visual_path}/thetas.csv")

    key_iterate = iterates[0]
    df_x_primals = pd.DataFrame(x_primals[:, key_iterate, :])
    df_x_primals.to_csv(f"{visual_path}/x_primals.csv")
    # df_x_stars.to_csv('x_stars.csv') #, index=False)




def get_x_kalman_from_x_primal(x_primal, T):
    x_state = x_primal[:T * 4]
    # x_control = x_primal[T * 4: T * 6 - 2]
    x1_kalman = x_state[0::4]
    x2_kalman = x_state[1::4]
    x_kalman = jnp.stack([x1_kalman, x2_kalman])
    return x_kalman


def get_full_x(x0, x_w, y, T, Ad, Bd, rho):
    '''
    returns full x variable without redundant constraints
    '''
    nx, nu, no = 4, 2, 2
    x = jnp.zeros(nx)
    x_x = jnp.zeros(T*nx)
    x_v = jnp.zeros(T*no)
    x_s = jnp.zeros(T)
    x_z = jnp.zeros(T)
    x_u = jnp.zeros(T)
    curr_x = x0
    for i in range(T):
        curr_w = x_w[nu*i:nu*(i+1)]
        curr_x = Ad @ curr_x + Bd @ curr_w
        curr_y = y[no*i:no*(i+1)]
        x_pos = jnp.array([curr_x[0], curr_x[1]])
        curr_v = curr_y - x_pos

        curr_s = jnp.linalg.norm(curr_v)
        curr_z = jnp.min(jnp.array([curr_s, rho]))
        curr_u = curr_s - curr_z

        x_x = x_x.at[nx*i:nx*(i+1)].set(curr_x)
        x_v = x_v.at[no*i:no*(i+1)].set(curr_v)
        x_s = x_s.at[i].set(curr_s)
        x_u = x_u.at[i].set(curr_u)
        x_z = x_z.at[i].set(curr_z)

    x = jnp.concatenate([x_x, x_w, x_s, x_v, x_u, x_z])
    return x


def static_canon(T, gamma, dt, mu, rho, B_const, rho_x=1, scale=1):
    """
    variables
    (x_t, w_t, s_t, v_t,  u_t, z_t) in (nx + nu + no + 3)
    (nx,  nu,  1,   no,   no, 1, 1)
    min sum_{i=0}^{T-1} ||w_t||_2^2 + mu (u_t+rho*z_t^2)
        s.t. x_{t+1} = Ax_t + Bw_t  t=0,...,T-2 (dyn)
             y_t = Cx_t + v_t       t=0,...,T-1 (obs)
             u_t + z_t = s_t        t=0,...,T-1 (aux)
             z_t <= rho             t=0,...,T-1 (z ineq)
             u_t >= 0               t=0,...,T-1 (u ineq)
             ||v_t||_2 <= s_t       t=0,...,T-1 (socp)
    (x_0, ..., x_{T-1})
    (y_0, ..., y_{T-1})
    (w_0, ..., w_{T-2})
    (v_0, ..., v_{T-1})
    """
    # nx, nu, no don't change
    nx, nu, no = 4, 2, 2

    # to make indexing easier
    single_len = nx + nu + no + 3
    nvars = single_len * T
    w_start = nx * T
    s_start = w_start + nu * T
    v_start = s_start + T
    u_start = v_start + no * T
    z_start = u_start + T

    assert z_start + T == single_len * T

    # get A, B, C
    Ad, Bd, C = robust_kalman_setup(gamma, dt, B_const)

    # Quadratic objective
    P = np.zeros((single_len, single_len))
    P[nx: nx + nu, nx: nx + nu] = np.eye(nu)
    P[-1, -1] = mu * rho
    # P_sparse = sparse.kron(sparse.eye(T), P)
    P_sparse = 2*sparse.kron(P, sparse.eye(T))

    # Linear objective
    c = np.zeros(single_len * T)
    c[-2 * T: -T] = 2 * mu

    # dyn constraints
    Ax = sparse.kron(sparse.eye(T), -sparse.eye(nx)) + sparse.kron(
        sparse.eye(T, k=-1), Ad
    )
    Ax = Ax[nx:, :]

    # Bw = sparse.kron(sparse.eye(T), Bd)
    Bw = sparse.kron(sparse.eye(T), Bd)
    bw = Bw.todense()
    bw = bw[:(T-1)*nx, :]
    A_dyn = np.zeros(((T-1) * nx, nvars))
    A_dyn[:, :w_start] = Ax.todense()

    A_dyn[:, w_start:s_start] = bw  # Bw.todense()
    b_dyn = np.zeros((T-1) * nx)

    # obs constraints
    Cx = np.kron(np.eye(T), C)

    Iv = np.kron(np.eye(2), np.eye(T))

    A_obs = np.zeros((T * no, nvars))
    A_obs[:, :w_start] = Cx

    A_obs[:, v_start:u_start] = Iv
    # b_obs will be updated by the parameter stack(y_1, ..., y_T)
    b_obs = np.zeros(T * no)

    # aux constraints
    n_aux = T
    A_aux = np.zeros((n_aux, nvars))
    A_aux[:, s_start:v_start] = -np.eye(T)
    A_aux[:, u_start:z_start] = np.eye(T)
    A_aux[:, z_start:] = np.eye(T)
    b_aux = np.zeros(n_aux)

    # z_ineq constraints
    n_z_ineq = T
    A_z_ineq = np.zeros((n_z_ineq, nvars))
    A_z_ineq[:, z_start:] = np.eye(n_z_ineq)
    b_z_ineq = rho * np.ones(n_z_ineq)

    # u_ineq constraints
    n_u_ineq = T
    A_u_ineq = np.zeros((n_u_ineq, nvars))
    A_u_ineq[:, u_start:z_start] = -np.eye(n_u_ineq)
    b_u_ineq = np.zeros(n_u_ineq)

    # socp constraints
    n_socp = T * 3
    A_socp = np.zeros((n_socp, nvars))
    for i in range(T):
        A_socp[3 * i, s_start + i] = -1
        A_socp[
            3 * i + 1: 3 * i + 3, v_start + 2 * i: v_start + 2 * (i + 1)
        ] = -np.eye(2)

    b_socp = np.zeros(n_socp)

    # stack A
    A_sparse = sparse.vstack([A_dyn, A_obs, A_aux, A_z_ineq, A_u_ineq, A_socp])

    # get b
    b = np.hstack([b_dyn, b_obs, b_aux, b_z_ineq, b_u_ineq, b_socp])

    q_array = [3 for i in range(T)]  # np.ones(T) * 3
    # q_array_jax = jnp.array(q_array)
    cones = dict(z=T * (1 + nx + no) - nx, l=n_z_ineq + n_u_ineq, q=q_array)
    cones_array = jnp.array([cones["z"], cones["l"]])
    cones_array = jnp.concatenate([cones_array, jnp.array(cones["q"])])

    # create the matrix M
    m, n = A_sparse.shape
    M = jnp.zeros((n + m, n + m))
    P = P_sparse.todense()
    A = A_sparse.todense()
    P_jax = jnp.array(P)
    A_jax = jnp.array(A)
    M = M.at[:n, :n].set(P_jax)
    M = M.at[:n, n:].set(A_jax.T)
    M = M.at[n:, :n].set(-A_jax)

    # factor for DR splitting
    # algo_factor = jsp.linalg.lu_factor(M + jnp.eye(n + m))
    algo_factor, scale_vec = get_scaled_vec_and_factor(M, rho_x, scale, scale, m, n,
                                                       cones['z'])

    A_sparse = csc_matrix(A)
    P_sparse = csc_matrix(P)

    out_dict = dict(
        M=M,
        algo_factor=algo_factor,
        cones_dict=cones,
        cones_array=cones_array,
        A_sparse=A_sparse,
        P_sparse=P_sparse,
        b=b,
        c=c,
        A_dynamics=Ad,
        Bd=Bd
    )
    return out_dict


def robust_kalman_setup(gamma, dt):
    A = jnp.zeros((4, 4))
    B = jnp.zeros((4, 2))
    C = jnp.zeros((2, 4))

    A = A.at[0, 0].set(1)
    A = A.at[1, 1].set(1)
    A = A.at[0, 2].set((1 - gamma * dt / 2) * dt)
    A = A.at[1, 3].set((1 - gamma * dt / 2) * dt)
    A = A.at[2, 2].set(1 - gamma * dt)
    A = A.at[3, 3].set(1 - gamma * dt)

    B = B.at[0, 0].set(dt**2 / 2)
    B = B.at[1, 1].set(dt**2 / 2)
    B = B.at[2, 0].set(dt)
    B = B.at[3, 1].set(dt)

    C = C.at[0, 0].set(1)
    C = C.at[1, 1].set(1)

    return A, B, C
