#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         plots.py
# Description:  Plotting functions.
#
# Authors:      Francesco Cenni, Giulio Mazzaroppi, Giacomo Pezzolati
# Affiliation:  Alma Mater Studiorum – Università di Bologna
# Date:         December 2025
#########################################################################################

#######################################
# Imports
#######################################

# Python libraries
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Matplotlib settings
plt.rcParams['font.family'] = 'serif'

# Color palette
colors = {
    "blue":    '#1f77b4',
    "orange":  '#ff7f0e',
    "green":   '#2ca02c',
    "red":     '#d62728',
    "purple":  '#9467bd',
    "brown":   '#8c564b',
    "pink":    '#e377c2',
    "grey":    '#7f7f7f',
    "olive":   '#bcbd22',
    "cyan":    '#17becf',
    "black":   '#000000',
}

#######################################
# Functions
#######################################

def show_traj(time: np.ndarray, x_traj: np.ndarray, u_traj: np.ndarray, eq_pair1: tuple, eq_pair2: tuple, title: str) -> None:
    """
    Plots the reference trajectory for a Double Pendulum system.

    Parameters
    ----------
    time : np.ndarray
        Array of time points.
    x_traj : np.ndarray
        State trajectory (shape: [N, 4]).
    u_traj : np.ndarray
        Control inputs (shape: [N, 1]).
    eq_pair1 : tuple
        First equilibrium pair.
    eq_pair2 : tuple
        Second equilibrium pair.
    title : str
        Title for the plots.
    """

    # Labels based on title content
    if "ref" in title.lower():
        lbl_q1, lbl_q2 = r'$q_1^\text{ref}(t)$', r'$q_2^\text{ref}(t)$'
        lbl_dq1, lbl_dq2 = r'$\dot{q}_1^\text{ref}(t)$', r'$\dot{q}_2^\text{ref}(t)$'
        lbl_u = r'$u^\text{ref}(t)$'
    else:
        lbl_q1, lbl_q2 = r'$q_1^*(t)$', r'$q_2^*(t)$'
        lbl_dq1, lbl_dq2 = r'$\dot{q}_1^*(t)$', r'$\dot{q}_2^*(t)$'
        lbl_u = r'$u^*(t)$'
    
    # Metadata
    config = {
        "q1":      (x_traj[0, :], lbl_q1,  r'$\mathrm{[rad]}$',   0),
        "q2":      (x_traj[1, :], lbl_q2,  r'$\mathrm{[rad]}$',   1),
        "q1_dot":  (x_traj[2, :], lbl_dq1, r'$\mathrm{[rad/s]}$', 2),
        "q2_dot":  (x_traj[3, :], lbl_dq2, r'$\mathrm{[rad/s]}$', 3),
        "input":   (u_traj[0, :], lbl_u,   r'$\mathrm{[Nm]}$',    None),
    }

    # Figure setup
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(f"{title}", fontsize=14, fontweight='bold', ha='center', x=0.5)
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])

    # 3. Definition of axes based on desired position
    # ROW 1: q1 and q2q
    ax_q1     = fig.add_subplot(gs[0, 0])
    ax_q2     = fig.add_subplot(gs[0, 1])
    # ROW 2: q1_dot and q2_dot
    ax_q1_dot = fig.add_subplot(gs[1, 0])
    ax_q2_dot = fig.add_subplot(gs[1, 1])
    # ROW 3: Input (takes both columns)
    ax_u      = fig.add_subplot(gs[2, :])

    # Mapping axes to dictionary for plotting loop
    axes_map = {
        "q1": ax_q1, "q2": ax_q2,
        "q1_dot": ax_q1_dot, "q2_dot": ax_q2_dot,
        "input": ax_u
    }

    # Plot trajectory
    for key, ax in axes_map.items():
        data, label, unit, idx = config[key]

        for i, eq_pair in enumerate([eq_pair1, eq_pair2]):
            x_eq, u_eq = eq_pair
            val = u_eq if key == "input" else x_eq[idx]

            ax.axhline(y=val, color=colors["orange"], linestyle='--', alpha=0.7, 
                        label=f'Eq {i+1}: {val:.2f}', linewidth=1.5)
                
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.plot(time, data, linewidth=2, label=label, linestyle='-', color=colors["blue"])
        ax.set_ylabel(unit)
        ax.set_xlabel(r'$t \mathrm{[s]}$')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def trajectory_comparison(
    time: np.ndarray, 
    x_ref: np.ndarray, 
    u_ref: np.ndarray, 
    x_guess: np.ndarray, 
    u_guess: np.ndarray, 
    title: str
) -> None:
    """
    Plots the comparison between reference and guess trajectories.

    Parameters
    ----------
    time : np.ndarray
        Array of time points.
    x_ref : np.ndarray
        Reference state trajectory .
    u_ref : np.ndarray
        Reference control inputs.
    x_guess : np.ndarray
        Guess state trajectory.
    u_guess : np.ndarray
        Guess control inputs.
    title : str
        Title for the plots.
    """
    
    config = {
        "q1": (
            x_ref[0, :], x_guess[0, :], 
            r'$q_{1, \text{ref}}$', r'$q_{1, \text{guess}}$', 
            r'$\mathrm{[rad]}$'
        ),
        "q2": (
            x_ref[1, :], x_guess[1, :], 
            r'$q_{2, \text{ref}}$', r'$q_{2, \text{guess}}$', 
            r'$\mathrm{[rad]}$'
        ),
        "q1_dot": (
            x_ref[2, :], x_guess[2, :], 
            r'$\dot{q}_{1, \text{ref}}$', r'$\dot{q}_{1, \text{guess}}$', 
            r'$\mathrm{[rad/s]}$'
        ),
        "q2_dot": (
            x_ref[3, :], x_guess[3, :], 
            r'$\dot{q}_{2, \text{ref}}$', r'$\dot{q}_{2, \text{guess}}$', 
            r'$\mathrm{[rad/s]}$'
        ),
        "input": (
            u_ref[0, :], u_guess[0, :], 
            r'$u_{\text{ref}}$', r'$u_{\text{guess}}$', 
            r'$\mathrm{[Nm]}$'
        ),
    }

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f"Trajectory Comparison: {title}", fontsize=16, fontweight='bold', ha='center', x=0.5)
    
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Map
    axes_map = {
        "q1":      fig.add_subplot(gs[0, 0]),
        "q2":      fig.add_subplot(gs[0, 1]),
        "q1_dot":  fig.add_subplot(gs[1, 0]),
        "q2_dot":  fig.add_subplot(gs[1, 1]),
        "input":   fig.add_subplot(gs[2, :])
    }

    # Plot
    for key, ax in axes_map.items():
        data_ref, data_guess, label_ref, label_guess, unit = config[key]
        
        # Plot Reference 
        ax.plot(time, data_ref, 
                color=colors['orange'], linestyle='--', linewidth=1.5, 
                label=label_ref)
        
        # Plot Guess 
        ax.plot(time, data_guess, 
                color=colors['blue'], linestyle='-', linewidth=2, 
                label=label_guess)
        
        ax.set_title(f"{label_ref} vs {label_guess}", fontsize=14)
        ax.set_ylabel(unit)
        ax.set_xlabel(r'$t \mathrm{[s]}$')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='best', fontsize='small', frameon=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def equilibrium_manifold(manifold_data: dict, plot_2d: bool = True, plot_3d: bool = True, print_results: bool = True) -> None:
        """
        Plot the equilibrium manifold in 2D projections and 3D.

        Parameters
        ----------
        manifold_data : dict
            Dictionary containing equilibrium data for 'q1_fix', 'q2_fix', and 'u_fix'.
        plot_2d : bool, optional
            Whether to plot 2D projections (default is True).
        plot_3d : bool, optional
            Whether to plot 3D manifold (default is True).
        """
        r1, r2, r3 = manifold_data['q1_fix'], manifold_data['q2_fix'], manifold_data['u_fix']

        if plot_2d:
            fig2d, axs = plt.subplots(1, 3, figsize=(18, 5))
            fig2d.suptitle(f'Equilibria analysis', fontsize=16)

            # q1 vs q2
            if r1.size > 0: axs[0].plot(r1[:, 0], r1[:, 1], 'r-', label='Fix $q_1$')
            if r2.size > 0: axs[0].plot(r2[:, 0], r2[:, 1], 'b--', label='Fix $q_2$')
            if r3.size > 0: axs[0].scatter(r3[:, 0], r3[:, 1], c=r3[:, 2], cmap='viridis', s=30, label='Fix $u$')
            axs[0].set_xlabel('$q_1$ [rad]')
            axs[0].set_ylabel('$q_2$ [rad]')
            axs[0].grid(True)
            axs[0].legend()

            # u vs q1
            if r1.size > 0: 
                axs[1].plot(r1[:, 0], r1[:, 2], 'r-')
                axs[1].set_xlabel('$q_1$ [rad]')
                axs[1].set_ylabel('$u$ [Nm]')
                axs[1].grid(True)

            # u vs q2
            if r2.size > 0:
                axs[2].plot(r2[:, 1], r2[:, 2], 'b--')
                axs[2].set_xlabel('$q_2$ [rad]')
                axs[2].set_ylabel('$u$ [Nm]')
                axs[2].grid(True)

        if plot_3d:
            fig3d = plt.figure(figsize=(10, 8))
            ax = fig3d.add_subplot(111, projection='3d')

            if r1.size > 0:
                ax.plot(r1[:, 0], r1[:, 1], r1[:, 2], 'r-', linewidth=3, label='Fix $q_1$')
            if r2.size > 0:
                ax.plot(r2[:, 0], r2[:, 1], r2[:, 2], 'b--', linewidth=2, label='Fix $q_2$')
            if r3.size > 0:
                sc = ax.scatter(r3[:, 0], r3[:, 1], r3[:, 2], c=r3[:, 2], cmap='viridis', s=30, label='Fix $u$')
                fig3d.colorbar(sc, ax=ax, label='Torque $u$ [Nm]')

            ax.set_xlabel('$q_1$ [rad]')
            ax.set_ylabel('$q_2$ [rad]')
            ax.set_zlabel('$u$ [Nm]')
            ax.set_title('3D Equilibrium Manifold')
            ax.view_init(elev=25, azim=-45)
            ax.legend()
        
        if plot_2d or plot_3d:        
            plt.show()

        # Display results
        if print_results:
            for key in manifold_data:
                print(f"\nEquilibria computed for {key}: {len(manifold_data[key])} state/input pairs found.")
                print("=============== Sampled equilibria (q1, q2, u) ===============")
                print(f"Fixed Variable: {key.replace('Fix ', '')}")
                for i, pair in enumerate(manifold_data[key][:]):
                    print(f"Sample {i+1}: q1 = {pair[0]:.3f},\t q2 = {pair[1]:.3f},\t u = {pair[2]:.3f}")

def armijo_descent(steps: np.ndarray, costs: np.ndarray, JJ: float, descent_arm: float, c: float, stepsizes: list, costs_armijo: list, kk: int) -> None:
    """
    Plots the Armijo descent direction and the costs associated with different stepsizes.

    Parameters
    ----------
    steps : np.ndarray
        Array of stepsizes.
    costs : np.ndarray
        Array of costs corresponding to the stepsizes.
    JJ : float
        Current cost.
    descent_arm : float
        Descent direction value.
    c : float
        Armijo threshold.
    stepsizes : list
        List of stepsizes tested.
    costs_armijo : list
        List of costs associated with the tested stepsizes.
    kk : int
        Current iteration index.
    """
    plt.figure(1)
    
    plt.clf()

    plt.plot(steps, costs, color=colors['green'], label='$J(\\mathbf{u}^k - \\gamma \\cdot d^k)$')
    plt.plot(steps, JJ + descent_arm*steps, color=colors['red'], label='$J(\\mathbf{u}^k) - \\gamma \\cdot \\nabla J(\\mathbf{u}^k)^{\\top} \\cdot d^k$')
    # plt.plot(steps, JJ - descent*steps, color=colors['red'], label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
    plt.plot(steps, JJ + c*descent_arm*steps, color=colors['green'], linestyle='dashed', label='$J(\\mathbf{u}^k) - \\gamma \\cdot c \\cdot \\nabla J(\\mathbf{u}^k)^{\\top} \\cdot d^k$')

    plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize
    
    
    plt.grid()
    plt.xlabel('stepsize ($\\gamma$)')
    plt.title(f'Armijo stepsize selection - Iteration {kk}')
    plt.legend()
    plt.draw()  
    plt.pause(0.01)
    

def descent_direction(descent: np.ndarray, iters: int) -> None:
    """
    Plots the descent direction norm over iterations.

    Parameters
    ----------
    descent : np.ndarray
        Array containing the descent direction norms.
    iters : int
        Number of iterations to plot.
    """
    plt.figure('descent direction', figsize=(10, 6))
    plt.plot(np.arange(iters), descent[:iters], linewidth=2, color=colors['blue'])
    plt.xlabel('Iteration $k$', fontsize=14)
    plt.ylabel(r'$\| \Delta u \| ^2$', fontsize=14)
    plt.title('Descent', fontsize=16)
    plt.yscale('log')
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show(block=False)

def cost(JJ: np.ndarray, iters: int) -> None:
    """
    Plots the cost function over iterations.

    Parameters
    ----------
    JJ : np.ndarray
        Array containing the cost values.
    max_iters : int
        Maximum number of iterations to plot.
    """
    plt.figure('cost', figsize=(10, 6))
    plt.plot(np.arange(iters-1), JJ[:iters-1], linewidth=2, color=colors['orange'])
    plt.xlabel('Iteration $k$', fontsize=14)
    plt.ylabel(r'$J(\mathbf{u}^k)$', fontsize=14)
    plt.title('Cost Function', fontsize=16)
    plt.yscale('log')
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.show(block=False)

def set_animation_figure(ax: plt.Axes, lim_size: float) -> None:
    """
    Sets the figure limits and aspect ratio for animation.
    Parameters
    ----------
    ax : plt.Axes
        The axes to set the limits on.
    lim_size : float
        The limit size for both x and y axes.
    """
    # Plot settings
    ax.set_xlim([-lim_size, lim_size])
    ax.set_ylim([-lim_size, lim_size])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'Double Pendulum Trajectory Comparison') 

def draw_pendulum_configuration(ax: plt.Axes, p1: np.ndarray, p2: np.ndarray, pc1: np.ndarray, pc2: np.ndarray, label: str = None) -> None:
    """
    Draws the double pendulum configuration on the given axes.

    Parameters
    ----------
    ax : plt.Axes
        The axes to draw the pendulum on.
    p1 : np.ndarray
        Position of the first joint.
    p2 : np.ndarray
        Position of the second joint.
    pc1 : np.ndarray
        Position of the center of mass of the first link.
    pc2 : np.ndarray
        Position of the center of mass of the second link.
    label : str
        Label for the pendulum configuration.
    """
    if 'equilibrium' in label.lower():
        link_color = colors['grey']
        alpha = 0.2
    elif 'ref' in label.lower():
        link_color = colors['orange']
        alpha = 0.7
    elif 'current' in label.lower() or 'optimal' in label.lower():
        link_color = colors['blue']
        alpha = 1.0
    ax.plot([0, p1[0]], [0, p1[1]], color=link_color, lw=3, alpha=alpha, label=label) # Link 1
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=link_color, lw=3, alpha=alpha) # Link 2
    ax.scatter([0.0], [0.0], color=colors['black'], s=20, zorder=5, alpha=alpha) # Joint 1
    ax.scatter(p1[0], p1[1], color=colors['black'], s=20, zorder=5, alpha=alpha) # Joint 2
    ax.scatter(pc1[0], pc1[1], color=colors['red'], s=40, zorder=5, alpha=alpha)   # CM1
    ax.scatter(pc2[0], pc2[1], color=colors['red'], s=40, zorder=5, alpha=alpha) # CM2