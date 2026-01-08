#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         plots.py
# Description:  This module contains all the plotting functions used in the project.
#
# Authors:      Francesco Cenni, Giulio Mazzaroppi, Giacomo Pezzolati
# Affiliation:  Alma Mater Studiorum – Università di Bologna
# Date:         January 2026
#########################################################################################

#######################################
# Imports
#######################################

# Standard libraries
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import numpy as np
import os

# Set font family
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

def show_traj(time: np.ndarray, x_traj: np.ndarray, u_traj: np.ndarray, eq_pair1: tuple, eq_pair2: tuple, title: str, plot_flag: bool = True, folder: str = None) -> None:
    """
    Plots the given state and input trajectory.
    This function highlights two equilibrium pairs on the plots for reference.

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
    plot_flag : bool, optional
        Whether to display the plots. Default is True.
    folder : str, optional
        Folder to save the plots. Default is None.
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

        # Equilibrium lines
        for i, eq_pair in enumerate([eq_pair1, eq_pair2]):
            x_eq, u_eq = eq_pair
            val = u_eq[0] if key == "input" else x_eq[idx]

            ax.axhline(y=val, color=colors["orange"], linestyle='--', alpha=0.7, 
                        label=f'Eq {i+1}: {val:.2f}', linewidth=1.5)
                
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.plot(time, data, linewidth=2, label=label, linestyle='-', color=colors["blue"])
        ax.set_ylabel(unit)
        if key == "input":
            ax.set_xlabel(r'$t \mathrm{[s]}$')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    # Adjust layout
    plt.subplots_adjust(hspace=0.5)
    
    # Show plot if requested
    if plot_flag:
        plt.show()

    # Save plot if folder is provided
    if folder is not None:
        fig.savefig(f"{folder}/reference_trajectory.png", dpi=300, bbox_inches='tight')

def trajectory_comparison(
    time: np.ndarray,
    x_ref: np.ndarray,
    u_ref: np.ndarray,
    x_track: np.ndarray,
    u_track: np.ndarray,
    u_min: float = None,
    u_max: float = None,
    title: str = "",
    plot_flag: bool = True,
    folder: str = None
) -> None:
    """
    Plots the comparison between two trajectories.
    It can work in different scenarios:
    - **Single run**: compares reference vs tracked trajectory.
    - **Multiple runs**: compares reference vs multiple tracked trajectories for different perturbed initial conditions.

    Parameters
    ----------
    time : np.ndarray
        Array of time points.
    x_ref : np.ndarray
        Reference state trajectory (ns, TT).
    u_ref : np.ndarray
        Reference control inputs (ni, TT).
    x_track : np.ndarray
        Tracked state trajectories (ns, TT, runs).
    u_track : np.ndarray
        Tracked control inputs (ns, TT, runs).
    u_min : float, optional
        Minimum control input limit. Default is None.
    u_max : float, optional
        Maximum control input limit. Default is None.
    title : str
        Title for the plots.
    plot_flag : bool, optional
        Whether to display the plots. Default is True.
    folder : str, optional
        Folder to save the plots. Default is None.
    """
    # Ensure x_track and u_track have 3 dimensions
    if x_track.ndim < 3:
        x_track = x_track.reshape(x_track.shape[0], x_track.shape[1], 1)
    
    if u_track.ndim < 3:
        u_track = u_track.reshape(u_track.shape[0], u_track.shape[1], 1)

    # Extract dimensions
    num_runs = x_track.shape[2]
    num_vars = x_track.shape[0] + u_track.shape[0]

    # Suffixes for labels
    if 'ref' in title.lower():
        s_ref, s_track = r'\text{ref}', r'\text{guess}'
    else:
        s_ref, s_track = r'*', r'\text{track}'

    # Figure setup for single or multiple runs
    if num_runs == 1:
        fig = plt.figure(figsize=(10, 12))

        # Grid definition
        gs = gridspec.GridSpec(3, 4, figure=fig)

        # Axes definition
        ax_q1  = fig.add_subplot(gs[0, 0:2])
        ax_q2  = fig.add_subplot(gs[0, 2:4])
        ax_dq1 = fig.add_subplot(gs[1, 0:2])
        ax_dq2 = fig.add_subplot(gs[1, 2:4])
        ax_u   = fig.add_subplot(gs[2, 1:3])

        axes_to_use = [ax_q1, ax_q2, ax_dq1, ax_dq2, ax_u]
    else:
        # Subplots definition
        fig, axes = plt.subplots(num_vars, num_runs, figsize=(4 * num_runs, 15), sharex=True)
        if axes.ndim == 1: axes = axes.reshape(-1, 1)
        
    fig.suptitle(f"Trajectory Comparison: {title}", fontsize=18, fontweight='bold', x=0.5, y=0.98, ha='center')

    # Loop over runs
    for run_idx in range(num_runs):
        # Initial condition for this run
        ic_str = r"$\varepsilon = [0, 0, 0, 0]^\top$"

        # Compute the initial condition error
        if x_ref[:,0].tolist() != x_track[:,0, run_idx].tolist():
            eps = x_ref[:,0] - x_track[:,0, run_idx]
            ic_str = rf"$ \varepsilon = [{eps[0]:.2g}, {eps[1]:.2g}, {eps[2]:.2g}, {eps[3]:.2g}]^\top$"

        # Metadata (ref_data, track_data, label_ref, label_track, unit)
        config = [
            (x_ref[0, :], x_track[0, :, run_idx], rf'$q_1^{s_ref}$', rf'$q_1^{s_track}$', r'$\mathrm{[rad]}$'),
            (x_ref[1, :], x_track[1, :, run_idx], rf'$q_2^{s_ref}$', rf'$q_2^{s_track}$', r'$\mathrm{[rad]}$'),
            (x_ref[2, :], x_track[2, :, run_idx], rf'$\dot{{q}}_1^{s_ref}$', rf'$\dot{{q}}_1^{s_track}$', r'$\mathrm{[rad/s]}$'),
            (x_ref[3, :], x_track[3, :, run_idx], rf'$\dot{{q}}_2^{s_ref}$', rf'$\dot{{q}}_2^{s_track}$', r'$\mathrm{[rad/s]}$'),
            (u_ref[0, :], u_track[0, :, run_idx], rf'$u^{s_ref}$', rf'$u^{s_track}$', r'$\mathrm{[Nm]}$'),
        ]

        # Loop over variables (q1, q2, dq1, dq2, u)
        for var_idx, (d_ref, d_track, l_ref, l_track, unit) in enumerate(config):
            if num_runs == 1:
                ax = axes_to_use[var_idx]
            else:
                ax = axes[var_idx, run_idx]
            
            # Plotting
            ax.plot(time, d_ref, color=colors['orange'], linestyle='--', linewidth=1.5, label=l_ref)
            ax.plot(time, d_track, color=colors['blue'], linestyle='-', linewidth=2, label=l_track)

            # Title: IC only for the first row
            ax.set_title(f"{ic_str}\n{l_ref} vs {l_track}" if var_idx < 2 else f"{l_ref} vs {l_track}", fontsize=9)
            ax.set_ylabel(unit, fontsize=8)
            
            # Title logic: IC only for the first row (q1)
            if num_runs == 1:
                if var_idx == 4: ax.set_xlabel(r'$t \mathrm{[s]}$', fontsize=8)
            elif var_idx == len(config) - 1:
                ax.set_xlabel(r'$t \mathrm{[s]}$', fontsize=8)

            # Input limits visualization (only for input plot)
            if var_idx == num_vars - 1 and u_min is not None and u_max is not None:
                # Augment Ylimits
                y_bot, y_top = ax.get_ylim()
                margin = (y_top - y_bot) * 0.25 # Add a 25% margin
                new_ylim = (y_bot - margin, y_top + margin)

                # Fill areas
                ax.fill_between(time, u_max, new_ylim[1], color='tab:red', alpha=0.1)
                ax.fill_between(time, u_min, u_max, color='tab:green', alpha=0.1)
                ax.fill_between(time, new_ylim[0], u_min, color='tab:red', alpha=0.1)

                # Draw limit lines
                ax.axhline(u_max, color=colors['red'], linestyle='--', linewidth=1.0, label='Input Limits')
                ax.axhline(u_min, color=colors['red'], linestyle='--', linewidth=1.0)

                # Apply new limits
                ax.set_ylim(new_ylim)

            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='best', fontsize='small')
    
    # Adjust layout
    plt.subplots_adjust(hspace=0.5)
    
    # Show plot if requested
    if plot_flag:
        plt.show()

    # Save plot if folder is provided
    if folder is not None:
        # Substitute spaces in title with underscores for filename
        fig.savefig(f"{folder}/trajectory_comparison_{'_'.join(title.split())}.png", dpi=300, bbox_inches='tight')

def tracking_error(
    time: np.ndarray,
    x_ref: np.ndarray,
    u_ref: np.ndarray,
    x_track: np.ndarray,
    u_track: np.ndarray,
    title: str,
    plot_flag: bool = True,
    folder: str = None,
) -> None:
    """
    Plots the tracking error between reference and multiple tracked trajectories for different perturbed intial conditions.

    Parameters
    ----------
    time : np.ndarray
        Array of time points.
    x_ref : np.ndarray
        Reference state trajectory.
    u_ref : np.ndarray
        Reference control inputs.
    x_track : np.ndarray
        Tracked state trajectories (ns, TT, runs).
    u_track : np.ndarray
        Tracked control inputs (ns, TT, runs).
    title : str
        Title for the plots.
    plot_flag : bool, optional
        Whether to display the plots. Default is True.
    folder : str, optional
        Folder to save the plots. Default is None.
    """
    # Extract number of runs
    num_runs = x_track.shape[2]

    # Color map for runs
    cmap = plt.get_cmap('tab10')
    run_colors = [cmap(i) for i in range(num_runs)]

    # Figure setup
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(f"Tracking Error Analysis: {title}", fontsize=16, fontweight='bold', x=0.5, ha='center')

    # 3x2 layout: row1(q1,q2), row2(velocities), row3(u, legend)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

    # Axes mapping
    axes_map = {
        "q1":      fig.add_subplot(gs[0, 0]),
        "q2":      fig.add_subplot(gs[0, 1]),
        "q1_dot":  fig.add_subplot(gs[1, 0]),
        "q2_dot":  fig.add_subplot(gs[1, 1]),
        "input":   fig.add_subplot(gs[2, 0]),
        "legend":  fig.add_subplot(gs[2, 1])
    }

    # Metadata (state_index, label_tex, unit)
    config = {
        "q1":      (0, r'$e_{q_1}$', r'$\mathrm{[rad]}$'),
        "q2":      (1, r'$e_{q_2}$', r'$\mathrm{[rad]}$'),
        "q1_dot":  (2, r'$e_{\dot{q}_1}$', r'$\mathrm{[rad/s]}$'),
        "q2_dot":  (3, r'$e_{\dot{q}_2}$', r'$\mathrm{[rad/s]}$'),
    }

    # Store epsilon labels for legend
    epsilon_labels = []

    # Loop over runs
    for r in range(num_runs):
        # Compute epsilon
        eps = x_track[:, 0, r] - x_ref[:, 0]
        eps_str = fr"Run {r+1}: $\varepsilon$ = [{eps[0]:.2f}, {eps[1]:.2f}, {eps[2]:.2f}, {eps[3]:.2f}]"

        # Add epsilon label
        epsilon_labels.append(eps_str)

        # Plot of the states
        for key, (idx, label, unit) in config.items():
            ax = axes_map[key]
            error = x_ref[idx, :] - x_track[idx, :, r]
            ax.plot(time, error, color=run_colors[r], label=f"Run {r+1}", linewidth=1.5, alpha=0.8)
            ax.set_title(f"Error {label}", fontsize=12)
            ax.set_ylabel(unit)
            ax.grid(True, linestyle=':', alpha=0.6)

        # Plot of the input (special case) in cell [2,0]
        ax_u = axes_map["input"]
        error_u = u_ref[0, :] - u_track[0, :, r]
        ax_u.plot(time, error_u, color=run_colors[r], label=f"Run {r+1}", linewidth=1.5, alpha=0.8)
        ax_u.set_title(r"Error $e_u$", fontsize=12)
        ax_u.set_ylabel(r"$\mathrm{[Nm]}$")
        ax_u.grid(True, linestyle=':', alpha=0.6)

    # Label X axis for the bottom row
    axes_map["q2_dot"].set_xlabel(r'$t \mathrm{[s]}$')
    axes_map["input"].set_xlabel(r'$t \mathrm{[s]}$')

    # Legend cell: show runs with color and epsilon
    ax_leg = axes_map["legend"]
    ax_leg.axis('off')

    # Create dummy handles for the legend with run colors
    for r in range(num_runs):
        ax_leg.plot([], [], color=run_colors[r], label=epsilon_labels[r])
    
    # Plot legend
    ax_leg.legend(loc='center', ncol=1, fontsize=10, frameon=True,
                  title="Perturbed Initial Conditions ($\\epsilon$)")

    # Adjust layout
    plt.subplots_adjust(hspace=0.5)
    
    # Show plot if requested
    if plot_flag:
        plt.show()
    
    # Save plot if folder is provided
    if folder is not None:
        fig.savefig(f"{folder}/tracking_error_analysis.png", dpi=300, bbox_inches='tight')
    
def trajectory_comparison_mid(
    time: np.ndarray,
    x_total: np.ndarray,
    u_total: np.ndarray,
    n_mid: int = 15,
    eq_pair1: tuple = None,
    eq_pair2: tuple = None,
    max_iters: int = 300, 
    title: str = "",
    plot_flag: bool = True,
    folder: str = None,
) -> None:
    """
    Plots the comparison between initial guess, in-mid optimized trajectory, and final optimal trajectory.
    The mid trajectories are plotted at logarithmically spaced iterations to visualize convergence.
    This enhances the understanding of optimization behaviour over iterations, especially in the firsts.

    Parameters
    ----------
    time : np.ndarray
        Array of time points.
    x_total : np.ndarray
        All optimized state trajectory (ns, TT, max_iters).
    u_total : np.ndarray
        All optimized control inputs (ni, TT, max_iters).
    n_mid : int, optional
        Number of in-mid iterations to plot. Default is 15.
    eq_pair1 : tuple
        First equilibrium pair.
    eq_pair2 : tuple
        Second equilibrium pair.
    max_iters : int, optional
        Maximum number of iterations. Default is 300.
    title : str, optional
        Title for the plots. Default is "".
    plot_flag : bool, optional
        Whether to display the plots. Default is True.
    folder : str, optional
        Folder to save the plots. Default is None.
    """
    # Determine indices of mid iterations to plot (logarithmic spacing)
    indices_to_plot = np.geomspace(1, max_iters - 2, n_mid, dtype=int)
    indices_to_plot = np.unique(indices_to_plot)

    # Color palette 
    colors_mid = plt.cm.Blues(np.linspace(0.3, 0.9, len(indices_to_plot)))
    c_init = 'blue'
    c_final = 'orange'

    # Figure setup
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(f"Newton Method Evolution: {title}", fontsize=18, fontweight='bold', ha='center', x=0.5)
    
    # Grid view
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Axes mapping
    axes_keys = ["q1", "q2", "q1_dot", "q2_dot", "input"]
    axes_map = {
        "q1":      fig.add_subplot(gs[0, 0]),
        "q2":      fig.add_subplot(gs[0, 1]),
        "q1_dot":  fig.add_subplot(gs[1, 0]),
        "q2_dot":  fig.add_subplot(gs[1, 1]),
        "input":   fig.add_subplot(gs[2, 0]),
        "legend":  fig.add_subplot(gs[2, 1]) 
    }

    # Metadata (x_index, label_tex, unit)
    config = {
        "q1":     (0, r'$q_1$', r'$\mathrm{[rad]}$'),
        "q2":     (1, r'$q_2$', r'$\mathrm{[rad]}$'),
        "q1_dot": (2, r'$\dot{q}_1$', r'$\mathrm{[rad/s]}$'),
        "q2_dot": (3, r'$\dot{q}_2$', r'$\mathrm{[rad/s]}$'),
        "input":  (0, r'$u$', r'$\mathrm{[Nm]}$')
    }

    # Loop over variables (q1, q2, dq1, dq2, u)
    for key in axes_keys:
        ax = axes_map[key]
        idx, tex, unit = config[key]
        
        # Select data
        data_init = x_total[idx, :, 0] if key != "input" else u_total[idx, :, 0]
        data_total = x_total if key != "input" else u_total

        # 1. Plot Initial Guess
        ax.plot(time, data_init, color=c_init, linestyle='--', linewidth=1.2, label='Initial Guess')

        # 2. Plot Intermediate Trajectories
        for count, i in enumerate(indices_to_plot):
            ax.plot(time, data_total[idx, :, i], color=colors_mid[count], 
                    linewidth=1.0, alpha=0.6, label=f'Iter {i}')
            
        # 3. Plot Final Optimal
        ax.plot(time, data_total[idx, :, max_iters-1], color=c_final, linewidth=2.0, 
                label='Final Optimal')
        
        # 4. Plot Equilibrium Lines
        for i, eq_pair in enumerate([eq_pair1, eq_pair2]):
            x_eq, u_eq = eq_pair
            val = u_eq[0] if key == "input" else x_eq[idx]

            ax.axhline(y=val, color=colors["orange"], linestyle='--', alpha=0.7, 
                        label=f'Eq {i+1}: {val:.2f}', linewidth=1.5)
                
        ax.set_title(f"State {tex}" if key != "input" else "Control Input", fontsize=13)
        ax.set_ylabel(unit)
        if key in ['q2_dot', 'input']:
            ax.set_xlabel(r'$t \mathrm{[s]}$')
        ax.grid(True, linestyle=':', alpha=0.6)

    # Legend handling
    ax_leg = axes_map["legend"]
    ax_leg.axis('off')
    
    # Recover handles and labels from one of the axes
    handles, labels = axes_map["input"].get_legend_handles_labels()
    
    # Position the legend at the center of the sixth cell
    ax_leg.legend(handles, labels, loc='center', fontsize=10, 
                  title="Trajectory Evolution", title_fontsize=12, frameon=True, ncol=3)
    
    # Adjust layout
    plt.subplots_adjust(hspace=2)

    # Show plot if requested
    if plot_flag:
        plt.show()

    # Save plot if folder is provided
    if folder is not None:
        fig.savefig(f"{folder}/trajectory_comparison_mid.png", dpi=300, bbox_inches='tight')

def trajectory_comparison_phases(
    phase_history: list,
    eq_pair1: tuple,
    eq_pair2: tuple,
    title: str,
    plot_flag: bool = True,
    folder: str = None,
) -> None:
    """
    Plots the evolution of the trajectory across dynamic horizon extensions.
    Ensures later iterations are drawn ON TOP of previous ones.
    
    Parameters
    ----------
    phase_history : list
        List of dictionaries containing phase data with keys 'time', 'xx', 'uu', and 'final_idx'.
    eq_pair1 : tuple
        First equilibrium pair.
    eq_pair2 : tuple
        Second equilibrium pair.
    title : str
        Title for the plots.
    plot_flag : bool, optional
        Whether to display the plots. Default is True.
    folder : str, optional
        Folder to save the plots. Default is None.
    """
    # Number of phases
    num_phases = len(phase_history)

    # Color map for phases (From Red to Green)
    cmap = plt.cm.RdYlGn
    
    # Figure setup
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(f"{title}", fontsize=18, fontweight='bold', ha='center', x=0.5)
    
    # Grid view
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Axes mapping
    axes_map = {
        "q1":      fig.add_subplot(gs[0, 0]),
        "q2":      fig.add_subplot(gs[0, 1]),
        "q1_dot":  fig.add_subplot(gs[1, 0]),
        "q2_dot":  fig.add_subplot(gs[1, 1]),
        "input":   fig.add_subplot(gs[2, 0]),
        "legend":  fig.add_subplot(gs[2, 1]) 
    }

    # Metadata (x_index, label_tex, unit)
    config = {
        "q1":     (0, r'$q_1$', r'$\mathrm{[rad]}$'),
        "q2":     (1, r'$q_2$', r'$\mathrm{[rad]}$'),
        "q1_dot": (2, r'$\dot{q}_1$', r'$\mathrm{[rad/s]}$'),
        "q2_dot": (3, r'$\dot{q}_2$', r'$\mathrm{[rad/s]}$'),
        "input":  (0, r'$u$', r'$\mathrm{[Nm]}$')
    }

    # Iterate through phases
    for i, phase_data in enumerate(phase_history):
        # Extract phase data
        t_phase = phase_data['time']
        xx_phase = phase_data['xx']
        uu_phase = phase_data['uu']
        final_idx = phase_data['final_idx']
        
        # Color Progression (Early=Red, Late=Green)
        color_val = i / max(1, (num_phases - 1))
        c_phase = cmap(color_val)

        # Z-ordering, line_width, alpha
        z_ord = 20 - i              
        lw_val = 2.0 
        alpha_val = 1.0           

        # Determine Label for Legend
        lbl = None
        if i == 0: lbl = "Phase 0 (Initial - Top Layer)"
        elif i == num_phases - 1: lbl = f"Phase {i} (Final - Bottom Layer)"
        
        # Plot Logic
        for key in axes_map:
            # Skip legend cell
            if key == "legend": continue
            
            # Select Axis
            ax = axes_map[key]
            idx, _, unit = config[key]
            
            # Select Data
            if key == "input":
                data_final = uu_phase[idx, :, final_idx]
            else:
                data_final = xx_phase[idx, :, final_idx]
            
            # Plot Phase Trajectory
            ax.plot(t_phase, data_final, color=c_phase, 
                    linewidth=lw_val, alpha=alpha_val, zorder=z_ord)
            
            ax.set_ylabel(unit)
            if key in ['q2_dot', 'input']:
                ax.set_xlabel(r'$t \mathrm{[s]}$')
            ax.grid(True, linestyle=':', alpha=0.6)

    for key in axes_map:
        if key == "legend": continue
        
        ax = axes_map[key]
        idx, _, unit = config[key]

        # Plot Equilibrium Lines
        for i, eq_pair in enumerate([eq_pair1, eq_pair2]):
            x_eq, u_eq = eq_pair

            val = u_eq[0] if key == "input" else x_eq[idx]

            ax.axhline(y=val, color=colors['grey'], linestyle='--', alpha=0.7, 
                       label=f'Eq {i+1}: {val:.2f}' if i==0 else f'Eq {i+1}: {val:.2f}', 
                       linewidth=1.5, zorder=1) # zorder basso per stare dietro alle traiettorie
            
            ax.legend()

    # Titles
    axes_map["q1"].set_title(r"State $q_1$", fontsize=13)
    axes_map["q2"].set_title(r"State $q_2$", fontsize=13)
    axes_map["q1_dot"].set_title(r"State $\dot{q}_1$", fontsize=13)
    axes_map["q2_dot"].set_title(r"State $\dot{q}_2$", fontsize=13)
    axes_map["input"].set_title(r"Control Input $u$", fontsize=13)

    # Custom Legend / Colorbar
    ax_leg = axes_map["legend"]
    ax_leg.axis('off')
    
    # Colorbar for phases
    norm = mpl.colors.Normalize(vmin=0, vmax=num_phases-1)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_leg, orientation='horizontal', fraction=0.1)
    cb.set_label('Horizon Extension (Red=Short $\to$ Green=Long)')
    
    # Adjust layout
    plt.subplots_adjust(hspace=0.4)
    
    # Show plot if requested
    if plot_flag:
        plt.show()

    # Save plot if folder is provided
    if folder is not None:
        fig.savefig(f"{folder}/trajectory_comparison_phases.png", dpi=300, bbox_inches='tight')

def equilibrium_manifold(manifold_data: dict, eq_pair1: tuple, eq_pair2: tuple, plot_2d: bool = True, plot_3d: bool = True, plot_flag: bool = True, folder: str = None) -> None:
    """
    Plot the equilibrium manifold using scatter plots with torque-based coloring, highlighting two specific equilibrium pairs.

    Parameters
    ----------
    manifold_data : dict
        Dictionary containing manifold data with keys 'q1_fix' and 'q2_fix'.
    eq_pair1 : tuple
        First equilibrium pair (state and input).
    eq_pair2 : tuple
        Second equilibrium pair (state and input).
    plot_2d : bool, optional
        Whether to plot 2D scatter plots. Default is True.
    plot_3d : bool, optional
        Whether to plot 3D scatter plots. Default is True.
    print_results : bool, optional
        Whether to print results to the console. Default is True.
    plot_flag : bool, optional
        Whether to display the plots. Default is True.
    folder : str, optional
        Folder to save the plots. Default is None.
    """
    # Unpack equilibrium pairs
    x_eq1, u_eq1 = eq_pair1
    x_eq2, u_eq2 = eq_pair2

    # Safely retrieve data
    r1 = manifold_data.get('q1_fix', np.array([]))
    r2 = manifold_data.get('q2_fix', np.array([]))
    
    # Combine datasets for unified plotting
    if r1.size > 0 and r2.size > 0:
        data_combined = np.vstack((r1, r2))
    elif r1.size > 0:
        data_combined = r1
    elif r2.size > 0:
        data_combined = r2
    else:
        print("No data to plot.")
        return

    # Extract columns for clarity
    q1_all = data_combined[:, 0]
    q2_all = data_combined[:, 1]
    u_all  = data_combined[:, 2]

    # 2D Plots
    if plot_2d:
        fig2d, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig2d.suptitle('Equilibrium Manifold Analysis', fontsize=18, fontweight='bold')
        
        # --- Plot 1: Config Space (q1 vs q2) ---
        axs[0].set_title(r'Configuration Space ($q_1$ vs $q_2$)', fontsize=12)
        axs[0].set_xlabel(r'$q_1$ [rad]')
        axs[0].set_ylabel(r'$q_2$ [rad]')
        
        # Reference grid for pi
        for val in [-np.pi, 0, np.pi]:
            axs[0].axvline(val, color='k', linestyle=':', alpha=0.2)
            axs[0].axhline(val, color='k', linestyle=':', alpha=0.2)

        # Manifold scatter
        sc1 = axs[0].scatter(q1_all, q2_all, c=u_all, cmap='coolwarm', s=15, alpha=0.8)

        cbar = plt.colorbar(sc1, ax=axs[0])
        cbar.set_label('Torque $u$ [Nm]')
        axs[0].grid(True, alpha=0.3)

        # --- Plot 2: Torque Profile (u vs q1) ---
        axs[1].set_title(r'Torque Profile vs $q_1$', fontsize=12)
        axs[1].set_xlabel(r'$q_1$ [rad]')
        axs[1].set_ylabel(r'$u$ [Nm]')
        axs[1].axhline(0, color='k', linewidth=1, alpha=0.5) 
        
        # Manifold scatter
        axs[1].scatter(q1_all, u_all, c=u_all, cmap='coolwarm', s=10, alpha=0.6)

        # --- Highlight Eq Pairs (Plot 2) ---
        axs[1].scatter(x_eq1[0], u_eq1, color='red', marker='*', s=200, edgecolors='black', zorder=10)
        axs[1].scatter(x_eq2[0], u_eq2, color='cyan', marker='*', s=200, edgecolors='black', zorder=10)

        axs[1].grid(True, alpha=0.3)

        # --- Plot 3: Torque Profile (u vs q2) ---
        axs[2].set_title(r'Torque Profile vs $q_2$', fontsize=12)
        axs[2].set_xlabel(r'$q_2$ [rad]')
        axs[2].set_ylabel(r'$u$ [Nm]')
        axs[2].axhline(0, color='k', linewidth=1, alpha=0.5)

        # Manifold scatter
        axs[2].scatter(q2_all, u_all, c=u_all, cmap='coolwarm', s=10, alpha=0.6)

        # --- Highlight Eq Pairs (Plot 3) ---
        axs[2].scatter(x_eq1[1], u_eq1, color='red', marker='*', s=200, edgecolors='black', zorder=10)
        axs[2].scatter(x_eq2[1], u_eq2, color='cyan', marker='*', s=200, edgecolors='black', zorder=10)

        axs[2].grid(True, alpha=0.3)

        # --- Highlight Eq Pairs (Plot 1) ---
        axs[0].scatter(x_eq1[0], x_eq1[1], color='red', marker='*', s=200, edgecolors='black', label='Eq Pair 1', zorder=10)
        axs[0].scatter(x_eq2[0], x_eq2[1], color='cyan', marker='*', s=200, edgecolors='black', label='Eq Pair 2', zorder=10)
        axs[0].legend()

        # save figure
        if folder is not None:
            fig2d.savefig(f"{folder}/equilibrium_manifold_2d.png", dpi=300, bbox_inches='tight')
        
    # 3D Plot
    if plot_3d:
        fig3d = plt.figure(figsize=(10, 8))
        ax = fig3d.add_subplot(111, projection='3d')

        # 3D Manifold Scatter
        sc_3d = ax.scatter(q1_all, q2_all, u_all, c=u_all, cmap='coolwarm', s=20, alpha=0.5)

        ax.set_xlabel(r'$q_1$ [rad]', fontsize=11, labelpad=8)
        ax.set_ylabel(r'$q_2$ [rad]', fontsize=11, labelpad=8)
        ax.set_zlabel(r'$u$ [Nm]', fontsize=11, labelpad=8)
        ax.set_title('3D Equilibrium Surface', fontsize=14)
        
        # Zero-torque plane reference
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 2), np.linspace(ylim[0], ylim[1], 2))
        ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray')

        # Add colorbar for 3D plot
        cbar3d = fig3d.colorbar(sc_3d, ax=ax, pad=0.1)
        cbar3d.set_label('Torque $u$ [Nm]')

        # --- Highlight Eq Pairs (3D) ---
        ax.scatter(x_eq1[0], x_eq1[1], u_eq1, color='red', marker='*', s=300, edgecolors='black', label='Eq Pair 1', depthshade=False)
        ax.scatter(x_eq2[0], x_eq2[1], u_eq2, color='cyan', marker='*', s=300, edgecolors='black', label='Eq Pair 2', depthshade=False)
        ax.legend()

        ax.view_init(elev=25, azim=-60)

        # save figure
        if folder is not None:
            fig3d.savefig(f"{folder}/equilibrium_manifold_3d.png", dpi=300, bbox_inches='tight')

    # Show plots if requested 
    if (plot_2d or plot_3d) and plot_flag:
        plt.show()
        
def armijo_descent(steps: np.ndarray, costs: np.ndarray, JJ: float, descent_arm: float, c: float, stepsizes: list, costs_armijo: list, kk: int, plot: bool = True, save: bool = False, folder: str = None) -> None:
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
    plot_flag : bool, optional
        Whether to display the plot. Default is True.
    save : bool, optional
        Whether to save the plot. Default is False.
    folder : str, optional
        Folder to save the plot. Default is None.
    """
    # Setup figure
    plt.figure("Armijo Stepsize Selection")
    plt.clf()

    # Plotting the Armijo conditions
    plt.plot(steps, costs, color=colors['green'], label='$J(\\mathbf{u}^k - \\gamma \\cdot d^k)$')
    plt.plot(steps, JJ + descent_arm*steps, color=colors['red'], label='$J(\\mathbf{u}^k) - \\gamma \\cdot \\nabla J(\\mathbf{u}^k)^{\\top} \\cdot d^k$')
    plt.plot(steps, JJ + c*descent_arm*steps, color=colors['green'], linestyle='dashed', label='$J(\\mathbf{u}^k) - \\gamma \\cdot c \\cdot \\nabla J(\\mathbf{u}^k)^{\\top} \\cdot d^k$')

    plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize
    
    # Plot settings
    plt.grid()
    plt.xlabel('stepsize ($\\gamma$)')
    plt.title(f'Armijo stepsize selection - Iteration {kk}')
    plt.legend()

    # Show the plot
    if plot:
        plt.draw()
        plt.pause(0.001) #when saving it is not needed  
    
    # Save the plot
    if folder is not None and save:
        if not os.path.exists(f"{folder}/armijo"):
            os.makedirs(f"{folder}/armijo")

        plt.savefig(f"{folder}/armijo/armijo_iteration_{kk}.png", dpi=300, bbox_inches='tight')
    
def descent_direction(descent: np.ndarray, iters: int, phase_iters: list = None, plot_flag: bool = True, folder: str = None) -> None:
    """
    Plots the descent direction norm over iterations.

    Parameters
    ----------
    descent : np.ndarray
        Array containing the descent direction norms.
    iters : int
        Number of iterations to plot.
    phase_iters : list, optional
        List of iteration indices where phase changes occur. Default is None.
    plot_flag : bool, optional
        Whether to display the plot. Default is True.
    folder : str, optional
        Folder to save the plot. Default is None.
    """
    # Setup figure
    plt.figure('Descent direction', figsize=(10, 6))
    plt.plot(np.arange(iters), descent[:iters], linewidth=2, color=colors['blue'])
    plt.xlabel('Iteration $k$', fontsize=14)
    plt.ylabel(r'$\| \Delta u \| ^2$', fontsize=14)
    plt.title('Descent', fontsize=16)
    plt.yscale('log')
    plt.grid(alpha=0.4)

    max_val = np.max(descent[:iters])

    # Plot phases changes if provided
    if phase_iters is not None:
        for phase_iter in phase_iters:
            plt.axvline(x=phase_iter, color='grey', linestyle='--', alpha=0.7)
            plt.text(phase_iter + 0.5, max_val / 10, 
                 f'Phase {phase_iters.index(phase_iter)}', 
                 rotation=90, color='grey', alpha=0.7)
    
    # Show plot if requested
    if plot_flag:
        plt.show(block = False)

    # Save plot if folder is provided
    if folder is not None:
        plt.savefig(f"{folder}/descent_direction.png", dpi=300, bbox_inches='tight')

def cost(JJ: np.ndarray, iters: int, phase_iters: list = None, plot_flag: bool = True, folder: str = None) -> None:
    """
    Plots the cost function over iterations.

    Parameters
    ----------
    JJ : np.ndarray
        Array containing the cost values.
    iters : int
        Number of iterations to plot.
    phase_iters : list, optional
        List of iteration indices where phase changes occur. Default is None.
    plot_flag : bool, optional
        Whether to display the plot. Default is True.
    folder : str, optional
        Folder to save the plot. Default is None.
    """
    # Setup figure
    plt.figure('Cost', figsize=(10, 6))
    plt.plot(np.arange(iters-1), JJ[:iters-1], linewidth=2, color=colors['orange'])
    plt.xlabel('Iteration $k$', fontsize=14)
    plt.ylabel(r'$J(\mathbf{u}^k)$', fontsize=14)
    plt.title('Cost Function', fontsize=16)
    plt.yscale('log')
    plt.grid(alpha=0.4)

    # Plot phases changes if provided
    max_val = np.max(JJ[:iters-1])
    if phase_iters is not None:
        for phase_iter in phase_iters:
            plt.axvline(x=phase_iter, color='grey', linestyle='--', alpha=0.7)
            plt.text(phase_iter + 0.5, max_val / 2, 
                 f'Phase {phase_iters.index(phase_iter)}', 
                 rotation=90, color='grey', alpha=0.7)
    
    # Show plot if requested
    if plot_flag:
        plt.show(block=False)

    # Save plot if folder is provided
    if folder is not None:
        plt.savefig(f"{folder}/cost_function.png", dpi=300, bbox_inches='tight')

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
    # Determine link color and alpha based on label
    if 'equilibrium' in label.lower():
        link_color = colors['grey']
        alpha = 0.2
    elif 'ref' in label.lower():
        link_color = colors['orange']
        alpha = 0.7
    elif 'current' in label.lower() or 'optimal' in label.lower():
        link_color = colors['blue']
        alpha = 1.0
    
    # ==== Draw links and joints ====
    ax.plot([0, p1[0]], [0, p1[1]], color=link_color, lw=3, alpha=alpha, label=label) # Link 1
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=link_color, lw=3, alpha=alpha) # Link 2
    ax.scatter([0.0], [0.0], color=colors['black'], s=20, zorder=5, alpha=alpha) # Joint 1
    ax.scatter(p1[0], p1[1], color=colors['black'], s=20, zorder=5, alpha=alpha) # Joint 2
    ax.scatter(pc1[0], pc1[1], color=colors['red'], s=40, zorder=5, alpha=alpha)   # CM1
    ax.scatter(pc2[0], pc2[1], color=colors['red'], s=40, zorder=5, alpha=alpha) # CM2

def gradient_error(grad_errors: np.ndarray, iters: int, plot_flag: bool = True, folder: str = None) -> None:
    """
    Plots the gradient error over iterations.

    Parameters
    ----------
    grad_errors : np.ndarray
        Array containing the gradient error values.
    iters : int
        Number of iterations to plot.
    plot_flag : bool, optional
        Whether to display the plot. Default is True.
    folder : str, optional
        Folder to save the plot. Default is None.
    """
    # Setup figure
    plt.figure('Gradient Error', figsize=(10, 6))
    plt.plot(np.arange(iters), grad_errors[:iters], linewidth=2, color=colors['purple'])
    plt.xlabel('Iteration $k$', fontsize=14)
    plt.ylabel(r'Gradient Error', fontsize=14)
    plt.title('Gradient Error over Iterations', fontsize=16)
    plt.grid(alpha=0.4)
    
    # Show plot if requested
    if plot_flag:
        plt.show()

    # Save plot if folder is provided
    if folder is not None:
        plt.savefig(f"{folder}/gradient_error.png", dpi=300, bbox_inches='tight')

def print_header(title: str, rank: int = 0, size: int = 80, char: str = '=', capitalize: bool = True) -> None:
    """
    Prints a formatted header to the console terminal.

    Parameters
    ----------
    title : str
        The title to display in the header.
    rank : int, optional
        The indentation rank for formatting. Default is 0.
    size : int, optional
        The total size of the header line. Default is 80.
    char : str, optional
        The character to use for the header line. Default is '='.
    """
    # Clean title by replacing underscores with spaces
    clean_title = title.replace('_', ' ')

    # Capitalize title if requested
    if capitalize:
        clean_title = clean_title.title()

    padding_each_side = int((size / 2) * (1 - (0.7**rank)))
    line_length = size - (2 * padding_each_side)
    
    centered_str = " " * padding_each_side + char * line_length + " " * padding_each_side

    print(centered_str)
    print(clean_title.center(size))
    print(centered_str)