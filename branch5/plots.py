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
import matplotlib as mpl
import numpy as np
import os

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

def show_traj(time: np.ndarray, x_traj: np.ndarray, u_traj: np.ndarray, eq_pair1: tuple, eq_pair2: tuple, title: str, plot_flag: bool = True, folder: str = None) -> None:
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

    plt.subplots_adjust(hspace=0.5)
    
    if plot_flag:
        plt.show()

    if folder is not None:
        fig.savefig(f"{folder}/reference_trajectory.png", dpi=300, bbox_inches='tight')

def trajectory_comparison(
    time: np.ndarray,
    x_ref: np.ndarray,
    u_ref: np.ndarray,
    x_track: np.ndarray,
    u_track: np.ndarray,
    title: str,
    plot_flag: bool = True,
    folder: str = None
) -> None:
    """
    Plots the comparison between reference and multiple tracked trajectories for different perturbed intial conditions.

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
    title : str
        Title for the plots.
    plot_flag : bool, optional
        Whether to display the plots. Default is True.
    folder : str, optional
        Folder to save the plots. Default is None.
    """

    # Ensure x_track and u_track have 3 dimensions
    if x_track.ndim < 3:
        #print(f"before: {x_track.ndim}")
        x_track = x_track.reshape(x_track.shape[0], x_track.shape[1], 1)
        #print(f"after: {x_track.ndim}")
    
    if u_track.ndim < 3:
        #print(f"before: {u_track.ndim}")
        u_track = u_track.reshape(u_track.shape[0], u_track.shape[1], 1)
        #print(f"after: {u_track.ndim}")

    # Extract dimensions
    num_runs = x_track.shape[2]
    num_vars = x_track.shape[0] + u_track.shape[0]

    # Suffixes for labels
    if 'ref' in title.lower():
        s_ref, s_track = r'\text{ref}', r'\text{guess}'
    else:
        s_ref, s_track = r'*', r'\text{track}'

    fig, axes = plt.subplots(num_vars, num_runs, figsize=(4 * num_runs, 15), sharex=True)
    fig.suptitle(f"Trajectory Comparison: {title}", fontsize=18, fontweight='bold', x=0.5, y=0.98, ha='center')

    if num_runs == 1:
        axes = axes.reshape(-1, 1)

    for run_idx in range(num_runs):

        ic_str = r"$\varepsilon = [0, 0, 0, 0]^\top$"
        # Initial condition for this run

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

        for var_idx, (d_ref, d_track, l_ref, l_track, unit) in enumerate(config):
            ax = axes[var_idx, run_idx]
            
            # Plotting
            ax.plot(time, d_ref, color=colors['orange'], linestyle='--', linewidth=1.5, label=l_ref)
            ax.plot(time, d_track, color=colors['blue'], linestyle='-', linewidth=2, label=l_track)
            
            # Title logic: IC only for the first row (q1)
            if var_idx == 0:
                ax.set_title(f"{ic_str}\n{l_ref} vs {l_track}", fontsize=8)
            else:
                ax.set_title(f"{l_ref} vs {l_track}", fontsize=8)

            # Labels and style
            ax.set_ylabel(unit, fontsize=8)
            if var_idx == num_vars - 1: # Only last row has X label
                ax.set_xlabel(r'$t \mathrm{[s]}$', fontsize=8)
            
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='best', fontsize='small')
            
    plt.subplots_adjust(hspace=0.5)
    
    if plot_flag:
        plt.show()

    if folder is not None:
        # substitute spaces in title with underscores for filename
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
    num_runs = x_track.shape[2]
    cmap = plt.get_cmap('tab10')
    run_colors = [cmap(i) for i in range(num_runs)]

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(f"Tracking Error Analysis: {title}", fontsize=16, fontweight='bold', x=0.5, ha='center')

    # 3x2 layout: row1(q1,q2), row2(velocities), row3(u, legend)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

    # Assi: 6 celle
    axes_map = {
        "q1":      fig.add_subplot(gs[0, 0]),
        "q2":      fig.add_subplot(gs[0, 1]),
        "q1_dot":  fig.add_subplot(gs[1, 0]),
        "q2_dot":  fig.add_subplot(gs[1, 1]),
        "input":   fig.add_subplot(gs[2, 0]),
        "legend":  fig.add_subplot(gs[2, 1])
    }

    # Configurazione metadati (indice_stato, label_errore, unità)
    # Nota: l'input usa u_ref/u_track, gli altri x_ref/x_track
    config = {
        "q1":      (0, r'$e_{q_1}$', r'$\mathrm{[rad]}$'),
        "q2":      (1, r'$e_{q_2}$', r'$\mathrm{[rad]}$'),
        "q1_dot":  (2, r'$e_{\dot{q}_1}$', r'$\mathrm{[rad/s]}$'),
        "q2_dot":  (3, r'$e_{\dot{q}_2}$', r'$\mathrm{[rad/s]}$'),
    }

    epsilon_labels = []  # Per la legenda per-run

    # Ciclo sulle RUN per raccogliere epsilon e preparare i plot
    for r in range(num_runs):
        # Calcolo epsilon (differenza condizioni iniziali)
        eps = x_track[:, 0, r] - x_ref[:, 0]
        eps_str = fr"Run {r+1}: $\varepsilon$ = [{eps[0]:.2f}, {eps[1]:.2f}, {eps[2]:.2f}, {eps[3]:.2f}]"
        epsilon_labels.append(eps_str)

        # Plot degli stati
        for key, (idx, label, unit) in config.items():
            ax = axes_map[key]
            error = x_ref[idx, :] - x_track[idx, :, r]
            ax.plot(time, error, color=run_colors[r], label=f"Run {r+1}", linewidth=1.5, alpha=0.8)
            ax.set_title(f"Error {label}", fontsize=12)
            ax.set_ylabel(unit)
            ax.grid(True, linestyle=':', alpha=0.6)

        # Plot dell'input (caso speciale) nella cella [2,0]
        ax_u = axes_map["input"]
        error_u = u_ref[0, :] - u_track[0, :, r]
        ax_u.plot(time, error_u, color=run_colors[r], label=f"Run {r+1}", linewidth=1.5, alpha=0.8)
        ax_u.set_title(r"Error $e_u$", fontsize=12)
        ax_u.set_ylabel(r"$\mathrm{[Nm]}$")
        ax_u.grid(True, linestyle=':', alpha=0.6)

    # Label asse X per la riga finale
    axes_map["q2_dot"].set_xlabel(r'$t \mathrm{[s]}$')
    axes_map["input"].set_xlabel(r'$t \mathrm{[s]}$')

    # Cella legenda: mostra le run con colore e epsilon
    ax_leg = axes_map["legend"]
    ax_leg.axis('off')
    # Crea handle fittizi per la legenda con i colori delle run
    for r in range(num_runs):
        ax_leg.plot([], [], color=run_colors[r], label=epsilon_labels[r])
    ax_leg.legend(loc='center', ncol=1, fontsize=10, frameon=True,
                  title="Perturbed Initial Conditions ($\\epsilon$)")

    plt.subplots_adjust(hspace=0.5)
    
    if plot_flag:
        plt.show()
        
    if folder is not None:
        fig.savefig(f"{folder}/tracking_error_analysis.png", dpi=300, bbox_inches='tight')
    

def trajectory_comparison_mid(
    phase_history: list,
    n_mid: int,
    title: str,
    plot_flag: bool = True,
    folder: str = None,
) -> None:
    """
    Plots the evolution of the trajectory across dynamic horizon extensions.
    Ensures later iterations are drawn ON TOP of previous ones.
    """
    
    num_phases = len(phase_history)
    cmap = plt.cm.RdYlGn
    
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(f"{title}", fontsize=18, fontweight='bold', ha='center', x=0.5)
    
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    axes_map = {
        "q1":      fig.add_subplot(gs[0, 0]),
        "q2":      fig.add_subplot(gs[0, 1]),
        "q1_dot":  fig.add_subplot(gs[1, 0]),
        "q2_dot":  fig.add_subplot(gs[1, 1]),
        "input":   fig.add_subplot(gs[2, 0]),
        "legend":  fig.add_subplot(gs[2, 1]) 
    }

    config = {
        "q1":     (0, r'$q_1$', r'$\mathrm{[rad]}$'),
        "q2":     (1, r'$q_2$', r'$\mathrm{[rad]}$'),
        "q1_dot": (2, r'$\dot{q}_1$', r'$\mathrm{[rad/s]}$'),
        "q2_dot": (3, r'$\dot{q}_2$', r'$\mathrm{[rad/s]}$'),
        "input":  (0, r'$u$', r'$\mathrm{[Nm]}$')
    }

    # Iterate through phases
    for i, phase_data in enumerate(phase_history):
        
        t_phase = phase_data['time']
        xx_phase = phase_data['xx']
        uu_phase = phase_data['uu']
        final_idx = phase_data['final_idx']
        
        # 1. Color Progression (Early=Purple -> Late=Yellow)
        color_val = i / max(1, (num_phases - 1))
        c_phase = cmap(color_val)

        z_ord = 20 - i              
        lw_val = 2.0 
        alpha_val = 1.0           

        # Determine Label for Legend
        lbl = None
        if i == 0: lbl = "Phase 0 (Initial - Top Layer)"
        elif i == num_phases - 1: lbl = f"Phase {i} (Final - Bottom Layer)"
        
        # Plot Logic
        for key in axes_map:
            if key == "legend": continue
            
            ax = axes_map[key]
            idx, _, unit = config[key]
            
            # Select Data
            if key == "input":
                data_final = uu_phase[idx, :, final_idx]
            else:
                data_final = xx_phase[idx, :, final_idx]
            
            ax.plot(t_phase, data_final, color=c_phase, 
                    linewidth=lw_val, alpha=alpha_val, zorder=z_ord,
                    label=lbl if key=="input" else None)
            
            ax.set_ylabel(unit)
            if key in ['q2_dot', 'input']:
                ax.set_xlabel(r'$t \mathrm{[s]}$')
            ax.grid(True, linestyle=':', alpha=0.6)

    # Titles
    axes_map["q1"].set_title(r"State $q_1$", fontsize=13)
    axes_map["q2"].set_title(r"State $q_2$", fontsize=13)
    axes_map["q1_dot"].set_title(r"State $\dot{q}_1$", fontsize=13)
    axes_map["q2_dot"].set_title(r"State $\dot{q}_2$", fontsize=13)
    axes_map["input"].set_title(r"Control Input $u$", fontsize=13)

    # Custom Legend / Colorbar
    ax_leg = axes_map["legend"]
    ax_leg.axis('off')
    
    
    norm = mpl.colors.Normalize(vmin=0, vmax=num_phases-1)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_leg, orientation='horizontal', fraction=0.1)
    cb.set_label('Horizon Extension (Dark=Short $\to$ Light=Long)')
    
    plt.subplots_adjust(hspace=0.4)
    
    if plot_flag:
        plt.show()

    if folder is not None:
        fig.savefig(f"{folder}/trajectory_comparison_mid_extended.png", dpi=300, bbox_inches='tight')

def equilibrium_manifold(manifold_data: dict, eq_pair1: tuple, eq_pair2: tuple, plot_2d: bool = True, plot_3d: bool = True, print_results: bool = True, plot_flag: bool = True, folder: str = None) -> None:
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
    # Assuming structure is x_eq = [q1, q2], u_eq = scalar
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
        
    if (plot_2d or plot_3d) and plot_flag:
        plt.show()
        
    # Display results text
    if print_results:
        print(f"\nHighlighting Equilibrium 1: q1={x_eq1[0]:.3f}, q2={x_eq1[1]:.3f}, u={u_eq1:.3f}")
        print(f"Highlighting Equilibrium 2: q1={x_eq2[0]:.3f}, q2={x_eq2[1]:.3f}, u={u_eq2:.3f}")

        for key in manifold_data:
            data = manifold_data[key]
            if len(data) == 0:
                print(f"\nNo equilibria found for sweep: {key}")
                continue
            
            print(f"\nEquilibria computed for {key}: {len(data)} state/input pairs found.")
            limit = 10 
            print(f"Showing first {limit} samples:")
            
            for i, pair in enumerate(data[:limit]):
                print(f"Sample {i+1}: q1 = {pair[0]:.3f},\t q2 = {pair[1]:.3f},\t u = {pair[2]:.3f}")
            
            if len(data) > limit:
                print(f"... and {len(data) - limit} more.")

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
    folder : str, optional
        Folder to save the plot. Default is None.
    """
    plt.figure("Armijo Stepsize Selection")
    
    plt.clf()

    plt.plot(steps, costs, color=colors['green'], label='$J(\\mathbf{u}^k - \\gamma \\cdot d^k)$')
    plt.plot(steps, JJ + descent_arm*steps, color=colors['red'], label='$J(\\mathbf{u}^k) - \\gamma \\cdot \\nabla J(\\mathbf{u}^k)^{\\top} \\cdot d^k$')
    plt.plot(steps, JJ + c*descent_arm*steps, color=colors['green'], linestyle='dashed', label='$J(\\mathbf{u}^k) - \\gamma \\cdot c \\cdot \\nabla J(\\mathbf{u}^k)^{\\top} \\cdot d^k$')

    plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize
    
    
    plt.grid()
    plt.xlabel('stepsize ($\\gamma$)')
    plt.title(f'Armijo stepsize selection - Iteration {kk}')
    plt.legend()

    if plot:
        plt.draw()
        plt.pause(0.001) #when saving it is not needed  
    
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
    plot_flag : bool, optional
        Whether to display the plot. Default is True.
    folder : str, optional
        Folder to save the plot. Default is None.
    """
    plt.figure('Descent direction', figsize=(10, 6))
    plt.plot(np.arange(iters), descent[:iters], linewidth=2, color=colors['blue'])
    plt.xlabel('Iteration $k$', fontsize=14)
    plt.ylabel(r'$\| \Delta u \| ^2$', fontsize=14)
    plt.title('Descent', fontsize=16)
    plt.yscale('log')
    plt.grid(alpha=0.4)


    max_val = np.max(descent[:iters])
    # plot phases changes if provided
    if phase_iters is not None:
        for phase_iter in phase_iters:
            plt.axvline(x=phase_iter, color='grey', linestyle='--', alpha=0.7)
            plt.text(phase_iter + 0.5, max_val / 10, 
                 f'Phase {phase_iters.index(phase_iter)}', 
                 rotation=90, color='grey', alpha=0.7)
    
    
    if plot_flag:
        plt.show(block = False)
    if folder is not None:
        plt.savefig(f"{folder}/descent_direction.png", dpi=300, bbox_inches='tight')

def cost(JJ: np.ndarray, iters: int, phase_iters: list = None, plot_flag: bool = True, folder: str = None) -> None:
    """
    Plots the cost function over iterations.

    Parameters
    ----------
    JJ : np.ndarray
        Array containing the cost values.
    max_iters : int
        Maximum number of iterations to plot.
    plot_flag : bool, optional
        Whether to display the plot. Default is True.
    folder : str, optional
        Folder to save the plot. Default is None.
    """

    
    plt.figure('Cost', figsize=(10, 6))
    plt.plot(np.arange(iters-1), JJ[:iters-1], linewidth=2, color=colors['orange'])
    plt.xlabel('Iteration $k$', fontsize=14)
    plt.ylabel(r'$J(\mathbf{u}^k)$', fontsize=14)
    plt.title('Cost Function', fontsize=16)
    plt.yscale('log')
    plt.grid(alpha=0.4)

    

    # plot phases changes if provided
    max_val = np.max(JJ[:iters-1])
    if phase_iters is not None:
        for phase_iter in phase_iters:
            plt.axvline(x=phase_iter, color='grey', linestyle='--', alpha=0.7)
            plt.text(phase_iter + 0.5, max_val / 2, 
                 f'Phase {phase_iters.index(phase_iter)}', 
                 rotation=90, color='grey', alpha=0.7)
    
    if plot_flag:
        plt.show(block=False)

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
    plt.figure('Gradient Error', figsize=(10, 6))
    plt.plot(np.arange(iters), grad_errors[:iters], linewidth=2, color=colors['purple'])
    plt.xlabel('Iteration $k$', fontsize=14)
    plt.ylabel(r'Gradient Error', fontsize=14)
    plt.title('Gradient Error over Iterations', fontsize=16)
    plt.grid(alpha=0.4)
    
    if plot_flag:
        plt.show()

    if folder is not None:
        plt.savefig(f"{folder}/gradient_error.png", dpi=300, bbox_inches='tight')
