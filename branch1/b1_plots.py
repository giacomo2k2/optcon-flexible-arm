import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Matplotlib settings
plt.rcParams['font.family'] = 'serif'

def plot_reference_traj(time: np.ndarray, x_ref: np.ndarray, u_ref: np.ndarray, title: str) -> None:
    """
    Plots the reference trajectory for a Double Pendulum system.

    Parameters
    ----------
    time : np.ndarray
        Array of time points.
    x_ref : np.ndarray
        Reference state trajectory (shape: [N, 4]).
    u_ref : np.ndarray
        Reference control inputs (shape: [N, 1]).
    title : str
        Title for the plots.
    """
    
    # Metadata
    config = {
        "q1":      (x_ref[0, :], r'$q_{1(t),\text{ref}}$',           r'$\mathrm{[rad]}$'),
        "q1_dot":  (x_ref[2, :], r'$\dot{q}_{1(t),\text{ref}}$',     r'$\mathrm{[rad/s]}$'),
        "q2":      (x_ref[1, :], r'$q_{2(t),\text{ref}}$',           r'$\mathrm{[rad]}$'),
        "q2_dot":  (x_ref[3, :], r'$\dot{q}_{2(t),\text{ref}}$',     r'$\mathrm{[rad/s]}$'),
        "input":   (u_ref[0, :], r'$u_{(t),\text{ref}}$',             r'$\mathrm{[Nm]}$'),
    }

    # Figure setup
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(f"Reference trajectory: {title}", fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])

    # 3. Definition of axes based on desired position
    # ROW 1: q1 and q2
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

    # Plot loop
    for key, ax in axes_map.items():
        data, label, unit = config[key]
        
        ax.plot(time, data, linewidth=2, label=label, linestyle='--')
        ax.set_ylabel(unit)
        ax.set_xlabel(r'$t \mathrm{[s]}$')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def plot_trajectory_comparison(
    time: np.ndarray, 
    x_ref: np.ndarray, 
    u_ref: np.ndarray, 
    x_guess: np.ndarray, 
    u_guess: np.ndarray, 
    title: str
) -> None:
    """
    Plots the comparison between reference and guess trajectories.
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
    fig.suptitle(f"Trajectory Comparison: {title}", fontsize=16, fontweight='bold')
    
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Mapping degli assi
    axes_map = {
        "q1":      fig.add_subplot(gs[0, 0]),
        "q2":      fig.add_subplot(gs[0, 1]),
        "q1_dot":  fig.add_subplot(gs[1, 0]),
        "q2_dot":  fig.add_subplot(gs[1, 1]),
        "input":   fig.add_subplot(gs[2, :])
    }

    # Loop di plotting
    for key, ax in axes_map.items():
        # Unpacking completo dal dizionario
        data_ref, data_guess, label_ref, label_guess, unit = config[key]
        
        # Plot Reference (Tratteggio)
        ax.plot(time, data_ref, 
                color='red', linestyle='--', linewidth=1.5, 
                label=label_ref)
        
        # Plot Guess (Continuo)
        ax.plot(time, data_guess, 
                color='royalblue', linestyle='-', linewidth=2, 
                label=label_guess)
        
        # Formattazione assi
        ax.set_ylabel(unit)
        ax.set_xlabel(r'$t \mathrm{[s]}$')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # La legenda usa ora le label estratte direttamente dal dizionario
        ax.legend(loc='best', fontsize='small', frameon=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_equilibrium_manifold(manifold_data: dict, plot_2d: bool = True, plot_3d: bool = True, print_results: bool = True) -> None:
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
