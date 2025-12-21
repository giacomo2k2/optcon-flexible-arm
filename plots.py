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
        "q1":      (x_ref[:, 0], r'$q_1(t)$',           r'$\mathrm{[rad]}$'),
        "q1_dot":  (x_ref[:, 2], r'$\dot{q}_1(t)$',     r'$\mathrm{[rad/s]}$'),
        "q2":      (x_ref[:, 1], r'$q_2(t)$',           r'$\mathrm{[rad]}$'),
        "q2_dot":  (x_ref[:, 3], r'$\dot{q}_2(t)$',     r'$\mathrm{[rad/s]}$'),
        "input":   (u_ref[:, 0], r'$u(t)$',             r'$\mathrm{[Nm]}$'),
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
        
        ax.plot(time, data, linewidth=2, label=label)
        ax.set_ylabel(unit)
        ax.set_xlabel(r'$t \mathrm{[s]}$')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

def plot_equilibrium_manifold(manifold_data: dict, plot_2d: bool = True, plot_3d: bool = True) -> None:
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
