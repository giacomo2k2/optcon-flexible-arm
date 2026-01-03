#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         animation.py
# Description:  Animation helper functions.
#
# Authors:      Francesco Cenni, Giulio Mazzaroppi, Giacomo Pezzolati
# Affiliation:  Alma Mater Studiorum – Università di Bologna
# Date:         December 2025
#########################################################################################

#######################################
# Imports
#######################################
import matplotlib.pyplot as plt
import time
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from model import DoublePendulumFlexible

plt.rcParams['font.family'] = 'serif'

#######################################
# Functions
#######################################
def animate(model: DoublePendulumFlexible, x_traj: np.ndarray, target_fps: int = 30) -> None:
    """
    Helper function to animate the model along a given trajectory in real-time.
    
    Parameters
    ----------
    model : DoublePendulumFlexible
        The model to animate.
    x_traj : np.ndarray
        Trajectory to follow.
    target_fps : int, optional
        Target frames per second for rendering, by default 30.
    """
    fig, ax = plt.subplots()
    plt.ion()
    plt.show()

    # Calculate time per frame
    render_interval = 1.0 / target_fps
    last_render_time = -render_interval  # Ensure we draw the first frame

    start_time = time.time()
    
    # Loop through trajectory states
    for k, x in enumerate(x_traj.T):
        # The exact time this specific state represents
        current_sim_time = k * model.dt
        
        # --- SYNCHRONIZATION ---
        # Check actual wall-clock time elapsed
        real_time_elapsed = time.time() - start_time
        
        # How long we need to wait to match real-time
        time_to_wait = current_sim_time - real_time_elapsed
        
        if time_to_wait > 0:
            # We are ahead of schedule (computer is fast). Sleep to sync.
            time.sleep(time_to_wait)
        
        # --- RENDERING (Frame Skipping Logic) ---
        # Only draw if enough time has passed since the last draw to match target_fps.
        # AND do not draw if we are significantly lagging behind (time_to_wait < -0.1)
        now = time.time()
        if now - last_render_time >= render_interval:
            
            # If we are lagging significantly (drawing is too slow), 
            # we skip the draw to let the loop catch up to real-time.
            if time_to_wait > -0.05: 
                ax.clear()
                model.draw_frame(x, ax)
                
                # Add timestamp
                ax.text(
                    0.02, 0.95,
                    f"t = {current_sim_time:.2f} s",
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top'
                )
                
                # Legend
                ax.legend(loc='upper right')

                # Update plot
                plt.draw()
                plt.pause(0.001)
                last_render_time = time.time()

                
def animate_comparison(model, x_traj1: np.ndarray, x_traj2: np.ndarray,
                       eq_pair1: tuple, eq_pair2: tuple, dt: float, plot_flag: bool = True, folder=None) -> None:
    """
    Animates the model and saves as a GIF at 30 FPS.
    Uses PillowWriter to avoid 'FFmpeg not found' errors on Windows.
    """
    
    # 1. Calculate Stride for 30 FPS
    # This ensures we don't save every single simulation step, only what's needed for video
    target_fps = 30
    video_dt = 1 / target_fps
    skip_steps = max(1, int(video_dt / dt))
    
    # Create indices for the frames we want to render
    indices = np.arange(0, x_traj1.shape[1], skip_steps)
    
    fig, ax = plt.subplots()
    
    def update(frame_idx):
        ax.clear()
        
        # Extract states at the specific frame index
        x_true = x_traj1[:, frame_idx]
        x_ref = x_traj2[:, frame_idx]
        
        # Draw the model
        model.draw_comparison_frame(ax, x_true, x_ref, eq_pair1[0], eq_pair2[0])
        
        # Add timestamp
        time_sec = frame_idx * dt
        ax.text(
            0.02, 0.95,
            f"t = {time_sec:.2f} s", 
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top'
        )                
        
        ax.legend(loc='upper right')
        
    # 2. Create Animation Object
    anim = FuncAnimation(fig, update, frames=indices)
    
    # 3. Save Animation
    if folder is not None:
        # Changed extension to .gif and writer to PillowWriter
        video_filename = f"{folder}/comparison_animation.gif"
        print(f"\nSaving animation to {video_filename} at {target_fps} FPS...")
        
        # PillowWriter works purely in Python (no external FFmpeg.exe needed)
        writer = PillowWriter(fps=target_fps)
        
        anim.save(video_filename, writer=writer)
        print(f"Animation saved successfully.")

    plt.close(fig)