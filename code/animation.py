#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         animation.py
# Description:  This module contains animation helper functions.
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
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Custom libraries
#  - Model
from model import DoublePendulumFlexible

# Set font family
plt.rcParams['font.family'] = 'serif'

#######################################
# Functions
#######################################

def animate_comparison(model: DoublePendulumFlexible, x_traj1: np.ndarray, x_traj2: np.ndarray,
                       eq_pair1: tuple, eq_pair2: tuple, dt: float, plot_flag:bool = False, folder=None) -> None:
    """
    creates an animation comparing two trajectories of the double pendulum model.
    Parameters:
        - model: DoublePendulumFlexible
        - x_traj1: First trajectory to compare (shape: state_dim x time_steps).
        - x_traj2: Second trajectory to compare (shape: state_dim x time_steps).
        - eq_pair1: Tuple containing the equilibrium point and label for the first trajectory.
        - eq_pair2: Tuple containing the equilibrium point and label for the second trajectory.
        - dt: Time step between states in the trajectories.
        - plot_flag: If True, display the animation after creation.
        - folder: If provided, the folder path where to save the animation as a GIF.

    Returns:
        None            
    """
    
    # --- SPEED CONTROLS ---
    target_fps = 30
    speed_factor = 1.0  
    
    # Calculate Stride
    video_dt = 1 / target_fps
    
    skip_steps = max(1, int((video_dt / dt) * speed_factor))
    
    # Create indices for the frames we want to render
    indices = np.arange(0, x_traj1.shape[1], skip_steps)

    fig, ax = plt.subplots()
    
    def update(frame_idx):
        # Clear axis for new frame drawing
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

        
        # If this is the last frame in our list, close the plot
        if frame_idx == indices[-1] and plot_flag:
            plt.close(fig)
        
    # 2. Create Animation Object
    interval_ms = 1000 / target_fps
    anim = FuncAnimation(fig, update, frames=indices, interval=interval_ms, repeat=False)
    
    # 3. Save Animation
    if folder is not None:
        video_filename = f"{folder}/comparison_animation.gif"
        print(f"\nSaving animation to {video_filename} at {target_fps} FPS...")
        
        # Use PillowWriter for GIF
        writer = PillowWriter(fps=target_fps)
        anim.save(video_filename, writer=writer)
        print(f"Animation saved successfully.")
        
        # Close figure after saving to prevent it from popping up if plot_flag is False
        plt.close(fig)

    # Only show if requested (and if we haven't already closed it after saving)
    if plot_flag and plt.fignum_exists(fig.number):
        plt.show()
        plt.close(fig)