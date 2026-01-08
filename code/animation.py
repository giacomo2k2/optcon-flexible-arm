#########################################################################################
# OPTCON-RL 2025/2026 - Course Project #2
# Optimal Control of a Flexible Robotic Link
#
# File:         animation.py
# Description:  This module contains animation helper functions.
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
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Model
from model import DoublePendulumFlexible

# Set font family
plt.rcParams['font.family'] = 'serif'

#######################################
# Functions
#######################################

def animate_comparison(model: DoublePendulumFlexible, x_traj1: np.ndarray, x_traj2: np.ndarray,
                       eq_pair1: tuple, eq_pair2: tuple, dt: float, folder=None) -> None:
    """
    Animates the model and saves as a GIF at 30 FPS if a folder is provided.
    Uses PillowWriter to avoid 'FFmpeg not found' errors on Windows.

    Parameters
    ----------
    model : DoublePendulumFlexible
        The model to animate.
    x_traj1 : np.ndarray
        First trajectory to compare (e.g., actual).
    x_traj2 : np.ndarray
        Second trajectory to compare (e.g., reference).
    eq_pair1 : tuple
        Initial equilibrium points.
    eq_pair2 : tuple
        Final equilibrium points.
    dt : float
        Time step between trajectory points.
    plot_flag : bool, optional
        Whether to save the animation, by default True.
    folder : str, optional
        Folder to save the animation, by default None.
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
    anim = FuncAnimation(fig, update, frames=indices,)
    
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
