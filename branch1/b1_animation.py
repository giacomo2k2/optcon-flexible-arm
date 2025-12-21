import matplotlib.pyplot as plt
import time
import numpy as np

# Matplotlib settings
plt.rcParams['font.family'] = 'serif'


# TODO: consider flexible arm

def animate(model: object, x_traj: np.ndarray, dt: float, target_fps: int = 30) -> None:
    """
    Helper function to animate the model along a given trajectory in real-time.
    
    Args:
        model (object): The model to animate.
        x_traj (np.ndarray): States trajectory to follow [state_dim, time_steps].
        dt (float): Time step.
        target_fps (int, optional): Target frames per second for rendering. Defaults to 30.
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
        current_sim_time = k * dt
        
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
                model.animation(x, ax)
                
                # Add timestamp
                ax.text(
                    0.02, 0.95,
                    f"t = {current_sim_time:.2f} s",
                    transform=ax.transAxes,
                    fontsize=12,
                    verticalalignment='top'
                )
                
                # Update plot
                plt.legend(['link', 'pivot', 'mass'])
                plt.draw()
                plt.pause(0.001)
                
                
                last_render_time = time.time()