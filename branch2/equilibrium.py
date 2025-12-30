import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.optimize import fsolve

# Import your model
from model import DoublePendulumFlexible, PARAM_SET_1

# --- SOLVER 1: Fix q1, find q2 ---
def find_all_equilibria_q1_fixed(robot, q1_fixed):
    """
    Given q1, find all q2 such that the PASSIVE joint 2 is in equilibrium:
    G2(q1, q2) = 0
    """
    def error_func(q2_guess):
        x_test = np.array([q1_fixed, float(q2_guess), 0.0, 0.0])
        _, _, _, G, _ = robot.dynamic_model(x_test)
        return G[1, 0] # Torque at joint 2 must be 0

    # Search full circle
    guesses = [0.0, np.pi, -np.pi, np.pi/2, -np.pi/2, 0.5, -0.5]
    valid_solutions = []
    seen_vals = []

    for guess in guesses:
        q2_sol, info, ier, msg = fsolve(error_func, guess, full_output=True)
        if ier == 1:
            # Normalize to [-pi, pi]
            val = np.arctan2(np.sin(q2_sol[0]), np.cos(q2_sol[0]))
            
            # Check uniqueness
            is_new = True
            for existing in seen_vals:
                if np.abs(val - existing) < 1e-3:
                    is_new = False
                    break
            
            # Double check residual
            if is_new and abs(error_func(val)) < 1e-4:
                # Calculate required u
                x_final = np.array([q1_fixed, val, 0.0, 0.0])
                _, _, _, G_final, _ = robot.dynamic_model(x_final)
                u_calc = G_final[0, 0]
                
                valid_solutions.append({'q1': q1_fixed, 'q2': val, 'u': u_calc})
                seen_vals.append(val)

    valid_solutions.sort(key=lambda x: x['q2'])
    return valid_solutions

# --- SOLVER 2: Fix q2, find q1 ---
def find_all_equilibria_q2_fixed(robot, q2_fixed):
    """
    Given q2, find all q1 such that the PASSIVE joint 2 is in equilibrium:
    G2(q1, q2) = 0
    """
    def error_func(q1_guess):
        x_test = np.array([float(q1_guess), q2_fixed, 0.0, 0.0])
        _, _, _, G, _ = robot.dynamic_model(x_test)
        return G[1, 0] # Torque at joint 2 must be 0 (it depends on q1 via gravity)

    # Search full circle
    guesses = [0.0, np.pi, -np.pi, np.pi/2, -np.pi/2, 0.5, -0.5]
    valid_solutions = []
    seen_vals = []

    for guess in guesses:
        q1_sol, info, ier, msg = fsolve(error_func, guess, full_output=True)
        if ier == 1:
            val = np.arctan2(np.sin(q1_sol[0]), np.cos(q1_sol[0]))
            
            is_new = True
            for existing in seen_vals:
                if np.abs(val - existing) < 1e-3:
                    is_new = False
                    break
            
            if is_new and abs(error_func(val)) < 1e-4:
                # Calculate required u
                x_final = np.array([val, q2_fixed, 0.0, 0.0])
                _, _, _, G_final, _ = robot.dynamic_model(x_final)
                u_calc = G_final[0, 0]
                
                valid_solutions.append({'q1': val, 'q2': q2_fixed, 'u': u_calc})
                seen_vals.append(val)

    valid_solutions.sort(key=lambda x: x['q1'])
    return valid_solutions













def find_equilibria(model, fix_type, q1=0.0, q2=0.0, right=True, up=True):
    
    def error_func(guess):
        if fix_type == 'q1':
            x_test = np.array([q1, float(guess), 0.0, 0.0])
        else:
            x_test = np.array([float(guess), q2, 0.0, 0.0])
        
        _, _, _, G, _ = model.dynamic_model(x_test)
        return G[1, 0] 


    # Search full circle
    guesses = [0.0, np.pi, -np.pi, np.pi/2, -np.pi/2, 0.5, -0.5]
    valid_solutions = []
    seen_vals = []


    for guess in guesses:
        sol, info, ier, msg = fsolve(error_func, guess, full_output=True)
        if ier == 1:
            # Normalize to [-pi, pi]
            val = np.arctan2(np.sin(sol[0]), np.cos(sol[0]))
            
            # Check uniqueness
            is_new = True
            for existing in seen_vals:
                if np.abs(val - existing) < 1e-3:
                    is_new = False
                    break
            
            # Double check residual
            if is_new and abs(error_func(val)) < 1e-4:
                # Calculate required u
                if fix_type == 'q1':
                    x_final = np.array([q1, val, 0.0, 0.0])
                else:
                    x_final = np.array([val, q2, 0.0, 0.0])

                _, _, _, G_final, _ = model.dynamic_model(x_final)
                u_calc = G_final[0, 0]
                if fix_type == 'q1':
                    valid_solutions.append({'q1': q1, 'q2': val, 'u': u_calc})
                else:
                    valid_solutions.append({'q1': val, 'q2': q2, 'u': u_calc})
                seen_vals.append(val)


    index = 0    
    if fix_type == 'q1':
        for i, sol in enumerate(valid_solutions):
            q2 = sol['q2']
            if up:
                if q2 >= 0:
                    index = i
            if not up:
                if q2 < 0:
                    index = i

    else:            
        for i, sol in enumerate(valid_solutions):
            
            q1 = sol['q1']
            if right:
                if q1 >= 0:
                    index = i
            if not right:
                if q1 < 0:
                    index = i


    print(valid_solutions[index])




# --- MAIN INTERACTIVE SCRIPT ---
def interactive_full():
    robot = DoublePendulumFlexible(**PARAM_SET_1)
    
    fig, ax = plt.subplots(figsize=(9, 9))
    plt.subplots_adjust(left=0.1, bottom=0.35) # More space for controls

    # Plot Setup
    L = robot.l1 + robot.l2
    ax.set_xlim(-1.2*L, 1.2*L)
    ax.set_ylim(-1.2*L, 1.2*L)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title("Double Pendulum Equilibrium Explorer")
    
    # Graphics
    line1, = ax.plot([], [], 'o-', lw=4, color='#333333', label='Link 1')
    line2, = ax.plot([], [], 'o-', lw=4, color='#666666', label='Link 2')
    ghost_line, = ax.plot([], [], 'o--', lw=1, color='red', alpha=0.4, label='Alt Sol')
    
    info_box = ax.text(0.05, 0.95, '', transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                       verticalalignment='top', fontsize=11, family='monospace')

    # State
    mode = 'Fix q1' # or 'Fix q2'
    solutions = []
    sol_idx = 0

    def update_plot(val=None):
        nonlocal solutions, sol_idx
        
        angle_val = slider_angle.val
        
        # 1. Solve based on mode
        if mode == 'Fix q1':
            solutions = find_all_equilibria_q1_fixed(robot, angle_val)
            slider_angle.label.set_text("Set q1")
        else:
            solutions = find_all_equilibria_q2_fixed(robot, angle_val)
            slider_angle.label.set_text("Set q2")

        # 2. Check solutions
        if not solutions:
            line1.set_data([], [])
            line2.set_data([], [])
            info_box.set_text("No equilibrium found\nfor this configuration.")
            fig.canvas.draw_idle()
            return

        if sol_idx >= len(solutions):
            sol_idx = 0
            
        # 3. Draw Active Solution
        curr = solutions[sol_idx]
        x_curr = np.array([curr['q1'], curr['q2'], 0, 0])
        p1, p2, _, _ = robot._get_forward_kinematics(x_curr, com=True)
        
        line1.set_data([0, p1[0]], [0, p1[1]])
        line2.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        
        # 4. Draw Ghost (Alternative) Solution
        if len(solutions) > 1:
            alt_idx = (sol_idx + 1) % len(solutions)
            alt = solutions[alt_idx]
            x_alt = np.array([alt['q1'], alt['q2'], 0, 0])
            p1_a, p2_a, _, _ = robot._get_forward_kinematics(x_alt, com=True)
            ghost_line.set_data([p1_a[0], p2_a[0]], [p1_a[1], p2_a[1]])
        else:
            ghost_line.set_data([], [])
            
        # 5. Update Text
        info_box.set_text(
            f"Mode: {mode}\n"
            f"Solutions found: {len(solutions)}\n"
            f"Displaying: #{sol_idx + 1}\n"
            f"------------------\n"
            f"q1: {curr['q1']:6.3f} rad\n"
            f"q2: {curr['q2']:6.3f} rad\n"
            f"u : {curr['u']:6.3f} Nm"
        )
        
        btn_switch.label.set_text(f"Next Sol ({len(solutions)})")
        fig.canvas.draw_idle()

    # --- Callbacks ---
    def on_slider_change(val):
        update_plot()

    def on_switch_clicked(event):
        nonlocal sol_idx
        if solutions:
            sol_idx = (sol_idx + 1) % len(solutions)
            update_plot()
            
    def on_mode_change(label):
        nonlocal mode, sol_idx
        mode = label
        sol_idx = 0 # Reset index when changing mode
        update_plot()

    # --- WIDGETS ---
    # 1. Slider
    ax_slider = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor='#ffffee')
    slider_angle = Slider(ax_slider, 'Set q1', -np.pi, np.pi, valinit=0.0)
    slider_angle.on_changed(on_slider_change)

    # 2. Mode Selector (Radio Buttons)
    ax_radio = plt.axes([0.05, 0.05, 0.2, 0.2], facecolor='#f0f0f0')
    radio = RadioButtons(ax_radio, ('Fix q1', 'Fix q2'))
    radio.on_clicked(on_mode_change)
    
    # 3. Switch Solution Button
    ax_btn = plt.axes([0.35, 0.05, 0.2, 0.08])
    btn_switch = Button(ax_btn, 'Next Sol')
    btn_switch.on_clicked(on_switch_clicked)

    # 4. Description
    fig.text(0.6, 0.05, 
             "INSTRUCTIONS:\n"
             "1. 'Fix q1': You rotate the base.\n    Solver finds q2 angles.\n"
             "2. 'Fix q2': You bend the elbow.\n    Solver finds base angles q1.",
             fontsize=9, color='#555555')

    update_plot()
    plt.show()

if __name__ == "__main__":
    interactive_full()