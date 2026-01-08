Optimal Control of a Flexible Robotic Link

Course: OPTCON-RL 2025/2026 - Course Project #2 Authors: Francesco Cenni, Giulio Mazzaroppi, Giacomo Pezzolati Affiliation: Alma Mater Studiorum – Università di Bologna

This project implements optimal control strategies for a Flexible Double Pendulum. It features offline trajectory optimization using Newton's method (with optional dynamic horizon extension) and online trajectory tracking using both Linear Quadratic Regulation (LQR) and Model Predictive Control (MPC).

# Requirements
    To run this project, you need Python 3.x and the following libraries:
    numpy, scipy, matplotlib, casadi, control

# How to Use (main.py)
    The entry point of the project is main.py. You do not need to run the other scripts directly; main.py orchestrates the model definition, reference generation, optimization, and tracking.

    1. Configuration
        Open main.py and locate the configuration section at the top of the script. You can modify the following flags to change the behavior of the simulation:
        
        A. Algorithm Selection
            
            GENERATOR = "Newton"            # Options: "Newton" | "Newton-Extended"
            TRACKER = "LQR"                 # Options: "LQR" | "MPC"
            GENERATION_ONLY = False         # If True, stops after trajectory generation (no tracking)

                GENERATOR:
                    Newton: Standard Newton-based optimal control with a fixed time horizon.
                    Newton-Extended: Automatically extends the time horizon if the terminal error is too high (useful for complex maneuvers).

                TRACKER:
                    LQR: Tracks the optimal trajectory using a Linear Time-Varying LQR controller.
                    MPC: Tracks the trajectory using Nonlinear Model Predictive Control (requires CasADi).

        B. Task Selection

            Select which maneuvers the robot should perform.

            # Choose tasks from: 'swing_up', 'elbow_up', 'swing_down', 'complex_maneuver', 
            # 'topside_pump', 'diagonal_swing', 'top_to_elbow_up', 'low_diagonal_hold', 'half_swing_up'
            TASKS = {'swing_up', 'elbow_up'} 

        C. Hyperparameters (Optional)

            Suggested parameters to tune are: 
                
                # Model parameters
                discr_method = "fe"         # fe | rk4  
                dt = 0.01      
                
                # Reference Trajectory generation parameters
                method = 'poly5'             # Trajectory generation method: 'poly3' | 'poly5' | 'stepwise'
                tf = 0.7                        # End time for trajectory segment (relative)

                # Optimal Trajectory generation parameters
                extension_duration = 0.5    # Time extension for Newton-Extended method
                terminal_error_tol = 1e-2    # Tolerance for terminal time condition
                QQt_gen = np.diag([1.0, 1.0, 1.0, 1.0])
                RRt_gen = 1e-3 * np.eye(model.ni)
            
                # Trajectory tracking parameters
                runs = 5                    # Number of runs with different initial perturbations
                perturb = True              # Whether to perturb the initial state
                perturb_scale = 2           # Scale of the initial perturbation
                time_extension = 5          # Time extension after reaching the target state
                T_pred = 100                # Prediction horizon for MPC
                QQt_track = np.diag([1.0, 1.0, 1.0, 1.0])
                RRt_track = 1e-4 * np.eye(model.ni)

    2. Running the Simulation

        Once configured, simply run the script from your terminal:

    3. Output

        Console: Displays the optimization progress (cost, descent direction, step size) and tracking errors.

        Plots/Animations: Results are saved in the report/figs/code/ directory, organized by generator type, tracker type, and task name.

# File Structure & Descriptions

    Here is a detailed description of what each file in the project does.

    1. Core Logic & Entry Point

        main.py The orchestrator. It sets up parameters, initializes the model, runs the trajectory generator, passes the result to the tracker, and manages file saving.

        model.py Defines the DoublePendulumFlexible class.

            Contains the physics engine (Equations of Motion, Mass Matrix, Coriolis, Gravity).

            Handles Continuous and Discrete dynamics (Runge-Kutta 4 or Forward Euler).

            Computes Analytical Jacobians (A,B matrices).

            Includes utilities for Forward Kinematics, Equilibrium finding, and PBH stabilizability tests.

    2. Trajectory Generation (Offline Optimization)

        trajectory_reference_generation.py Generates the reference trajectory. It uses interpolation methods (polynomial or stepwise) to create a rough path between the start and end equilibrium points defined in the tasks dictionary.

        trajectory_generation_newton.py Implements the Newton Method for optimal control. It iteratively improves the reference trajectory to minimize the cost function by solving the Linear Time-Varying (LTV) LQR sub-problem (backward pass) and updating the trajectory (forward pass).

        trajectory_generation_newton_extended.py An advanced version of the Newton generator. If the solver converges but the robot hasn't reached the target (terminal error > tolerance), this script dynamically extends the time horizon and continues optimizing.

        armijo.py Implements the Armijo Rule line search. Used during the Newton forward pass to determine the optimal step size, ensuring global convergence and preventing the solver from taking steps that increase the cost.

    3. Trajectory Tracking (Online Control)

        trajectory_tracking_LQR.py Implements LTV-LQR for tracking. It linearizes the system along the optimal trajectory (computed previously) to calculate time-varying feedback gains K(t) that keep the robot on the path despite disturbances.

        trajectory_tracking_MPC.py Implements Nonlinear MPC using CasADi. It solves a finite-horizon optimization problem at every time step, allowing the controller to handle input constraints (umin​,umax​) explicitly, which LQR cannot do.

    4. Mathematical Solvers & Utilities

        solver.py A library of numerical methods:

            solver: Integrates dynamics (rk4/fe).

            jacobian_discrete_analytic: Computes discrete-time Jacobians.

            ltv_affine_LQR_step: Solves the Riccati recursion for the Newton method (generation).

            ltv_LQR: Solves the Riccati equation for the tracking controller.

            ltv_mpc: Sets up and solves the CasADi optimization problem.

        costs.py Defines the cost functions (Lagrangians):

            stagecost: Running cost (xTQx+uTRu).

            termcost: Final cost at time T.

            Calculates analytic derivatives (gradients and Hessians) required for the Newton method.

    5. Visualization (Assumed)

        plots.py & animation.py:  These files handle the graphical output. They generate 2D plots of states/inputs, convergence graphs, and animations of the robotic link moving.