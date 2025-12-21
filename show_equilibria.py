from model import DoublePendulumFlexible, PARAM_SET_2
import numpy as np
from plots import plot_equilibrium_manifold



def main():
     # Create model instance
     model = DoublePendulumFlexible(**PARAM_SET_2)
     # Compute equilibrium manifold
     data_dict = model.compute_equilibrium_manifold()
     # Plot equilibrium manifold
     plot_equilibrium_manifold(data_dict)

     #########################################
     # Ex. 1: q1 = 45 deg
     x_1, u_1 = model.find_equilibrium(fix_type='q1', value=np.pi)
     model.plot_equilibrium_point(x_1, u_1)

     # Ex. 2: u = 5.0 Nm
     x_2, u_2 = model.find_equilibrium(fix_type='u', value=5.0)
     model.plot_equilibrium_point(x_2, u_2)

     # Ex. 3: q2 = 45 deg
     x_3, u_3 = model.find_equilibrium(fix_type='q2', value=np.pi/4)
     model.plot_equilibrium_point(x_3, u_3)


if __name__ == "__main__":
    main()