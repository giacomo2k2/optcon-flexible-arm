import numpy as np
import matplotlib.pyplot as plt
import b1_costs as cst
from b1_model import DoublePendulumFlexible, PARAM_SET_2
import b1_solver as solver

model = DoublePendulumFlexible(**PARAM_SET_2)


def select_stepsize(stepsize_0,
                    armijo_maxiters,
                    c, beta, 
                    deltau, xx_ref, uu_ref,
                    x0, uu, JJ, descent_arm, QQT, plot = False):

      """
      Computes the stepsize using Armijo's rule.
      input parameters:
            - stepsize_0 : initial stepsize guess,
            - armijo_maxiters : maximum number of iterations for armijo rule,
            - deltau : descending direction for the control action,
            - xx_ref : reference trajectory state,
            - uu_ref : reference trajectory input,
            - x0 : initial state,
            - uu : input at current iteration,
            - JJ : cost at current iteration,
            - descent_arm: armijo descent direction at current iteration,
            - plot: whether or not to show descent plot.

      output parameters:
            - stepsize
      """

      TT = uu.shape[1]

      stepsizes = []  # list of stepsizes
      costs_armijo = []

      stepsize = stepsize_0

      ns = xx_ref.shape[0]
      ni = uu_ref.shape[0]

      for ii in range(armijo_maxiters):

            # Temp solution update

            xx_temp = np.zeros((ns, TT))
            uu_temp = np.zeros((ni, TT))

            xx_temp[:, 0] = x0
            JJ_temp = 0

            # Cost computation
            for tt in range(TT-1):
                  uu_temp[:,tt] = uu[:,tt] + stepsize*deltau[:,tt]
                  xx_temp[:,tt+1] = model.discrete_dynamics(xx_temp[:,tt], uu_temp[:,tt])
                  temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
                  JJ_temp += temp_cost
            temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1],QQT)[0]
            JJ_temp += temp_cost                  

            stepsizes.append(stepsize)      # save the stepsize
            costs_armijo.append(JJ_temp)    # save the cost associated to the stepsize

            if JJ_temp > JJ + c*stepsize*descent_arm:
                  # update the stepsize
                  stepsize = beta*stepsize

            else:
                  print('Armijo stepsize = {:.3e}'.format(stepsize))
                  break
            
            if ii == armijo_maxiters -1:
                  print("WARNING: no stepsize was found with armijo rule!")
            
            
      ############################
      # Descent Plot
      ############################

      if plot:
            steps = np.linspace(0,stepsize_0,int(2e1))
            costs = np.zeros(len(steps))

            for ii in range(len(steps)):
                step = steps[ii]

                # temp solution update
                xx_temp = np.zeros((ns,TT))
                uu_temp = np.zeros((ni,TT))

                xx_temp[:,0] = x0
                JJ_temp = 0

                for tt in range(TT-1):
                    uu_temp[:,tt] = uu[:,tt] + step*deltau[:,tt]
                    xx_temp[:,tt+1] = model.discrete_dynamics(xx_temp[:,tt], uu_temp[:,tt])
                    # temp cost calculation
                    temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
                    JJ_temp += temp_cost
                temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1],QQT)[0]
                JJ_temp += temp_cost
                
                costs[ii] = JJ_temp


            plt.figure(1)
            plt.clf()


            plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - \\gamma \\cdot d^k)$')
            plt.plot(steps, JJ + descent_arm*steps, color='r', label='$J(\\mathbf{u}^k) - \\gamma \\cdot \\nabla J(\\mathbf{u}^k)^{\\top} \\cdot d^k$')
            # plt.plot(steps, JJ - descent*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
            plt.plot(steps, JJ + c*descent_arm*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - \\gamma \\cdot c \\cdot \\nabla J(\\mathbf{u}^k)^{\\top} \\cdot d^k$')

            plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize

            plt.grid()
            plt.xlabel('stepsize')
            plt.legend()
            plt.draw()

            plt.show()

      return stepsize

