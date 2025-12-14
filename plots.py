import matplotlib.pyplot as plt

# Matplotlib settings
plt.rcParams['font.family'] = 'serif'

def plot_trajectory(time, x_ref, u_ref, title):
    plt.figure('Reference trajectory', figsize=(10,6))
    plt.suptitle(title)

    plt.subplot(2, 3, 1)
    plt.title('State 1')
    plt.plot(time, x_ref[:,0], label='$q_1^{ref}$', linewidth=2)
    plt.ylabel('rad')
    plt.grid()

    plt.subplot(2, 3, 2)
    plt.title('State 2')      
    plt.plot(time, x_ref[:,1], label='$q_2^{ref}$', linewidth=2)
    plt.ylabel('rad')
    plt.grid()

    plt.subplot(2,3,3)
    plt.title('State 3')
    plt.plot(time, x_ref[:,2], label='$\dot{q}_1^{ref}$', linewidth=2)
    plt.xlabel('$t$')
    plt.ylabel('rad\s')
    plt.grid()

    plt.subplot(2, 3,4)
    plt.title('State 4')
    plt.plot(time, x_ref[:,3], label='$\dot{q}_2^{ref}$', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('rad\s')
    plt.grid()

    plt.subplot(2,3,5)
    plt.title('Input')
    plt.plot(time, u_ref[:, 0], label='u', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('Nm')
    plt.grid()

    plt.legend()
    plt.show()