import matplotlib.pyplot as plt

# Matplotlib settings
plt.rcParams['font.family'] = 'serif'

def plot_trajectory(time, x_ref, u_ref, title):
    plt.figure('Reference trajectory', figsize=(10,6))
    plt.suptitle(title)

    plt.subplot(2, 3, 1)
    plt.title('q1_ref')
    plt.plot(time, x_ref[:,0], label='q1_ref', linewidth=2)
    plt.ylabel('rad')
    plt.grid()

    plt.subplot(2, 3, 2)
    plt.title('q2_ref')      
    plt.plot(time, x_ref[:,1], label='q2_ref', linewidth=2)
    plt.ylabel('rad')
    plt.grid()

    plt.subplot(2,3,3)
    plt.title('q1_dot_ref')
    plt.plot(time, x_ref[:,2], label='q1_dot_ref', linewidth=2)
    plt.xlabel('$t$')
    plt.ylabel('rad\s')
    plt.grid()

    plt.subplot(2, 3,4)
    plt.title('q2_dot_ref')
    plt.plot(time, x_ref[:,3], label='q2_dot_ref', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('rad\s')
    plt.grid()

    plt.subplot(2,3,5)
    plt.title('Input')
    plt.plot(time, u_ref[:, 0], label='u_ref', linewidth=2)
    plt.xlabel('t')
    plt.ylabel('Nm')
    plt.grid()

    plt.legend()
    plt.show()