import time
from functools import wraps
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import jax.numpy as jnp
from ode import force
import re
import os
import imageio # only relevant to create gif out of images


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        """
        Function can be used as a wrapper and returns
        the time required to run another function
        """
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        return result, total_time
    return timeit_wrapper


def create_video(x, dt, N, save=False):
    """
    Create a video of the polecart and the inverted pendulum problem
    """
    def init():
        ax.set_xlim(-1.8, 1.8)  
        ax.set_ylim(-1.5, 1.5)  
        line.set_data([], [])
        time_text.set_text('')
        cart.set_xy([-cart_width / 2, -cart_height / 2]) 
        return line, cart, time_text

    def animate(i):
        x_cart = x[i, 0]       
        theta = x[i, 2]        
        pendulum_length = 1.0   

        ax.set_xlim(x_cart - 1, x_cart + 1)
        
        cart.set_xy([x_cart - cart_width / 2, -cart_height / 2])

        pendulum_x = [x_cart, x_cart + pendulum_length * jnp.sin(jnp.pi - theta)]
        pendulum_y = [0, -pendulum_length * jnp.cos(jnp.pi - theta)] 

        line.set_data(pendulum_x, pendulum_y)
        time_text.set_text(time_template % (i * dt))
        return line, cart, time_text

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2, color='red')

    cart_width = 0.4
    cart_height = 0.2
    cart = plt.Rectangle((-cart_width / 2, -cart_height / 2), cart_width, cart_height, fill=True, color='blue')
    ax.add_patch(cart)

    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    blit = True if save else False
    # To make the video slower plug in higher values for the interval
    ani = FuncAnimation(fig, animate, range(N), interval=50, blit=blit, init_func=init)
    if save:
        # Lower the fps to make the video slower
        ani.save('cart_pole_animation.gif', writer='pillow', fps=15)
    else:
        plt.show()


def plot_train_development(x, t, i, model):
    """
    Valid plot only for solver='RK4'
    returns a plot of the training development i
    """
    if not os.path.exists('video'):
        os.makedirs('video')

    F = [float(model(t_i.reshape(-1,))[0]) for t_i in t]

    fig, ax1 = plt.subplots()

    ax1.plot(t, x[:,2], 'b-', linewidth=2)
    ax1.set_xlabel('Time t in s')
    ax1.set_ylabel(r'Angle $\theta$ in rad', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.axvspan(0.75*jnp.max(t), jnp.max(t), color='gray', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(t, F, 'r-', linewidth=2)
    ax2.set_ylabel(r'Force F in N', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax1.set_xlim([0, 5])
    ax1.set_ylim([-0.5, 5.5])
    ax2.set_ylim([-25, 5])


    plt.title(f"Checkpoint at epoch: {i}")
    plt.savefig(f'video/img_{i}')

    plt.close(fig)


def create_individual_summary(x, t, model=None, nde=False, save=False):

    fig, (ax1, ax3, ax5) = plt.subplots(3, 1, figsize=(10, 12))

    if nde:
        F = [model(t_i.reshape(-1,)) for t_i in t]
        title = r'Results for F(t)=NN(t;$\phi$)'
        save_nde = '_NDE'
    else:
        F = force(t)
        title = r'Results for F(t)=10$\cdot$sin(t)'
        save_nde = ''

    ax1.plot(t, x[:,0], 'b-', linewidth=2)
    ax1.set_ylabel('Position x in m', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.axvspan(0.75*jnp.max(t), jnp.max(t), color='gray', alpha=0.3) 
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(t, x[:,1], 'r-', linewidth=2)
    ax2.set_ylabel(r'Velocity $\dot{x}$ in m/s', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    ax1.set_xlim([0, 5])

    ax3.plot(t, x[:,2], 'b-', linewidth=2)
    ax3.set_ylabel('Angle 'r'$\theta$ in rad', color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    ax3.axvspan(0.75*jnp.max(t), jnp.max(t), color='gray', alpha=0.3) 

    ax4 = ax3.twinx()
    ax4.plot(t, x[:,3], 'r-', linewidth=2)
    ax4.set_ylabel('Angular Velocity 'r'$\dot{\theta}$ in rad/s', color='r')
    ax4.tick_params(axis='y', labelcolor='r')

    ax3.set_xlim([0, 5])

    ax5.plot(t, F, 'b-', linewidth=2)
    ax5.set_ylabel(r'Force F in N', color='b')
    ax5.tick_params(axis='y', labelcolor='b')
    ax5.axvspan(0.75*jnp.max(t), jnp.max(t), color='gray', alpha=0.3) 
    ax5.set_xlim([0, 5])
    ax5.set_xlabel('Time t in s')

    plt.tight_layout()
    if save:
        plt.savefig('Ind_Summary_Plot'+save_nde)
    else:
        plt.show()


def create_summary(t, x, x_model, model, save=False):

    F_model = [model(t_i.reshape(-1,)) for t_i in t]
    title_model = r'Results for F(t)=NN(t;$\phi$)'

    F = force(t)
    title = r'Results for F(t)=10$\cdot$sin(t)'

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))

    ax1, ax3, ax5, ax7, ax9, ax10 = axs.flatten()

    ax1.plot(t, x[:,0], 'b-', linewidth=2)
    ax1.set_ylabel('Position x in m', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title(title)

    ax2 = ax1.twinx()
    ax2.plot(t, x[:,1], 'r-', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='r')

    ax1.set_xlim([0, 5])

    ax5.plot(t, x[:,2], 'b-', linewidth=2)
    ax5.set_ylabel('Angle 'r'$\theta$ in rad', color='b')
    ax5.tick_params(axis='y', labelcolor='b')

    ax6 = ax5.twinx()
    ax6.plot(t, x[:,3], 'r-', linewidth=2)
    ax6.tick_params(axis='y', labelcolor='r')

    ax5.set_xlim([0, 5])

    ax9.plot(t, F, 'b-', linewidth=2)
    ax9.set_ylabel(r'Force F in N', color='b')
    ax9.tick_params(axis='y', labelcolor='b')
    ax9.set_xlim([0, 5])
    ax9.set_xlabel('Time t in s')

    ax3.plot(t, x_model[:,0], 'b-', linewidth=2)
    ax3.tick_params(axis='y', labelcolor='b')
    ax3.axvspan(0.75*jnp.max(t), jnp.max(t), color='gray', alpha=0.3)
    ax3.set_title(title_model)

    ax4 = ax3.twinx()
    ax4.plot(t, x_model[:,1], 'r-', linewidth=2)
    ax4.set_ylabel(r'Velocity $\dot{x}$ in m/s', color='r')
    ax4.tick_params(axis='y', labelcolor='r')

    ax3.set_xlim([0, 5])

    ax7.plot(t, x_model[:,2], 'b-', linewidth=2)
    ax7.tick_params(axis='y', labelcolor='b')
    ax7.axvspan(0.75*jnp.max(t), jnp.max(t), color='gray', alpha=0.3)

    ax8 = ax7.twinx()
    ax8.plot(t, x_model[:,3], 'r-', linewidth=2)
    ax8.set_ylabel('Angular Velocity 'r'$\dot{\theta}$ in rad/s', color='r')
    ax8.tick_params(axis='y', labelcolor='r')

    ax8.set_xlim([0, 5])

    ax10.plot(t, F_model, 'b-', linewidth=2)
    ax10.tick_params(axis='y', labelcolor='b')
    ax10.axvspan(0.75*jnp.max(t), jnp.max(t), color='gray', alpha=0.3)
    ax10.set_xlim([0, 5])
    ax10.set_xlabel('Time t in s')

    plt.tight_layout()
    if save:
        plt.savefig('Summary_Plot')
    else:
        plt.show()


def solver_comparison(x1, x2, t, save=False):
    """
    Plot the comparison between the two solvers used
    - x1:   hardcoded RK4 solver
    - x2:   diffrax solver
    - t:    jnp array with the simulation time
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].plot(t, x1[:,0], 'b-', linewidth=2, label='RK4')
    axs[0].plot(t, x2.ys[0], 'r-', linewidth=2, label='Dopri8')
    axs[0].set_ylabel('Position x in m')
    axs[0].set_xlim([0, 5])
    axs[0].set_xlabel('Time t in s')
    axs[0].legend()

    axs[1].plot(t, x1[:,2], 'b-', linewidth=2, label='RK4')
    axs[1].plot(t, x2.ys[2], 'r-', linewidth=2, label='Dopri8')
    axs[1].set_ylabel(r'Angle $\theta$ in rad')
    axs[1].set_xlim([0, 5])
    axs[1].set_xlabel('Time t in s')
    axs[1].legend()

    plt.tight_layout()

    if save:
        plt.savefig('Solver_Comparison')
    else:
        plt.show()



def create_trainings_gif(folder_path):
    """
    Function returns a gif that was created out of the training evaluations
    In the folder_path there should be images in the form of img_i.png
    """
    if not os.path.exists(folder_path):
        raise FileExistsError('Folder with stored videos does not exist!')

    gif_filepath = 'training_eval.gif'

    def numerical_sort(value):
        """
        Helper function to extract the numeric value from a filename string
        """
        numbers = re.findall(r'\d+', value)
        return int(numbers[0]) if numbers else 0

    images = os.listdir(folder_path)
    sorted_img = sorted(images, key=numerical_sort)

    with imageio.get_writer(gif_filepath, mode='I', loop=0) as writer:
        for filename in sorted_img:
            image_path = os.path.join(folder_path, filename)
            image = imageio.imread(image_path)
            writer.append_data(image)

    print(f'GIF created: {gif_filepath}')