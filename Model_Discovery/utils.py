"""
file contains functions to load the data and create plots as well 
as videos.
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

def load_data(fname):
    """
    returns the data stored in a dictionary given a filename
    1_npz, 2_npz, 3_npz
    """
    # Add folder Data to current path
    path = os.getcwd()
    data_path = os.path.join(path, fname)

    # Load data
    data = np.load(data_path)

    if int(fname[0]) == 3:
        pde = {'u':data['u'], 'v':data['v'], 'x':data['x'], 'y':data['y'], 't':data['t']}
    else:
        pde = {'u':data['u'], 'x':data['x'], 't':data['t']}
    # return library with relevant data stored!
    return pde


def create_plot(pde, dudi, dudii=None, dudiii=None, dudiiii=None, 
                derivative='time', save=False, name=None):
    """
    Side by side plot for 1_npz and 2_npz of u(x,t) and dudt
    """
    if dudii is None and dudiii is None and dudiiii is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot u(x, t)
        c1 = ax1.pcolormesh(pde['t'], pde['x'], pde['u'])
        ax1.set_xlabel('t', fontsize=16)
        ax1.set_ylabel('x', fontsize=16)
        ax1.set_title(r'$u(x, t)$', fontsize=16)
        fig.colorbar(c1, ax=ax1)

        # Plot \dot{u}(x, t)
        c2 = ax2.pcolormesh(pde['t'], pde['x'], dudi)
        ax2.set_xlabel('t', fontsize=16)
        ax2.set_ylabel('x', fontsize=16)
        if derivative == 'time':
            ax2.set_title(r'$\dot{u}(x, t)$', fontsize=16)
        elif derivative == 'space':
            ax2.set_title(r'$u_{x}(x, t)$', fontsize=16)
        fig.colorbar(c2, ax=ax2)
        plt.tight_layout()
    
    else:
        assert dudii is not None and dudiii is not None and dudiiii is not None, \
            "Pass up to 4th order derivatives as input"
        assert derivative == 'space', \
        "This plot is only valid for the spacial x derivative"

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 4))
        # Plot u(x, t)
        c1 = ax1.pcolormesh(pde['t'], pde['x'], dudi)
        ax1.set_xlabel('t', fontsize=16)
        ax1.set_ylabel('x', fontsize=16)
        ax1.set_title(r'$u_{x}(x, t)$', fontsize=16)
        fig.colorbar(c1, ax=ax1)

        # Plot \dot{u}(x, t)
        c2 = ax2.pcolormesh(pde['t'], pde['x'], dudii)
        ax2.set_xlabel('t', fontsize=16)
        ax2.set_ylabel('x', fontsize=16)
        ax2.set_title(r'$u_{xx}(x, t)$', fontsize=16)
        fig.colorbar(c2, ax=ax2)

        c3 = ax3.pcolormesh(pde['t'], pde['x'], dudiii)
        ax3.set_xlabel('t', fontsize=16)
        ax3.set_ylabel('x', fontsize=16)
        ax3.set_title(r'$u_{xxx}(x, t)$', fontsize=16)
        fig.colorbar(c3, ax=ax3)

        c4 = ax4.pcolormesh(pde['t'], pde['x'], dudiiii)
        ax4.set_xlabel('t', fontsize=16)
        ax4.set_ylabel('x', fontsize=16)
        ax4.set_title(r'$u_{xxxx}(x, t)$', fontsize=16)
        fig.colorbar(c4, ax=ax4)
        plt.tight_layout()

    if save == False:
        plt.show()
    else:
        plt.savefig(name)


def create_3d_plot(pde, save=False, name=None):
    """
    Creates a 3D plot for 1_npz and 2_npz
    """
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(pde['x'], pde['t'], pde['u'], rstride=1, cstride=1, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
    ax.set_xlabel('x', fontsize = 16)
    ax.set_ylabel('t', fontsize = 16)
    ax.set_zlabel('u', fontsize = 16)
    if save == True:
        plt.savefig(name)
    else:
        plt.show()


def create_video(pde, save=False):
    """
    Side by side video of u(x,y,t) and v(x,y,t) for 3_npz
    """
    assert len(pde.keys()) > 3, "Video can only be created for the third PDE problem!"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    u = pde['u']
    v = pde['v']
    x = pde['x']
    y = pde['y']
    t = pde['t']

    def animate(i):
        ax1.clear()
        ax2.clear()
        
        ax1.pcolormesh(x[:, :, i], y[:, :, i], u[:, :, i])
        ax1.set_title(r'$u(x, y, t)$', fontsize=16)
        ax1.set_xlabel('x', fontsize=16)
        ax1.set_ylabel('y', fontsize=16)
        
        ax2.pcolormesh(x[:, :, i], y[:, :, i], v[:, :, i])
        ax2.set_title(r'$v(x, y, t)$', fontsize=16)
        ax2.set_xlabel('x', fontsize=16)
        ax2.set_ylabel('y', fontsize=16)

    ani = animation.FuncAnimation(fig, animate, frames=u.shape[2], interval=10)

    # Save the animation as a video
    if save == True:
        ani.save('../Output/pde3.gif', writer='ffmpeg', fps=30)
    else:
        plt.show()



def plot_residual(gt_1, pred_1, res_1, gt_2, pred_2, res_2,
                  gt_3, pred_3, res_3, save=False, name=None):
    """
    To visualize the resulting predictions
    """
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    # Plot u(x, t) for gt_1
    c1 = axs[0, 0].pcolormesh(gt_1.pde['t'], gt_1.pde['x'], gt_1.u_t)
    axs[0, 0].set_ylabel('x', fontsize=16)
    axs[0, 0].set_title(r'$Ground Truth u_t$', fontsize=16)
    fig.colorbar(c1, ax=axs[0, 0])

    # Plot predicted u(x, t) for gt_1
    c2 = axs[0, 1].pcolormesh(gt_1.pde['t'], gt_1.pde['x'], pred_1)
    axs[0, 1].set_title(r'$Regression u_t$', fontsize=16)
    fig.colorbar(c2, ax=axs[0, 1])

    # Plot residual u(x, t) for gt_1
    c3 = axs[0, 2].pcolormesh(gt_1.pde['t'], gt_1.pde['x'], res_1)
    axs[0, 2].set_title(r'$Residual u_t$', fontsize=16)
    fig.colorbar(c3, ax=axs[0, 2])

    # Plot u(x, t) for gt_2
    c4 = axs[1, 0].pcolormesh(gt_2.pde['t'], gt_2.pde['x'], gt_2.u_t)
    axs[1, 0].set_ylabel('x', fontsize=16)
    axs[1, 0].set_title(r'$Ground Truth u_t$', fontsize=16)
    fig.colorbar(c4, ax=axs[1, 0])

    # Plot predicted u(x, t) for gt_2
    c5 = axs[1, 1].pcolormesh(gt_2.pde['t'], gt_2.pde['x'], pred_2)
    axs[1, 1].set_title(r'$Regression u_t$', fontsize=16)
    fig.colorbar(c5, ax=axs[1, 1])

    # Plot residual u(x, t) for gt_2
    c6 = axs[1, 2].pcolormesh(gt_2.pde['t'], gt_2.pde['x'], res_2)
    axs[1, 2].set_title(r'$Residual u_t$', fontsize=16)
    fig.colorbar(c6, ax=axs[1, 2])

    # Plot u(x, t) for gt_3
    c7 = axs[2, 0].pcolormesh(gt_3.pde['t'], gt_3.pde['x'], gt_3.u_t)
    axs[2, 0].set_xlabel('t', fontsize=16)
    axs[2, 0].set_ylabel('x', fontsize=16)
    axs[2, 0].set_title(r'$Ground Truth u_t$', fontsize=16)
    fig.colorbar(c7, ax=axs[2, 0])

    # Plot predicted u(x, t) for gt_3
    c8 = axs[2, 1].pcolormesh(gt_3.pde['t'], gt_3.pde['x'], pred_3)
    axs[2, 1].set_xlabel('t', fontsize=16)
    axs[2, 1].set_title(r'$Regression u_t$', fontsize=16)
    fig.colorbar(c8, ax=axs[2, 1])

    # Plot residual u(x, t) for gt_3
    c9 = axs[2, 2].pcolormesh(gt_3.pde['t'], gt_3.pde['x'], res_3)
    axs[2, 2].set_xlabel('t', fontsize=16)
    axs[2, 2].set_title(r'$Residual u_t$', fontsize=16)
    fig.colorbar(c9, ax=axs[2, 2])

    plt.tight_layout()
    if save:
        plt.savefig('Residual_Plot.png')
    else:
        plt.show()