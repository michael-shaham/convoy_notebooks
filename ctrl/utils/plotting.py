import matplotlib.pyplot as plt
import numpy as np
import typing


def plot_traj(T: np.ndarray, traj: typing.List[np.ndarray], 
              labels=typing.List, figsize: typing.Tuple[int] = (6, 6)):
    N = len(traj)
    n = traj[0].shape[0]
    fig, ax = plt.subplots(n, 1, sharex=True, figsize=figsize)
    fig.tight_layout(rect=[0, 0, 1, .96])
    ax[0].set_ylabel("position [m]")
    ax[1].set_ylabel("velocity [m/s]")
    ax[2].set_ylabel(r"accel [m/s$^2$]")
    ax[n-1].set_xlabel("time [s]")
    if n == 4:
        ax[3].set_ylabel(r"input accel [m/s$^2$]")
    fig.suptitle("Platoon trajectory", size=14)
    for i in range(N):
        for j in range(n):
            if i == 0:
                ax[j].plot(T, traj[i][j, :], label=f'{labels[i]}', color='k')
            else:
                ax[j].plot(T, traj[i][j, :], label=f'{labels[i]}')
    for a in ax.flatten():
        a.grid()
    ax[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.0))
    if n == 3:
        ax[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))


def plot_errors(T: np.ndarray, errors: typing.List[np.ndarray], 
                title: typing.AnyStr, labels: typing.List,
                figsize: typing.Tuple[int] = (6, 6)):
    N = len(errors)
    n = errors[0].shape[0]
    fig, ax = plt.subplots(n, 1, sharex=True, figsize=figsize)
    fig.tight_layout(rect=[0, 0, 1, .96])
    ax[0].set_ylabel("spacing error [m]")
    ax[1].set_ylabel("velocity error [m/s]")
    ax[2].set_ylabel(r"acceleration error [m/s$^2$]")
    ax[2].set_xlabel("time [s]")
    fig.suptitle(f"{title}", size=14)
    for i in range(N):
            for j in range(n):
                ax[j].plot(T, errors[i][j, :], label=f'{labels[i]}')
    for a in ax.flatten():
        a.grid()
    ax[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))