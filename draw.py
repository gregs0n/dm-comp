import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter


def draw1D(
    data: list,
    limits: list,
    plot_name: str,
    yscale="linear",
    show_plot=True,
    ylim=[],
    legends=[],
):

    arg = np.linspace(limits[0], limits[1], data[0].size)
    fig, ax = plt.subplots()
    ax.set_title(plot_name)
    colors = ["blue", "red", "green"]
    for i in range(len(data)):
        lab = legends[i] if legends else f"plot{i+1}"
        ax.plot(arg, data[i], label=lab)
    if not ylim:
        ylim = [min([i.min() for i in data]), max([i.max() for i in data])]
    ax.set_yscale(yscale)
    ax.set_ylim(ymin=ylim[0], ymax=ylim[1])
    # ax.set_xlim(xmin=1.0/limits[0], xmax=1.0/limits[1])
    ax.grid(True)
    ax.legend()
    if show_plot:
        plt.show()
    else:
        fig.savefig(plot_name + ".png")#, dpi=500)
    plt.close()
    del fig, ax


def drawHeatmap(
    data: np.ndarray, limits: list, plot_name: str, show_plot=True, zlim=[]
):
    n = data.shape[0]
    fig, ax = plt.subplots()
    h = (limits[1] - limits[0]) / n
    x = np.arange(limits[0] + h / 2, limits[1], h)
    y = np.arange(limits[0] + h / 2, limits[1], h)
    ## x, h = np.linspace(limits[0], limits[1], n, retstep=True)
    ## y = np.linspace(limits[0], limits[1], n)
    xgrid, ygrid = np.meshgrid(x, y)
    if not zlim:
        zlim = [data.min(), data.max()]
    c = ax.pcolormesh(
        xgrid, ygrid, data, shading="nearest", cmap="RdBu_r", vmin=zlim[0], vmax=zlim[1]
    )  # cmap='hot' | 'afmhot' | 'gist_heat'
    ax.set_title(plot_name)
    ax.axis([limits[0], limits[1], limits[0], limits[1]])
    # ax.plot(xgrid.flat, ygrid.flat, '.', color='black')
    fig.colorbar(c, ax=ax)
    if show_plot:
        plt.show()
    else:
        fig.savefig(plot_name + ".png")#, dpi=1000)
    plt.close()
    del fig, ax


def drawGif(data):
    print(data.shape)

    def my_func(i):
        ax.cla()
        sns.heatmap(
            data[i, ...],
            ax=ax,
            cbar=True,
            cmap="RdBu_r",
            cbar_ax=cbar_ax,
            vmin=300,
            vmax=600,
            square=True,
        )

    grid_kws = {"width_ratios": (0.9, 0.05), "wspace": 0.2}
    fig, (ax, cbar_ax) = plt.subplots(1, 2, gridspec_kw=grid_kws, figsize=(12, 8))
    anim = FuncAnimation(
        fig=fig, func=my_func, frames=data.shape[0], interval=10, blit=False
    )
    writergif = PillowWriter(fps=20)
    # anim.save('plot_00.gif', writer=writergif)
    plt.show()
