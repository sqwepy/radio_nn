#! /usr/bin/env python
"""
Plot radio pulses and fluence maps.
"""
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import scipy.interpolate as intp
from radiotools.analyses import energy_fluence

fnt_size = 20
plt.rc("font", size=fnt_size)  # controls default text size
plt.rc("axes", titlesize=fnt_size)  # fontsize of the title
plt.rc("axes", labelsize=fnt_size)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=fnt_size)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=fnt_size)  # fontsize of the y tick labels
plt.rc("legend", fontsize=fnt_size)


def plot_scatter_interactive(real, sim):
    plots = []
    for i in range(2):
        fig = go.Figure(
            data=go.Scatter(
                x=np.arange(real.shape[0]),
                y=real[:, 0, i],
                mode="lines",
                name=f"real: - {i}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(sim.shape[0]),
                y=sim[:, i],
                mode="lines",
                name=f"sim: - {i}",
            )
        )
        plots.append(fig)
    return plots

def plot_hist(data):
    import plotly.express as px
    fig = px.histogram(data)
    return fig

def plot_pulses_interactive(real, sim, antenna=7):
    plots = []
    for i in range(2):
        fig = go.Figure(
            data=go.Scatter(
                x=np.arange(256),
                y=real[antenna, :, i],
                mode="lines",
                name=f"real: {antenna} - {i}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(256),
                y=sim[antenna, :, i],
                mode="lines",
                name=f"sim: {antenna} - {i}",
            )
        )
        plots.append(fig)
    return plots


def plot_pulses(pulses):
    """
    Plot pulses from efield.
    """

    timec7 = pulses[:, -4]
    exc7 = pulses[:, -3]
    eyc7 = pulses[:, -2]
    ezc7 = pulses[:, -1]

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(40, 30))
    ax[0].plot(timec7, exc7)
    ax[0].set_title("x - C7 CoREAS")

    ax[1].plot(timec7, eyc7)
    ax[1].set_title("y - C7 CoREAS")

    ax[2].plot(timec7, ezc7)
    ax[2].set_title("z - C7 CoREAS")

    fig.suptitle("Pulses")
    fig.supylabel("Electric Field [$\\mu$V/m]")
    fig.supxlabel("Time [s]")
    [i.grid(which="both", linestyle="dashed") for i in ax]
    plt.tight_layout()
    plt.savefig("pulse.pdf", format="pdf")
    plt.show()


def plot_interpolated_footprint(positions, energy_fluences, interp):
    """

    Plot the interpolated footprint.

    Parameters
    ----------
    positions: np.array
        Antenna positions
    energy_fluences: np.array
        Energy fluence for each antenna
    interp: Bool
        interpolate the intermediate points.

    Returns
    -------
    None
    """
    if len(energy_fluences.shape) == 1:
        energy_fluences = np.array([energy_fluences]).T
    x_pos, y_pos = positions[:, 0], positions[:, 1]
    for i in range(energy_fluences.shape[1]):
        energy_flu = energy_fluences[:, i]
        if np.min(energy_flu) == np.max(energy_flu):
            print(np.min(energy_flu))
            continue
        fig, ax = plt.subplots(
            1,
            2,
            figsize=(int(1.333 * fnt_size), fnt_size),
            gridspec_kw={"width_ratios": [30, 1]},
        )
        if interp:
            # construct the interpolation function
            interp_func = intp.Rbf(
                x_pos,
                y_pos,
                energy_flu,
                smooth=0,
                function="quintic",
            )
            # define positions where to interpolate
            xs = np.linspace(np.min(x_pos), np.max(x_pos), 100)
            ys = np.linspace(np.min(y_pos), np.max(y_pos), 100)
            xx, yy = np.meshgrid(xs, ys)
            # points within a circle
            in_star = xx**2 + yy**2 <= np.amax(x_pos**2 + y_pos**2)
            # interpolated values! but only in the star. outsite set to nan
            fp_interp = np.where(in_star, interp_func(xx, yy), np.nan)
            cmap = "inferno"  # set the colormap
            # with vmin/vmax control that both
            # pcolormesh and scatter use the same colorscale
            pcm = ax[0].pcolormesh(
                xx,
                yy,
                fp_interp,
                vmin=np.percentile(energy_flu, 0),
                vmax=np.percentile(energy_flu, 100),
                cmap=cmap,
                shading="gouraud",
            )  # use shading="gouraud" to make it smoother
            _ = ax[0].scatter(
                x_pos,
                y_pos,
                edgecolor="w",
                facecolor="none",
                s=5.0,
                lw=1.0,
            )
            cbi = fig.colorbar(pcm, pad=0.02, cax=ax[1])
            cbi.set_label(r"Energy Fluence $f$ / eV$\,$m$^{-2}$", fontsize=20)
        else:
            _ = ax[0].scatter(
                x_pos,
                y_pos,
                c=energy_flu,
                edgecolor="w",
                facecolor="none",
                s=fnt_size / 5,
                lw=fnt_size / 10,
            )

        ax[0].set_ylabel("y / m", fontsize=fnt_size)
        ax[0].set_xlabel("x / m", fontsize=fnt_size)
        ax[0].set_facecolor("black")
        ax[0].set_aspect(1)
        ax[0].set_xlim(np.min(x_pos), np.max(x_pos))
        ax[0].set_ylim(np.min(y_pos), np.max(y_pos))
        fig.suptitle(f"CORSIKA 7 - all {i}")
        print("vmin = ", np.amin(energy_flu))
        print("vmax = ", np.amax(energy_flu))
        plt.xticks(fontsize=fnt_size)
        plt.yticks(fontsize=fnt_size)
        plt.tight_layout()
        plt.savefig("fluence_maps.pdf", format="pdf")
        plt.show()


def plot_fluence_maps(
    pulses,
    pos_array,
    interp=True,
):
    """
    Plot fluence maps from given hdf5 file.

    Parameters
    ----------
    pulses:
        pulses seen in all the antennas
    pos_array:
        positions of all the antennas
    interp:
        interpolate the intermediate points when plotting fluences

    Returns
    -------
    None
    """
    energy_fluences = []
    positions = []
    for index in range(len(pulses)):
        pos = pos_array[index]
        trace_vB = pulses[index]  # 0,1,2,3: t, vxB, vxvxB, v
        positions.append(pos)
        ef = energy_fluence.calculate_energy_fluence_vector(
            trace_vB[:, 1:], trace_vB[:, 0], remove_noise=True
        )
        energy_fluences.append(ef)

    positions = np.array(positions)
    energy_fluences = np.array(energy_fluences)

    print(len(positions), len(energy_fluences))
    assert len(positions) == len(energy_fluences)
    print(len(positions))
    plot_interpolated_footprint(positions, energy_fluences, interp)
