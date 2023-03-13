"""
Plot radio pulses and fluence maps.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as intp
from radiotools.analyses import energy_fluence
from radiotools import helper as rdhelp

fnt_size = 20
plt.rc("font", size=fnt_size)  # controls default text size
plt.rc("axes", titlesize=fnt_size)  # fontsize of the title
plt.rc("axes", labelsize=fnt_size)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=fnt_size)  # fontsize of the x tick labels
plt.rc("ytick", labelsize=fnt_size)  # fontsize of the y tick labels
plt.rc("legend", fontsize=fnt_size)


def indent_level(level):
    """
    Indent according to hierarchy when printing hdf5 file.

    Parameters
    ----------
    level: int
        Indent level

    Returns
    -------
    None
    """
    return "".join(["  " for _ in range(level)])


def print_hdf5_file(file_name, level=0):
    """
    Print the structure of the hdf5 file.

    Parameters
    ----------
    file_name: str or h5py class
    level: Indentation level in printing (Used only internally for recursion)

    Returns
    -------
    None
    """
    if type(file_name) is str:
        ff = h5py.File(file_name, "r")
    else:
        ff = file_name
    for k in ff.keys():
        if isinstance(ff[k], h5py._hl.group.Group):
            print(" Group", ff[k].name)
            print(" ----------------------------------")
            print_hdf5_file(ff[ff[k].name], level=level + 1)
        elif isinstance(ff[k], h5py._hl.dataset.Dataset):

            print(f"{indent_level(level)} Group:{ff.name} Key:{k}")
            print(f"{indent_level(level)} Property: {ff[k]}")
            if len(ff[k].attrs.keys()) != 0:
                print(f"{indent_level(level)} Attributes")
            for ll in ff[k].attrs.keys():
                print(f"{indent_level(level+1)} Key:{ll}")
                print(f"{indent_level(level+1)} Value:{ff[k].attrs[ll]}")
                print(f"{indent_level(level+1)} -----------------------")
        else:
            raise KeyboardInterrupt

    if len(ff.attrs.keys()) != 0:
        print(f"{indent_level(level)} Attributes")
    for k in ff.attrs.keys():
        print(f"{indent_level(level)} Group:{ff.name} Key:{k}")
        print(f"{indent_level(level)} Value: {ff.attrs[k]}")

    if type(file_name) is str:
        ff.close()


def plot_pulses_at_rad(file_name, radius, data="vB_vvB"):
    """
    Plot pulses from given file at particular radius

    Parameters
    ----------
    file_name: str or hdf5 class
    antenna_label: str
        Which antenna to plot

    Returns
    -------
    None
    """
    if type(file_name) is str:
        f_h5 = h5py.File(file_name, "r")
    else:
        f_h5 = file_name
    labels = []
    avail_radius = []
    for label in f_h5[f"/highlevel/traces/{data}"].keys():
        split_label = label.split("_")
        if split_label[0] != "pos":
            continue
        elif split_label[1] == str(radius):
            labels.append(label)
        else:
            avail_radius.append(int(split_label[1]))
            continue

    if len(labels) == 0:
        print("No labels found to plot")
        print("Available Radii:")
        print(sorted(set(avail_radius)))
        return

    plotstyle1 = {"color": "r", "marker": "."}
    # plotstyle2 = {"color": "b", "marker": "."}
    fig, ax = plt.subplots(3, 8, figsize=(40, 30), sharey=True)
    for i, antenna_label in enumerate(labels):
        angle = antenna_label.split("_")[2]
        pulses = f_h5[f"/highlevel/traces/{data}"][antenna_label]

        timec7 = pulses[:, -4]
        exc7 = pulses[:, -3]
        eyc7 = pulses[:, -2]
        ezc7 = pulses[:, -1]

        ax[0, i].plot(timec7, exc7, **plotstyle1)
        ax[0, i].set_title(f"{data.split('_')[0]} - C7 CoREAS - {angle}")

        ax[1, i].plot(timec7, eyc7, **plotstyle1)
        ax[1, i].set_title(f"{data.split('_')[1]} - C7 CoREAS - {angle}")

        ax[2, i].plot(timec7, ezc7, **plotstyle1)
        ax[2, i].set_title(f"v - C7 CoREAS - {angle}")

    fig.suptitle(f"Run:{file_name} Radius: {radius}")
    fig.supylabel("Electric Field [$\\mu$V/m]")
    fig.supxlabel("Time [s]")
    [k.grid(which="both", linestyle="dashed") for i in ax for k in i]
    plt.tight_layout()
    plt.savefig(f"plot_rad_{radius}.pdf", format="pdf")
    plt.show()

    if type(file_name) is str:
        f_h5.close()


def plot_pulses(file_name, antenna_label):
    """
    Plot pulses from given file.

    Parameters
    ----------
    file_name: str or hdf5 class
    antenna_label: str
        Which antenna to plot

    Returns
    -------
    None
    """
    if type(file_name) is str:
        f_h5 = h5py.File(file_name, "r")
    else:
        f_h5 = file_name
    plotstyle1 = {"color": "r", "marker": "."}
    # plotstyle2 = {"color": "b", "marker": "."}
    pulses = f_h5["/highlevel/traces/ge_ce"][antenna_label]
    print(pulses.shape)

    timec7 = pulses[:, -4]
    exc7 = pulses[:, -3]
    eyc7 = pulses[:, -2]
    ezc7 = pulses[:, -1]

    fig, ax = plt.subplots(3, 2, sharex=True, figsize=(40, 30))
    ax[0, 0].plot(timec7, exc7, **plotstyle1)
    ax[0, 0].set_title(f"GEO - C7 CoREAS")

    ax[1, 0].plot(timec7, eyc7, **plotstyle1)
    ax[1, 0].set_title(f"CE - C7 CoREAS")

    ax[2, 0].plot(timec7, ezc7, **plotstyle1)
    ax[2, 0].set_title(f"Zeros - C7 CoREAS")

    pulses = f_h5["/highlevel/traces/vB_vvB"][antenna_label]
    print(pulses.shape)
    timec7 = pulses[:, -4]
    exc7 = pulses[:, -3]
    eyc7 = pulses[:, -2]
    ezc7 = pulses[:, -1]

    ax[0, 1].plot(timec7, exc7, **plotstyle1)
    ax[0, 1].set_title(f"vB - C7 CoREAS")

    ax[1, 1].plot(timec7, eyc7, **plotstyle1)
    ax[1, 1].set_title(f"vvB - C7 CoREAS")

    ax[2, 1].plot(timec7, ezc7, **plotstyle1)
    ax[2, 1].set_title(f"v - C7 CoREAS")

    fig.suptitle(f"Run:{file_name} Antenna: {antenna_label}")
    fig.supylabel("Electric Field [$\\mu$V/m]")
    fig.supxlabel("Time [s]")
    [k.grid(which="both", linestyle="dashed") for i in ax for k in i]
    plt.tight_layout()
    plt.savefig(f"plot_{antenna_label}.pdf", format="pdf")
    plt.show()

    if type(file_name) is str:
        f_h5.close()


def plot_interpolated_footprint(positions, energy_fluences, interp, highlight_antenna):
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
            # with vmin/vmax control that both pcolormesh and scatter use the same colorscale
            pcm = ax[0].pcolormesh(
                xx,
                yy,
                fp_interp,
                vmin=np.percentile(energy_flu, 0),
                vmax=np.percentile(energy_flu, 100),
                cmap=cmap,
                shading="gouraud",
            )  # use shading="gouraud" to make it smoother
            sct = ax[0].scatter(
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
            sct = ax[0].scatter(
                x_pos,
                y_pos,
                c=energy_flu,
                edgecolor="w",
                facecolor="none",
                s=fnt_size / 5,
                lw=fnt_size / 10,
            )

        if len(highlight_antenna) != 0:
            sct = ax[0].scatter(
                x_pos[highlight_antenna],
                y_pos[highlight_antenna],
                edgecolor="r",
                facecolor="none",
                s=fnt_size / 1,
                lw=fnt_size / 1,
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
        plt.savefig(f"fluence_{i}.pdf", format="pdf")
        plt.show()


def plot_fluence_maps(
    file_name,
    from_file=False,
    data="ge_ce",
    interp=True,
    highlight_radius=None,
    hack=False,
):
    """
    Plot fluence maps from given hdf5 file.

    Parameters
    ----------
    file_name: str, hdf5 class
        Source File
    from_file: Book
        Plot energy fluence already found in file, else calculate using
        radiotools
    data:
        which data to plot from source file
    interp:
        interpolate the intermediate points when plotting fluences

    Returns
    -------
    None
    """
    if type(file_name) is str:
        f_h5 = h5py.File(file_name, "r")
    else:
        f_h5 = file_name

    antennas = f_h5[f"/CoREAS/observers"]
    trace = f_h5[f"/highlevel/traces/{data}"]
    antennas_pos = f_h5[f"/highlevel/positions/ge_ce"]
    energy_fluences = []
    positions = []
    highlight_antenna = []
    for index, label in enumerate(antennas.keys()):
        # print(label)
        if label.split("_")[0] != "pos":
            continue
        if int(label.split("_")[1]) == highlight_radius:
            highlight_antenna.append(len(positions))
        pos = antennas_pos[label]
        trace_vB = trace[label]  # 0,1,2,3: t, vxB, vxvxB, v
        positions.append(pos)
        if not from_file:
            ef = energy_fluence.calculate_energy_fluence_vector(
                trace_vB[:, 1:], trace_vB[:, 0], remove_noise=True
            )
            # store all energy fluences (for all antennas) in a list
        else:
            ef = f_h5["/highlevel/obsplane_na_na_vB_vvB"]["energy_fluence_vector"][
                index
            ]
        # print(index, label, f"pos:{pos}", f"ef{ef}")

        energy_fluences.append(ef)

    positions = np.array(positions)
    energy_fluences = np.array(energy_fluences)

    if hack:
        for index in range(len(energy_fluences)):
            if index % 8 == 0:
                energy_fluences[index] = (
                    energy_fluences[index + 4] + energy_fluences[index + 5]
                ) / 2
            if index % 8 == 2:
                energy_fluences[index] = (
                    energy_fluences[index - 1] + energy_fluences[index + 1]
                ) / 2

    print(len(positions), len(energy_fluences))
    assert len(positions) == len(energy_fluences)
    print(len(positions))
    plot_interpolated_footprint(positions, energy_fluences, interp, highlight_antenna)

    if type(file_name) is str:
        f_h5.close()


def plot_long_prof(file_name):
    if type(file_name) is str:
        f_h5 = h5py.File(file_name, "r")
    else:
        f_h5 = file_name

    fig, ax = plt.subplots(3, 3, figsize=(40, 30))

    longprof = f_h5["atmosphere"]["NumberOfParticles"]

    depth = longprof[:, 0]
    names = longprof.attrs["comment"].split(",")
    for i in range(9):
        ax[i // 3, i % 3].plot(depth, longprof[:, i + 1])
        ax[i // 3, i % 3].set_title(
            f"{names[i+1]}, XMAX: " f"{depth[longprof[:,i+1].argmax()]}"
        )

    fig.suptitle(f"Run:{file_name} ")
    fig.supylabel("NumberOfParticles")
    fig.supxlabel("Depth")
    [k.grid(which="both", linestyle="dashed") for i in ax for k in i]
    plt.tight_layout()
    plt.savefig(f"plot_longprof.pdf", format="pdf")
    plt.show()

    if type(file_name) is str:
        f_h5.close()
