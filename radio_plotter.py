import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as intp
from radiotools.analyses import energy_fluence

fnt_size = 20
plt.rc('font', size=fnt_size) #controls default text size
plt.rc('axes', titlesize=fnt_size) #fontsize of the title
plt.rc('axes', labelsize=fnt_size) #fontsize of the x and y labels
plt.rc('xtick', labelsize=fnt_size) #fontsize of the x tick labels
plt.rc('ytick', labelsize=fnt_size) #fontsize of the y tick labels
plt.rc('legend', fontsize=fnt_size)
def indent_level(level):
    return "".join(["  " for _ in range(level)])

def print_hdf5_file(file_name, level=0):
    if type(file_name) is str:
        ff = h5py.File(file_name, 'r')
    else:
        ff = file_name
    for k in ff.keys():
        if isinstance(ff[k], h5py._hl.group.Group):
            print(" Group", ff[k].name)
            print(" ----------------------------------")
            print_hdf5_file(ff[ff[k].name], level=level+1)
        elif isinstance(ff[k], h5py._hl.dataset.Dataset):
            
            print(f"{indent_level(level)} Group:{ff.name} Key:{k}")
            print(f"{indent_level(level)} Property: {ff[k]}")
            if len(ff[k].attrs.keys()) != 0:
                print(f"{indent_level(level)} Attributes")
            for l in ff[k].attrs.keys():
                print(f"{indent_level(level+1)} Key:{l}")
                print(f"{indent_level(level+1)} Value:{ff[k].attrs[l]}")
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
    
def plot_pulses(file_name, antenna_label):
    if type(file_name) is str:
        f_h5 = h5py.File(file_name, 'r')
    else:
        f_h5 = file_name
    plotstyle1 = {'color': 'r',
                 'marker':'.'
              }
    plotstyle2 = {'color': 'b',
                 'marker':'.'
              }
    pulses = f_h5['/CoREAS/ge_ce'][antenna_label]
    
    timec7 = pulses[:,-4]
    exc7 = pulses[:,-3]
    eyc7 = pulses[:,-2]
    ezc7 = pulses[:,-1]
    
    assert np.all(f_h5['/CoREAS/observers'][antenna_label][:,0] == timec7)
    fig, ax = plt.subplots(3, 2, sharex=True, figsize=(40,30))
    ax[0,0].plot(timec7, eyc7, **plotstyle1)
    ax[0,0].set_title(f'GEO - C7 CoREAS')

    ax[1,0].plot(timec7, exc7, **plotstyle1)
    ax[1,0].set_title(f'CE - C7 CoREAS')

    ax[2,0].plot(timec7, ezc7, **plotstyle1)
    ax[2,0].set_title(f'Zeros - C7 CoREAS')

    
    pulses = f_h5['/CoREAS/observers'][antenna_label]
    timec7 = pulses[:,-4]
    exc7 = pulses[:,-3]
    eyc7 = pulses[:,-2]
    ezc7 = pulses[:,-1]

    ax[0,1].plot(timec7, eyc7, **plotstyle1)
    ax[0,1].set_title(f'vB - C7 CoREAS')

    ax[1,1].plot(timec7, exc7, **plotstyle1)
    ax[1,1].set_title(f'vvB - C7 CoREAS')

    ax[2,1].plot(timec7, ezc7, **plotstyle1)
    ax[2,1].set_title(f'v - C7 CoREAS')
    
    
    fig.suptitle(f"Run:{file_name} Antenna: {antenna_label}")
    fig.supylabel('Electric Field [$\mu$V/m]')
    fig.supxlabel('Time [s]')
    [k.grid(which="both", linestyle="dashed") for i in ax for k in i]
    plt.tight_layout()
    plt.savefig(f"plot_{antenna_label}.pdf", format='pdf')
    plt.show()
    
    if type(file_name) is str:
        f_h5.close()
        
def plot_interpolated_footprint(positions, energy_fluences, interp):
    # construct the interpolation function
    if len(energy_fluences.shape) == 1:
        energy_fluences = np.array([energy_fluences]).T
    for i in range(energy_fluences.shape[1]):
        energy_flu = energy_fluences[:,i]
        if np.min(energy_flu) == np.max(energy_flu):
            print(np.min(energy_flu))
            continue
        fig, ax = plt.subplots(1, 2, figsize=(40, 30), gridspec_kw={
            'width_ratios': [30, 1]})
        if interp:
            interp_func = intp.Rbf(
                positions[:, 0], positions[:, 1], energy_flu, smooth=0, function='quintic')
            # define positions where to interpolate
            xs = np.linspace(np.min(positions), np.max(positions), 100)
            ys = np.linspace(np.min(positions), np.max(positions), 100)
            xx, yy = np.meshgrid(xs, ys)
            # points within a circle
            in_star = xx ** 2 + yy ** 2 <= np.amax(positions[:, 0] ** 2 + positions[:, 1] ** 2)
            # interpolated values! but only in the star. outsite set to nan
            fp_interp = np.where(in_star, interp_func(xx, yy), np.nan)
            cmap = "inferno"  # set the colormap
            # with vmin/vmax control that both pcolormesh and scatter use the same colorscale
            pcm = ax[0].pcolormesh(yy, xx, fp_interp,
                                vmin=np.amin(energy_flu), vmax=np.amax(energy_flu),
                                cmap=cmap, shading="gouraud")  # use shading="gouraud" to make it smoother
            sct = ax[0].scatter(
                positions[:, 1],
                positions[:, 0],
                edgecolor="w",
                facecolor="none",
                s=5.0,
                lw=1.,
            )
            cbi = fig.colorbar(pcm, pad=0.02, cax=ax[1])
            cbi.set_label(r"Energy Fluence $f$ / eV$\,$m$^{-2}$", fontsize=20)
        else:
            sct = ax[0].scatter(
                    positions[:, 1],
                    positions[:, 0],
                    c=energy_flu,
                    edgecolor="w",
                    facecolor="none",
                    s=5.0,
                    lw=1.,
            )

        ax[0].set_ylabel("y / m",fontsize=20)
        ax[0].set_xlabel("x / m",fontsize=20)
        ax[0].set_facecolor("black")
        ax[0].set_aspect(1)
        ax[0].set_xlim(np.min(positions), np.max(positions))
        ax[0].set_ylim(np.min(positions), np.max(positions))
        fig.suptitle("CORSIKA 7 - all")
        print("vmin = ", np.amin(energy_flu))
        print("vmax = ", np.amax(energy_flu))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(f"fluence_{i}.pdf", format='pdf')
        plt.show()

def plot_fluence_maps(file_name, from_file=False, data='ge_ce', interp=True):
    if type(file_name) is str:
        f_h5 = h5py.File(file_name, 'r')
    else:
        f_h5 = file_name
        
    antennas_pos = f_h5['/highlevel/obsplane_na_na_vB_vvB'][
        'antenna_position_vBvvB']
    antennas = f_h5[f'/CoREAS/{data}']
    energy_fluences = []
    positions = []
    for index, label in enumerate(antennas.keys()):
        #print(label)
        if label[:3] != 'pos':
            continue
        pos = antennas_pos[index]
        #pos = antennas[label].attrs['position']
        trace_vB = antennas[label]  # 0,1,2,3: t, vxB, vxvxB, v
        positions.append(pos)
        if not from_file:
            ef = energy_fluence.calculate_energy_fluence_vector(
                trace_vB[:,1:], trace_vB[:,0], remove_noise=True)
            # store all energy fluences (for all antennas) in a list
            energy_fluences.append(ef)
        else:
            energy_fluences.append(f_h5['/highlevel/obsplane_na_na_vB_vvB'][
                                       'energy_fluence'][index])

    positions = np.array(positions)
    energy_fluences = np.array(energy_fluences)
    assert len(positions) == len(energy_fluences)
    print(len(positions))
    plot_interpolated_footprint(positions, energy_fluences, interp)

    
    if type(file_name) is str:
        f_h5.close()