import numpy as np
import units
import matplotlib.pyplot as plt
from radiotools import helper as hp
from radiotools import coordinatesystems

import h5py
from scipy import constants
from scipy.integrate import simps




def get_angles(corsika, declination):
    """
    Converting angles in corsika coordinates to local coordinates.

    Corsika positive x-axis points to the magnetic north, NRR coordinates positive x-axis points to the geographic east.
    Corsika positive y-axis points to the west, NRR coordinates positive y-axis points to the geographic north.
    Corsika z-axis points upwards, NuRadio z-axis points upwards.

    Corsika's zenith angle of a particle trajectory is defined between the particle momentum vector and the negative
    z-axis, meaning that the particle is described in the direction where it is going to. The azimuthal angle is
    described between the positive x-axis and the horizontal component of the particle momentum vector
    (i.e. with respect to the magnetic north) proceeding counterclockwise.

    NRR describes the particle zenith and azimuthal angle in the direction where the particle is coming from.
    Therefore, the zenith angle is the same, but the azimuthal angle has to be shifted by 180 + 90 degrees.
    The north has to be shifted by 90 degrees plus difference between geomagnetic and magnetic north.

    Parameters
    ----------
    corsika : hdf5 file object
        the open hdf5 file object of the corsika hdf5 file
    declination : float
        declination of the magnetic field, in internal units

    Returns
    -------
    zenith : float
        zenith angle
    azimuth : float
        azimuth angle
    magnetic_field_vector : np.ndarray
        magnetic field vector

    Examples
    --------
    The declinations can be obtained using the functions in the radiotools helper package, if you
    have the magnetic field for the site you are interested in.

    >>> magnet = hp.get_magnetic_field_vector('mooresbay')
    >>> dec = hp.get_declination(magnet)
    >>> evt = h5py.File('NuRadioReco/examples/example_data/example_data.hdf5', 'r')
    >>> get_angles(corsika, dec)[2] / units.gauss
    array([ 0.05646405, -0.08733734,  0.614     ])
    >>> magnet
    array([ 0.058457, -0.09042 ,  0.61439 ])
    """
    zenith = corsika['inputs'].attrs["THETAP"][0] * units.deg
    azimuth = hp.get_normalized_angle(
        3 * np.pi / 2. + np.deg2rad(corsika['inputs'].attrs["PHIP"][0]) + declination / units.rad
    ) * units.rad

    # in CORSIKA convention, the first component points North (y in NRR) and the second component points down (minus z)
    By, minBz = corsika['inputs'].attrs["MAGNET"]
    B_inclination = np.arctan2(minBz, By)  # angle from y-axis towards negative z-axis

    B_strength = np.sqrt(By ** 2 + minBz ** 2) * units.micro * units.tesla

    # zenith of the magnetic field vector is 90 deg + inclination, as inclination proceeds downwards from horizontal
    # azimuth of the magnetic field vector is 90 deg - declination, as declination proceeds clockwise from North
    magnetic_field_vector = B_strength * hp.spherical_to_cartesian(
        np.pi / 2 + B_inclination, np.pi / 2 - declination / units.rad
    )

    return zenith, azimuth, magnetic_field_vector

def get_geomagnetic_angle(zenith, azimuth, magnetic_field_vector):
    """
    Calculates the angle between the geomagnetic field and the shower axis defined by `zenith` and `azimuth`.

    Parameters
    ----------
    zenith : float
        zenith angle (in internal units)
    azimuth : float
        azimuth angle (in internal units)
    magnetic_field_vector : np.ndarray
        The magnetic field vector in the NRR coordinate system (x points East, y points North, z points up)

    Returns
    -------
    geomagnetic_angle : float
        geomagnetic angle
    """
    shower_axis_vector = hp.spherical_to_cartesian(zenith / units.rad, azimuth / units.rad)
    geomagnetic_angle = hp.get_angle(magnetic_field_vector, shower_axis_vector) * units.rad

    return geomagnetic_angle

def convert_obs_to_nuradio_efield(observer, zenith, azimuth, magnetic_field_vector):
    """
    Converts the electric field from one CoREAS observer to NuRadio units and the on-sky coordinate system.

    The on-sky CS in NRR has basis vectors eR, eTheta, ePhi.
    Before going to the on-sky CS, we account for the magnetic field which does not point strictly North.
    To get the zenith, azimuth and magnetic field vector, one can use `get_angles()`.
    The `observer` array should have the shape (n_samples, 4) with the columns (time, Ey, -Ex, Ez),
    where (x, y, z) is the NuRadio CS.

    Parameters
    ----------
    observer : np.ndarray
        The observer as in the HDF5 file, e.g. list(corsika['CoREAS']['observers'].values())[i].
    zenith : float
        zenith angle (in internal units)
    azimuth : float
        azimuth angle (in internal units)
    magnetic_field_vector : np.ndarray
        magnetic field vector

    Returns
    -------
    efield: np.array (3, n_samples)
        Electric field in the on-sky CS (r, theta, phi)
    efield_times: np.array (n_samples)
        The time values corresponding to the electric field samples

    """
    conversion_fieldstrength_cgs_to_SI = 2.99792458e10 * units.micro * units.volt / units.meter
    
    cs = coordinatesystems.cstrafo(
        zenith / units.rad, azimuth / units.rad,
        magnetic_field_vector  # the magnetic field vector is used to find showerplane, so only direction is important
    )

    efield_times = observer[:, 0] * units.second
    efield = np.array([
        observer[:, 2] * -1,  # CORSIKA y-axis points West
        observer[:, 1],
        observer[:, 3]
    ]) * conversion_fieldstrength_cgs_to_SI

    # convert coreas efield to NuRadio spherical coordinated eR, eTheta, ePhi (on sky)
    efield_geographic = cs.transform_from_magnetic_to_geographic(efield)
    efield_on_sky = cs.transform_from_ground_to_onsky(efield_geographic)

    return efield_on_sky, efield_times

def convert_obs_positions_to_nuradio_on_ground(observer_pos, declination=0):
    """
    Convert observer positions from the CORSIKA CS to the NRR ground CS.

    First, the observer position is converted to the NRR coordinate conventions (i.e. x pointing East,
    y pointing North, z pointing up). Then, the observer position is corrected for magnetic north
    (as CORSIKA only has two components to its magnetic field vector) and put in the geographic CS.
    To get the zenith, azimuth and magnetic field vector, one can use the `get_angles()` function.
    If multiple observers are to be converted, the `observer` array should have the shape (n_observers, 3).

    Parameters
    ----------
    observer_pos : np.ndarray
        The observer's position as extracted from the HDF5 file, e.g. corsika['CoREAS']['my_observer'].attrs['position']
    declination : float (default: 0)
        Declination of the magnetic field.

    Returns
    -------
    obs_positions_geo: np.ndarray
        observer positions in geographic coordinates, shaped as (n_observers, 3).
    """
    # If single position is given, make sure it has the right shape (3,) -> (1, 3)
    if observer_pos.ndim == 1:
        observer_pos = observer_pos[np.newaxis, :]

    obs_positions = np.array([
        observer_pos[:, 1] * -1,
        observer_pos[:, 0],
        observer_pos[:, 2]
    ]) * units.cm

    obs_positions = hp.rotate_vector_in_2d(obs_positions, -declination).T

    return np.squeeze(obs_positions)

def Fluence(traces):
    traces = np.array(traces)
    eps_0 = constants.epsilon_0 * units.farad / units.m
    c_vacuum = constants.c * units.m / units.s
    conversion_fieldstrength_cgs_to_SI = 2.99792458e10 * units.milli *units.volt / units.meter
    delta_t = 0.1 * units.ns

    # make sure to use your trace and define the relevant axis for the summing here
    traces *= conversion_fieldstrength_cgs_to_SI
    fluences = eps_0 * c_vacuum * delta_t * np.sum(np.array(traces[0])**2 + np.array(traces)[1]**2 + np.array(traces)[2]**2)
    
    return fluences

def plot(x_vals,y_vals,FluenceValues,Title,label):

    # Set style
    plt.style.use("seaborn-v0_8-darkgrid")  # Beautiful dark-grid background

    # Create figure
    plt.figure(figsize=(8, 6), dpi=300)

    # Scatter plot with enhanced aesthetics
    scatter = plt.scatter(
        x_vals, 
        y_vals, 
        c=np.array(FluenceValues), 
        #c=np.log(np.array(FluenceValues)), 
        cmap="viridis", 
        s=120, 
        edgecolors="white", 
        linewidths=0.8, 
        alpha=0.85  # Adds transparency for better visualization
    )

    # Add color bar with improved formatting
    cbar = plt.colorbar(scatter)
    cbar.set_label(f"{label}", fontsize=12, labelpad=10)
    cbar.ax.tick_params(labelsize=10)

    # Labels and title with improved font settings
    plt.xlabel("X Values", fontsize=14, fontweight="bold")
    plt.ylabel("Y Values", fontsize=14, fontweight="bold")
    plt.title(f"{Title}", fontsize=16, fontweight="bold", pad=15)

    # Improve axis ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Show plot
    plt.show()


SIM_NUMBER = '000003'
file_name = f"/Users/denis/Desktop/BachelorThesis/data/177113844/1/iron/SIM{SIM_NUMBER}.hdf5"

f_h5 = h5py.File(file_name, "r")
declination = 0  #np.pi+np.pi/4 + np.pi/38

zenith, azimuth, magnetic_field_vector = get_angles(f_h5,declination)
geomagnetic_angle = get_geomagnetic_angle(zenith, azimuth, magnetic_field_vector)



all_antennas = []
all_efield = []
all_efield_time = []

for j_obs, observer in enumerate(f_h5['CoREAS']['observers'].values()):
    
    obs_positions_geo = convert_obs_positions_to_nuradio_on_ground(
            observer.attrs['position'], declination
        )
    efield, efield_time = convert_obs_to_nuradio_efield(
            observer, zenith, azimuth, magnetic_field_vector
        )
    
    all_antennas.append(obs_positions_geo)
    all_efield.append(efield)
    all_efield_time.append(efield_time)
    
all_antennas = np.array(all_antennas) 
all_efield = np.array(all_efield)  
all_efield_time = np.array(all_efield_time)


print(all_antennas.shape)
all_fluences = []

for i in all_efield:
    fl = Fluence(i)
    all_fluences.append(fl)
    
cs = coordinatesystems.cstrafo(
        zenith, azimuth, magnetic_field_vector=magnetic_field_vector
    )

shower_antennas = cs.transform_to_vxB_vxvxB(all_antennas,[0,0,7.6])

plot(shower_antennas[:,0],shower_antennas[:,1],all_fluences,'Fluence with NuRadioReco conversion','Fluence with NuRadioReco conversion')