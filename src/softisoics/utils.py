import h5py
import numpy as np
from scipy.interpolate import CubicSpline


def get_interp(x_bins, y_bins):
    """
    Get the interpolated profiles.

    Parameters:
    ----------
    x_bins: array
        Array of the x-param.
    y_bins: array
        Array of the y-param.

    Returns:
    -------
    interp: CubicSpline
        Interpolated profile.
    """

    x_order = np.argsort(x_bins)
    x_increasing_mask = np.append([True], np.diff(x_bins[x_order]) > 0)

    x_bins = x_bins[x_order][x_increasing_mask]
    y_bins = y_bins[x_order][x_increasing_mask]

    finite_mask = np.logical_and(np.isfinite(x_bins), np.isfinite(y_bins))

    return CubicSpline(x_bins[finite_mask], y_bins[finite_mask])


def get_N_part_from_N_200(N_200, r200, r_sample_max, r_bins, mass_bins):
    """
    Get the number of particles to sample from the halo given the number of particles within r200.

    Parameters:
    ----------
    N_200: int
        Number of particles within r200.
    r200: float
        Virial radius of the halo.
    r_sample_max: float
        Maximum radius to sample particles from.
    r_bins: array
        Array of the radius bins.
    mass_bins: array
        Array of the mass bins corresponding to the radius bins.

    Returns:
    -------
    N_sample: int
        Number of particles to sample from the halo.
    """

    if r_sample_max > r_bins[-1]:
        raise ValueError("Sample radius out of bounds.")  # noqa: EM101, TRY003

    if r_sample_max < r200:
        raise ValueError("Not sampling the whole halo.")  # noqa: EM101, TRY003

    _log_log_mass_interp = get_interp(np.log(r_bins), np.log(mass_bins))

    _M_200 = np.exp(_log_log_mass_interp(np.log(r200)))
    _M_sample = np.exp(_log_log_mass_interp(np.log(r_sample_max)))

    return int(N_200 * _M_sample / _M_200)


class Halo:
    """
    Halo class to check the profiles of the halo sampled particles.
    """

    def input_from_particles(self, part_coords, part_velocs, part_mass):
        """
        Read the particle data from the input arrays.

        Parameters:
        ----------
        part_coords: array
            Array of particle coordinates.
        part_velocs: array
            Array of particle velocities.
        part_mass: float
            Mass of each particle.
        """

        self.part_coords = part_coords
        self.part_velocs = part_velocs

        self.part_mass = part_mass

    def input_from_file(self, data_file):
        """
        Read the particle data from the input file.

        Parameters:
        ----------
        data_file: str
            Path to the input file.
        """

        data = h5py.File(data_file, "r")

        self.part_coords = data["PartType1/Coordinates"][:]
        self.part_velocs = data["PartType1/Velocities"][:]

        self.part_mass = data["PartType1/Masses"][0]

    def _center_halo(self):
        """
        Center the halo and calculate the radii and radial velocities of the particles.
        """

        _halo_center = np.mean(self.part_coords, axis=0)

        self.part_coords -= _halo_center
        self.part_radii = np.sqrt(np.sum(self.part_coords**2, axis=1))

        self.part_radial_velocs = (
            np.sum(self.part_coords * self.part_velocs, axis=1) / self.part_radii
        )

    def get_profiles(self, r_bin_edges=None):
        """
        Get the density and radial velocity dispersion profiles of the halo.
        """

        self._center_halo()

        _r_bin_edges = (
            r_bin_edges if r_bin_edges is not None else np.logspace(-3, 2, 51)
        )
        r_bins = 0.5 * (_r_bin_edges[:-1] + _r_bin_edges[1:])
        _V_bins = 4 / 3 * np.pi * (_r_bin_edges[1:] ** 3 - _r_bin_edges[:-1] ** 3)

        _bin_indices = np.digitize(self.part_radii, _r_bin_edges) - 1

        _count_bins = np.array(
            [
                np.sum((self.part_radii[_bin_indices == i]).shape[0])
                for i in range(len(r_bins))
            ]
        )

        rho_bins = _count_bins * self.part_mass / _V_bins
        rho_bins_err = np.sqrt(_count_bins) * self.part_mass / _V_bins

        sigma_r_bins = np.array(
            [
                np.std(self.part_radial_velocs[_bin_indices == i])
                for i in range(len(r_bins))
            ]
        )
        sigma_r_bins_err = sigma_r_bins / np.sqrt(2 * _count_bins)

        return r_bins, rho_bins, rho_bins_err, sigma_r_bins, sigma_r_bins_err
