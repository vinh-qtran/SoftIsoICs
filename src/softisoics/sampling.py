import numpy as np
from scipy.integrate import cumulative_trapezoid
from tqdm import tqdm

from softisoics.utils import get_interp


class SampleParticles:
    def __init__(
        self,
        r_bins,
        rho_bins,
        mass_bins,
        total_phi_bins,
        eps_bins,
        f_eps_bins,
        r_sample_min,
        r_sample_max,
        N_part,
        kappa_L=0.0,
        seed=42,
    ):
        """
        Sample particles from the halo using the mass profile and the Eddington distribution function.

        Parameters:
        ----------
        r_bins: array
            Array of the radius bins.
        rho_bins: array
            Array of the density bins.
        mass_bins: array
            Array of the mass bins.
        total_phi_bins: array
            Array of the total potential bins.
        eps_bins: array
            Array of the eps bins.
        f_eps_bins: array
            Array of the distribution function bins.
        r_sample_min: float
            Minimum radius to sample particles from.
        r_sample_max: float
            Maximum radius to sample particles from.
        N_part: int
            Number of particles to sample from the halo.
        kappa_L: float
            Angular momentum parameter.
        seed: int
            Random seed for reproducibility.

        Attributes:
        ----------
        part_x: array
            Array of particle x-coordinates.
        part_y: array
            Array of particle y-coordinates.
        part_z: array
            Array of particle z-coordinates.
        part_vx: array
            Array of particle x-velocities.
        part_vy: array
            Array of particle y-velocities.
        part_vz: array
            Array of particle z-velocities.
        part_mass: float
            Mass of each particle.
        """

        self._r_sample_min = r_sample_min
        self._r_sample_max = r_sample_max

        self._check_sampling_range(r_bins)

        self._N_part = N_part

        self._seed = seed
        np.random.seed(seed)  # noqa: NPY002

        (
            self.log_log_mass_interp,
            self.log_log_inverse_mass_interp,
            self.lin_log_eddington_interp,
        ) = self._get_profiles_interp(r_bins, mass_bins, eps_bins, f_eps_bins)

        self.part_mass = self._get_particle_mass()

        _part_r, self.part_x, self.part_y, self.part_z = (
            self._sample_particle_positions()
        )

        self.part_vx, self.part_vy, self.part_vz = self._sample_particle_velocities(
            _part_r, r_bins, rho_bins, total_phi_bins
        )

        self._kappa_L = kappa_L

        if kappa_L > 0:
            self.part_vx, self.part_vy = self._apply_angular_momentum(
                self.part_vx, self.part_vy, self.part_x, self.part_y
            )

    def _check_sampling_range(self, r_bins):
        """
        Check if the sampling range is within the profile range.
        """

        if self._r_sample_min < r_bins[0]:
            raise ValueError(  # noqa: TRY003
                "The minimum sampling radius is less than the minimum profile radius."  # noqa: EM101
            )
        if self._r_sample_max > r_bins[-1]:
            raise ValueError(  # noqa: TRY003
                "The maximum sampling radius is greater than the maximum profile radius."  # noqa: EM101
            )

    def _get_profiles_interp(self, r_bins, mass_bins, eps_bins, f_eps_bins):
        """
        Get the interpolated profiles of the halo.

        Returns:
        -------
        _log_log_mass_interp: CubicSpline
            Interpolated mass profile of the halo in log-log x-y space.
        _log_log_inverse_mass_interp: CubicSpline
            Inverse interpolated mass profile of the halo in log-log x-y space.
        _lin_log_eddington_interp: CubicSpline
            Interpolated Eddington distribution of the halo in linear-log x-y space.
        """

        _log_log_mass_interp = get_interp(np.log(r_bins), np.log(mass_bins))
        _log_log_inverse_mass_interp = get_interp(np.log(mass_bins), np.log(r_bins))

        _lin_log_eddington_interp = get_interp(eps_bins, np.log(f_eps_bins))

        return (
            _log_log_mass_interp,
            _log_log_inverse_mass_interp,
            _lin_log_eddington_interp,
        )

    def _get_particle_mass(self):
        """
        Get the mass of the particles in the halo.

        Returns:
        -------
        particle_mass: float
            Mass of the particles.
        """
        return (
            np.exp(self.log_log_mass_interp(np.log(self._r_sample_max)))
            - np.exp(self.log_log_mass_interp(np.log(self._r_sample_min)))
        ) / self._N_part

    def _sample_particle_positions(self):
        """
        Sample the positions of the particles in the halo using the mass profile and the inversion sampling method.

        Returns:
        -------
        part_r: array
            Array of particle radii.
        part_x: array
            Array of particle x-coordinates.
        part_y: array
            Array of particle y-coordinates.
        part_z: array
            Array of particle z-coordinates.
        """

        _part_phi = 2 * np.pi * np.random.uniform(0, 1, self._N_part).astype(np.float64)  # noqa: NPY002
        _part_theta = np.arcsin(
            2 * np.random.uniform(0, 1, self._N_part).astype(np.float64) - 1  # noqa: NPY002
        )

        part_r = np.exp(
            self.log_log_inverse_mass_interp(
                np.log(
                    np.random.uniform(  # noqa: NPY002
                        np.exp(self.log_log_mass_interp(np.log(self._r_sample_min))),
                        np.exp(self.log_log_mass_interp(np.log(self._r_sample_max))),
                        self._N_part,
                    ).astype(np.float64)
                )
            )
        )

        return (
            part_r,
            part_r * np.cos(_part_theta) * np.cos(_part_phi),
            part_r * np.cos(_part_theta) * np.sin(_part_phi),
            part_r * np.sin(_part_theta),
        )

    def _sample_particle_velocities(self, part_r, r_bins, rho_bins, total_phi_bins):
        """
        Sample the velocities of the particles in the halo using inversion sampling.

        Parameters:
        ----------
        part_r: array
            Array of particle radii.
        r_bins: array
            Array of the radius bins.
        rho_bins: array
            Array of the density bins.
        total_phi_bins: array
            Array of the total potential bins.

        Returns:
        -------
        part_vx: array
            Array of particle x-velocities.
        part_vy: array
            Array of particle y-velocities.
        part_vz: array
            Array of particle z-velocities.
        """

        _interp_rho_bins = np.exp((np.log(rho_bins[:-1]) + np.log(rho_bins[1:])) / 2)
        _interp_total_psi_bins = -(total_phi_bins[:-1] + total_phi_bins[1:]) / 2
        _interp_vmax_bins = np.sqrt(2 * _interp_total_psi_bins)

        _bin_indices = np.digitize(part_r, r_bins) - 1

        _part_v = np.zeros(self._N_part, dtype=np.float64)

        _N_bins = len(r_bins)

        for i in tqdm(range(_N_bins - 1), desc="Sampling velocities"):
            _bin_mask = _bin_indices == i

            if np.sum(_bin_mask) == 0:
                continue

            _v_bins = np.linspace(0, _interp_vmax_bins[i], _N_bins, dtype=np.float64)

            _p_v_bins = (
                4
                * np.pi
                * _v_bins**2
                * np.exp(
                    self.lin_log_eddington_interp(
                        _interp_total_psi_bins[i] - _v_bins**2 / 2
                    )
                )
                / _interp_rho_bins[i]
            )

            _P_v_bins = cumulative_trapezoid(_p_v_bins, _v_bins, initial=0)
            _P_v_mask = np.logical_and(np.isfinite(_P_v_bins), _P_v_bins <= 1)

            _lin_lin_inverse_P_v_interp = get_interp(
                _P_v_bins[_P_v_mask] / _P_v_bins[_P_v_mask][-1], _v_bins[_P_v_mask]
            )

            _part_v[_bin_mask] = _lin_lin_inverse_P_v_interp(
                np.random.uniform(0, 1, np.sum(_bin_mask)).astype(np.float64)  # noqa: NPY002
            )

        part_v_phi = (
            2 * np.pi * np.random.uniform(0, 1, self._N_part).astype(np.float64)  # noqa: NPY002
        )
        part_v_theta = np.arcsin(
            2 * np.random.uniform(0, 1, self._N_part).astype(np.float64) - 1  # noqa: NPY002
        )

        return (
            _part_v * np.cos(part_v_theta) * np.cos(part_v_phi),
            _part_v * np.cos(part_v_theta) * np.sin(part_v_phi),
            _part_v * np.sin(part_v_theta),
        )

    def _apply_angular_momentum(self, part_vx, part_vy, part_x, part_y):
        """
        Apply the angular momentum to halo via flipping the velocities of a fraction of the retrograde particles.

        Parameters:
        ----------
        part_vx: array
            Array of particle x-velocities.
        part_vy: array
            Array of particle y-velocities.
        part_x: array
            Array of particle x-coordinates.
        part_y: array
            Array of particle y-coordinates.

        Returns:
        -------
        part_vx: array
            Array of particle x-velocities after applying angular momentum.
        part_vy: array
            Array of particle y-velocities after applying angular momentum.
        """

        _J_z = part_x * part_vy - part_y * part_vx

        _retrograde_mask = _J_z < 0
        _flip_mask = (
            np.random.uniform(0, 1, self._N_part).astype(np.float64) < self._kappa_L  # noqa: NPY002
        )

        _L_z_modifier = np.ones(self._N_part, dtype=np.float64)
        _L_z_modifier[np.logical_and(_retrograde_mask, _flip_mask)] = -1

        return part_vx * _L_z_modifier, part_vy * _L_z_modifier
