import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

from softisoics.utils import get_interp


class GenerateMesh:
    def __init__(
        self,
        r_bins,
        rho_bins,
        mass_bins,
        internal_energy_bins,
        r_sample_max,
        N_cell,
        random_rotation=True,
        seed=42,
    ):
        """
        Generate the mesh for the initial conditions of a spherical gas halo/bulge.

        The gas inside r_sample_max is split into a central cell plus concentric shells
        whose thickness matches the angular spacing of the cells inside them
        (cells with aspect ratio ~1). Each shell is populated with a Fibonacci lattice.

        Parameters:
        ----------
        r_bins: array
            Array of the radius bins.
        rho_bins: array
            Array of the density bins.
        mass_bins: array
            Array of the mass bins.
        internal_energy_bins: array
            Array of the (specific) internal energy bins.
        r_sample_max: float
            Maximum radius of the gas mesh. Only the gas inside r_sample_max is meshed.
        N_cell: int
            Number of cells to generate (including the central cell).
        random_rotation: bool, optional
            Randomly rotate the Fibonacci lattice of each shell, so that the lattice
            poles and seams of different shells are not aligned along the z-axis.
        seed: int, optional
            Random seed for the shell rotations.

        Attributes:
        ----------
        mesh_coords: array
            Array of the mesh-generating-point coordinates.
        mesh_velocs: array
            Array of the mesh velocities.
        mesh_masses: array
            Array of the mesh masses.
        mesh_internal_energies: array
            Array of the mesh specific internal energies.
        """

        if r_sample_max > r_bins[-1]:
            raise ValueError(  # noqa: TRY003
                "The maximum sampling radius is greater than the maximum profile radius."  # noqa: EM101
            )

        self._r_sample_max = r_sample_max
        self._N_cell = int(N_cell)
        self._rng = np.random.default_rng(seed)

        (
            self._log_log_rho_interp,
            self._log_log_mass_interp,
            self._log_log_inverse_mass_interp,
            self._log_log_accum_internal_energy_interp,
        ) = self._get_profile_interp(r_bins, rho_bins, mass_bins, internal_energy_bins)

        # Only the gas inside r_sample_max is meshed, so the cell mass is set by
        # M(<r_sample_max), not by the mass out to the edge of the profile grid.
        self._M_total = self._M(r_sample_max)
        self._m_cell = self._M_total / self._N_cell

        self._shell_boundaries, self._num_cell_per_shell = self._get_shell_boundaries()
        self._shell_lower_boundaries = self._shell_boundaries[:-1]

        self._r_shell_points = self._get_shell_point_radii(self._shell_boundaries)

        self.mesh_coords = self._get_mesh_generating_points(
            self._r_shell_points, self._num_cell_per_shell, random_rotation
        )
        self.mesh_velocs = self._get_mesh_velocities()

        (
            self.mesh_masses,
            self.mesh_internal_energies,
        ) = self._get_mesh_masses_and_internal_energies(
            self._r_shell_points, self._num_cell_per_shell
        )

    # ------------------------------------------------------------------ profiles

    def _M(self, r):
        return np.exp(self._log_log_mass_interp(np.log(r)))

    def _E(self, r):
        return np.exp(self._log_log_accum_internal_energy_interp(np.log(r)))

    def _get_profile_interp(self, r_bins, rho_bins, mass_bins, internal_energy_bins):
        """
        Get the interpolated profiles of the gas.

        Returns:
        -------
        log_log_rho_interp, log_log_mass_interp, log_log_inverse_mass_interp,
        log_log_accum_internal_energy_interp: CubicSpline
            Interpolated log-log density, enclosed mass, inverse enclosed mass and
            enclosed internal energy E(<r) = int 4 pi r^2 rho u dr.
        """

        log_log_rho_interp = get_interp(np.log(r_bins), np.log(rho_bins))

        log_log_mass_interp = get_interp(np.log(r_bins), np.log(mass_bins))
        log_log_inverse_mass_interp = get_interp(np.log(mass_bins), np.log(r_bins))

        # Same inner-boundary treatment as mass_bins: add the energy inside r_bins[0]
        # so that E(<r) and M(<r) are consistent (and log(E) is finite everywhere).
        _zero_internal_energy = mass_bins[0] * internal_energy_bins[0]
        _accum_internal_energy_integrand = (
            4 * np.pi * r_bins**2 * rho_bins * internal_energy_bins
        )
        _accum_internal_energy_bins = (
            cumulative_trapezoid(_accum_internal_energy_integrand, r_bins, initial=0)
            + _zero_internal_energy
        )

        log_log_accum_internal_energy_interp = get_interp(
            np.log(r_bins), np.log(_accum_internal_energy_bins)
        )

        return (
            log_log_rho_interp,
            log_log_mass_interp,
            log_log_inverse_mass_interp,
            log_log_accum_internal_energy_interp,
        )

    # -------------------------------------------------------------------- shells

    def _get_r_out(self, r_in):
        """
        Solve for the outer radius of a shell starting at r_in, such that the shell
        thickness equals the mean angular spacing of its cells:
            (r_out - r_in)^2 * N_shell = 4 pi r_mid^2,  N_shell = dM / m_cell.
        """
        _M_in = self._M(r_in)

        def _f(r_out):
            _r_mid = 0.5 * (r_in + r_out)
            return (r_out - r_in) ** 2 * (
                self._M(r_out) - _M_in
            ) - 4 * np.pi * _r_mid**2 * self._m_cell

        _r_hi = r_in + (
            self._m_cell / np.exp(self._log_log_rho_interp(np.log(r_in)))
        ) ** (1 / 3)
        while _f(_r_hi) < 0:
            _r_hi = r_in + 2 * (_r_hi - r_in)
            if _r_hi > 2 * self._r_sample_max:
                return _r_hi

        return brentq(_f, r_in * (1 + 1e-12), _r_hi, xtol=1e-12 * r_in, rtol=1e-10)

    def _get_shell_boundaries(self):
        """
        Get the shell boundaries for the mesh generation based on the equal-mass approach.

        Returns:
        -------
        shell_boundaries: array
            Array of the shell boundaries, from the central-cell radius to r_sample_max.
        num_cell_per_shell: array
            Array of the number of cells per shell.
        """

        # Central cell holds exactly one cell mass.
        _boundaries = [np.exp(self._log_log_inverse_mass_interp(np.log(self._m_cell)))]

        while True:
            r_in = _boundaries[-1]
            r_out = self._get_r_out(r_in)
            if r_out >= self._r_sample_max:
                # Close the mesh exactly at r_sample_max. If the last shell would be
                # less than half its natural thickness, stretch the previous one instead.
                if len(_boundaries) > 1 and (self._r_sample_max - r_in) < 0.5 * (
                    r_out - r_in
                ):
                    _boundaries[-1] = self._r_sample_max
                else:
                    _boundaries.append(self._r_sample_max)
                break
            _boundaries.append(r_out)

        _boundaries = np.array(_boundaries, dtype=np.float64)

        _cum_num = np.round(self._M(_boundaries) / self._m_cell).astype(np.int64)
        _cum_num[0] = 1
        _cum_num[-1] = self._N_cell

        # Drop boundaries that would leave a shell with zero cells.
        _keep = [0]
        for k in range(1, _boundaries.shape[0] - 1):
            if _cum_num[k] > _cum_num[_keep[-1]] and _cum_num[k] < self._N_cell:
                _keep.append(k)
        _keep.append(_boundaries.shape[0] - 1)

        shell_boundaries = _boundaries[_keep]
        num_cell_per_shell = np.diff(_cum_num[_keep])

        return shell_boundaries, num_cell_per_shell

    def _get_shell_point_radii(self, shell_boundaries):
        """
        Radius of the mesh-generating points of each shell: the mass midpoint of the
        shell (equal to the volume midpoint for thin shells, but robust for thick
        outer shells where the density drops steeply). Each shell gets its own radius inside the shell, independently of the
        other shells (the previous recursive reflection r_k = 2 r_in - r_(k-1)
        oscillated from shell to shell and could leave a shell entirely).
        """
        _r_in = shell_boundaries[:-1]
        _r_out = shell_boundaries[1:]
        r_shell_points = np.exp(
            self._log_log_inverse_mass_interp(
                np.log(0.5 * (self._M(_r_in) + self._M(_r_out)))
            )
        )
        r_shell_points = np.clip(r_shell_points, _r_in, _r_out)

        # The central cell's Voronoi face sits at r_shell_points[0] / 2. Put it at the
        # central-cell radius (so the central cell holds one cell mass) when possible.
        if 2 * _r_in[0] < _r_out[0]:
            r_shell_points[0] = 2 * _r_in[0]

        return r_shell_points

    # -------------------------------------------------------------------- points

    @staticmethod
    def _fibonacci_sphere(n):
        _k = np.arange(n, dtype=np.float64)
        _z = 1 - (2 * _k + 1) / n
        _phi = 2 * np.pi / ((1 + np.sqrt(5)) / 2) * _k
        _s = np.sqrt(1 - _z**2)
        return np.vstack([_s * np.cos(_phi), _s * np.sin(_phi), _z]).T

    def _get_mesh_generating_points(
        self, r_shell_points, num_cell_per_shell, random_rotation=True
    ):
        """
        Get the mesh generating points using the Fibonacci lattice method.

        Returns:
        -------
        mesh_coords: array
            Array of the mesh coordinates (central cell first, then shell by shell).
        """

        _coords = [np.zeros((1, 3), dtype=np.float64)]
        _rotations = (
            Rotation.random(num_cell_per_shell.shape[0], random_state=self._rng)
            if random_rotation
            else None
        )
        for i in range(num_cell_per_shell.shape[0]):
            _unit = self._fibonacci_sphere(num_cell_per_shell[i])
            if _rotations is not None:
                _unit = _rotations[i].apply(_unit)
            _coords.append(r_shell_points[i] * _unit)

        self._shell_rotations = _rotations

        return np.concatenate(_coords).astype(np.float64)

    def _get_mesh_velocities(self):
        """
        Get the mesh velocities, assuming no rotations.
        """

        return np.zeros((self._N_cell, 3), dtype=np.float64)

    # ------------------------------------------------------ masses and energies

    def _get_effective_faces(self, r_shell_points):
        """
        Radial positions of the Voronoi faces between consecutive shells: the
        bisector between the point radii (r_0 = 0 for the central cell). The outer
        face of the last shell is r_sample_max.
        """
        _r_points = np.append([0.0], r_shell_points)
        return np.append(0.5 * (_r_points[:-1] + _r_points[1:]), self._r_sample_max)

    def _get_mesh_masses_and_internal_energies(
        self, r_shell_points, num_cell_per_shell
    ):
        """
        Assign masses and specific internal energies from the gas actually enclosed by
        each cell's Voronoi faces (not by the nominal shell boundaries), so that
        mass/volume is the target density and u is the mass-weighted mean energy.

        Returns:
        -------
        mesh_masses: array
            Array of the mesh masses.
        mesh_internal_energies: array
            Array of the mesh specific internal energies.
        """

        _faces = self._get_effective_faces(r_shell_points)  # central + n_shell faces

        _M_faces = self._M(_faces)
        _E_faces = self._E(_faces)

        _cell_M = np.append(_M_faces[0], np.diff(_M_faces))  # per shell (incl. centre)
        _cell_E = np.append(_E_faces[0], np.diff(_E_faces))

        if np.any(_cell_M <= 0) or np.any(_cell_E <= 0):
            raise ValueError(  # noqa: TRY003
                "Empty mesh shell; check the profile and r_sample_max."  # noqa: EM101
            )
        _count = np.append([1], num_cell_per_shell)

        mesh_masses = np.repeat(_cell_M / _count, _count)
        mesh_internal_energies = np.repeat(_cell_E / _cell_M, _count)

        return mesh_masses, mesh_internal_energies
