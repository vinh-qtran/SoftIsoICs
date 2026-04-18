import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import fsolve

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
    ):
        """
        Generate the mesh for the initial conditions of spherical gas bulge.

        Parameters:
        ----------
        r_bins: array
            Array of the radius bins.
        rho_bins: array
            Array of the density bins.
        mass_bins: array
            Array of the mass bins.
        internal_energy_bins: array
            Array of the internal energy bins.
        r_sample_max: float
            Maximum radius to sample particles from.
        N_cell: int
            Number of cells to generate.

        Attributes:
        ----------
        mesh_coords: array
            Array of the mesh coordinates.
        mesh_velocs: array
            Array of the mesh velocities.
        mesh_masses: array
            Array of the mesh masses.
        mesh_internal_energies: array
            Array of the mesh internal energies.
        """

        self._r_sample_max = r_sample_max

        self._N_cell = N_cell
        self._M_total = mass_bins[-1]

        self._m_cell = self._M_total / self._N_cell

        (
            self._log_log_rho_interp,
            self._log_log_mass_interp,
            self._log_log_inverse_mass_interp,
            self._log_log_accum_internal_energy_interp,
        ) = self._get_profile_interp(r_bins, rho_bins, mass_bins, internal_energy_bins)

        self._total_accum_internal_energy = np.exp(
            self._log_log_accum_internal_energy_interp(np.log(r_bins[-1]))
        )

        self._shell_lower_boundaries, self._num_cell_per_shell = (
            self._get_shell_boundaries()
        )

        self.mesh_coords = self._get_mesh_generating_points(
            self._shell_lower_boundaries, self._num_cell_per_shell
        )
        self.mesh_velocs = self._get_mesh_velocities()
        self.mesh_masses = self._get_mesh_masses(
            self._shell_lower_boundaries, self._num_cell_per_shell
        )
        self.mesh_internal_energies = self._get_mesh_internal_energies(
            self._shell_lower_boundaries, self._num_cell_per_shell, self.mesh_masses
        )

    def _get_profile_interp(self, r_bins, rho_bins, mass_bins, internal_energy_bins):
        """
        Get the interpolated profiles of the bulge.

        Parameters:
        ----------
        r_bins: array
            Array of the radius bins.
        rho_bins: array
            Array of the density bins.
        mass_bins: array
            Array of the mass bins.
        internal_energy_bins: array
            Array of the internal energy bins.

        Returns:
        -------
        log_log_rho_interp: function
            Interpolated function of log-log density profile.
        log_log_mass_interp: function
            Interpolated function of log-log mass profile.
        log_log_inverse_mass_interp: function
            Interpolated function of log-log inverse mass profile.
        log_log_accum_internal_energy_interp: function
            Interpolated function of log-log accumulated internal energy profile.
        """

        log_log_rho_interp = get_interp(np.log(r_bins), np.log(rho_bins))

        log_log_mass_interp = get_interp(np.log(r_bins), np.log(mass_bins))
        log_log_inverse_mass_interp = get_interp(np.log(mass_bins), np.log(r_bins))

        _accum_internal_energy_integrand = (
            4 * np.pi * r_bins**2 * rho_bins * internal_energy_bins
        )
        _accum_internal_energy_bins = cumulative_trapezoid(
            _accum_internal_energy_integrand, r_bins, initial=0
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

    def _get_shell_boundaries(self):
        """
        Get the shell boundaries for the mesh generation basing on the equal-mass approach.

        Returns:
        -------
        shell_lower_boundaries: array
            Array of the shell lower boundaries.
        num_cell_per_shell: array
            Array of the number of cells per shell.
        """

        def _get_r_out(r_in):
            _M_in = np.exp(self._log_log_mass_interp(np.log(r_in)))

            def _f(r_out):
                if r_out > r_in:
                    return 1e10

                _M_out = np.exp(self._log_log_mass_interp(np.log(r_out)))
                _shell_num_cell = (_M_out - _M_in) / self._m_cell

                _r_mid = (r_in + r_out) / 2

                _delta_r = np.sqrt(4 * np.pi * _r_mid**2 / _shell_num_cell)

                return (r_out - r_in) - _delta_r

            _delta_r_guess = (
                self._m_cell / np.exp(self._log_log_rho_interp(np.log(r_in)))
            ) ** (1 / 3)
            _r_out_guess = r_in + _delta_r_guess

            return fsolve(_f, _r_out_guess)[0]

        _shell_boundaries = [
            np.exp(self._log_log_inverse_mass_interp(np.log(self._m_cell)))
        ]
        while _shell_boundaries[-1] < self._r_sample_max:
            r_in = _shell_boundaries[-1]
            r_out = _get_r_out(r_in)
            _shell_boundaries.append(r_out)

        _shell_boundaries = np.array(_shell_boundaries, dtype=np.float64)

        shell_lower_boundaries = _shell_boundaries[:-1]

        _total_cell_num = np.round(
            np.exp(self._log_log_mass_interp(np.log(_shell_boundaries))) / self._m_cell
        ).astype(np.int32)
        num_cell_per_shell = np.diff(_total_cell_num)

        num_cell_per_shell[-1] += self._N_cell - (1 + np.sum(num_cell_per_shell))

        return shell_lower_boundaries, num_cell_per_shell

    def _get_mesh_generating_points(self, shell_lower_boundaries, num_cell_per_shell):
        """
        Get the mesh generating points using the Fibonacci lattice method.

        Parameters:
        ----------
        shell_lower_boundaries: array
            Array of the shell lower boundaries.
        num_cell_per_shell: array
            Array of the number of cells per shell.

        Returns:
        -------
        mesh_coords: array
            Array of the mesh coordinates.
        """

        _r_mesh = [0]
        for i in range(shell_lower_boundaries.shape[0]):
            _r_mesh.append(
                shell_lower_boundaries[i] + (shell_lower_boundaries[i] - _r_mesh[-1])
            )

        _r_mesh = np.append([0], np.repeat(_r_mesh[1:], num_cell_per_shell)).astype(
            np.float64
        )

        _z_mesh_scaler = np.concatenate(
            [[0]]
            + [
                1 - (2 * np.arange(num_cell_per_shell[i]) + 1) / num_cell_per_shell[i]
                for i in range(num_cell_per_shell.shape[0])
            ],
            dtype=np.float64,
        )

        _fibonacci_ratio = (1 + np.sqrt(5)) / 2
        _phi_mesh = (
            2
            * np.pi
            / _fibonacci_ratio
            * np.concatenate(
                [[0]]
                + [
                    np.arange(num_cell_per_shell[i])
                    for i in range(num_cell_per_shell.shape[0])
                ],
                dtype=np.float64,
            )
        )

        _x_mesh = _r_mesh * np.sqrt(1 - _z_mesh_scaler**2) * np.cos(_phi_mesh)
        _y_mesh = _r_mesh * np.sqrt(1 - _z_mesh_scaler**2) * np.sin(_phi_mesh)
        _z_mesh = _r_mesh * _z_mesh_scaler

        return np.vstack([_x_mesh, _y_mesh, _z_mesh]).T

    def _get_mesh_velocities(self):
        """
        Get the mesh velocities, assuming no rotations.

        Returns:
        -------
        mesh_velocs: array
            Array of the mesh velocities.
        """

        return np.zeros((self._N_cell, 3), dtype=np.float64)

    def _get_mesh_masses(self, shell_lower_boundaries, num_cell_per_shell):
        """
        Get the mesh masses basing on the mass of the cells in each shell.

        Parameters:
        ----------
        shell_lower_boundaries: array
            Array of the shell lower boundaries.
        num_cell_per_shell: array
            Array of the number of cells per shell.

        Returns:
        -------
        mesh_masses: array
            Array of the mesh masses.
        """

        _shell_cell_masses = (
            np.diff(np.exp(self._log_log_mass_interp(np.log(shell_lower_boundaries))))
            / num_cell_per_shell[:-1]
        )

        m_mesh = np.zeros(self._N_cell, dtype=np.float64)

        _num_edge_cell = num_cell_per_shell[-1]

        m_mesh[0] = self._m_cell
        m_mesh[1:-_num_edge_cell] = np.repeat(
            _shell_cell_masses, num_cell_per_shell[:-1]
        )
        m_mesh[-_num_edge_cell:] = (
            self._M_total - np.sum(m_mesh[:-_num_edge_cell])
        ) / _num_edge_cell

        return m_mesh

    def _get_mesh_internal_energies(
        self, shell_lower_boundaries, num_cell_per_shell, mesh_masses
    ):
        """
        Get the mesh internal energies basing on the average internal energy of the cells in each shell.

        Parameters:
        ----------
        shell_lower_boundaries: array
            Array of the shell lower boundaries.
        num_cell_per_shell: array
            Array of the number of cells per shell.
        mesh_masses: array
            Array of the mesh masses.

        Returns:
        -------
        mesh_internal_energies: array
            Array of the mesh internal energies.
        """

        _shell_cell_internal_energies = np.diff(
            np.exp(
                self._log_log_accum_internal_energy_interp(
                    np.log(shell_lower_boundaries)
                )
            )
        ) / np.diff(np.exp(self._log_log_mass_interp(np.log(shell_lower_boundaries))))

        u_mesh = np.zeros(self._N_cell, dtype=np.float64)

        _num_edge_cell = num_cell_per_shell[-1]

        u_mesh[0] = (
            np.exp(
                self._log_log_accum_internal_energy_interp(
                    np.log(shell_lower_boundaries[0])
                )
            )
            / self._m_cell
        )
        u_mesh[1:-_num_edge_cell] = np.repeat(
            _shell_cell_internal_energies, num_cell_per_shell[:-1]
        )
        u_mesh[-_num_edge_cell:] = (
            self._total_accum_internal_energy
            - np.exp(
                self._log_log_accum_internal_energy_interp(
                    np.log(shell_lower_boundaries[-1])
                )
            )
        ) / np.sum(mesh_masses[-_num_edge_cell:])

        return u_mesh
