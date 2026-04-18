import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from tqdm import tqdm

from softisoics.utils import get_interp


class BaseSingleProfile:
    def __init__(self, r_bin_min, r_bin_max, N_bins, halo_edge=None, epsilons=[0]):  # noqa: B006
        """
        Initialize the profile.

        Parameters:
        ----------
        r_bin_min: float
            Minimum radius of the profile.
        r_bin_max: float
            Maximum radius of the profile.
        N_bins: int
            Number of bins for the profile.
        halo_edge: float, optional
            Radius of the halo edge. If None, it is set to r_bin_max.
        epsilons: list of float, optional
            List of softening lengths. If empty, no softening is applied.

        Attributes:
        ----------
        r_bins: array
            Array of the radius bins.
        rho_bins: array
            Array of the density bins.
        mass_bins: array
            Array of the mass bins.
        phi_bins: array
            Array of the potential bins.
        conv_phi_bins: array
            Array of the convoluted potential bins.
        """

        self._r_bin_min = r_bin_min
        self._r_bin_max = r_bin_max

        self._halo_edge = halo_edge or r_bin_max

        self._N_bins = N_bins

        self.epsilons = epsilons

        self.r_bins = np.logspace(
            np.log10(self._r_bin_min),
            np.log10(self._r_bin_max),
            self._N_bins,
            dtype=np.float64,
        )
        self.rho_bins = self._get_rho_bins(self.r_bins)
        self.mass_bins = self._get_mass_bins(self.r_bins, self.rho_bins)
        self.phi_bins = self._get_phi_bins(self.r_bins, self.mass_bins)

        self.conv_phi_bins = {}

        for epsilon in self.epsilons:
            _conv_rho_bins = self._get_convoluted_rho_bins(self.r_bins, epsilon)
            _conv_mass_bins = self._get_mass_bins(self.r_bins, _conv_rho_bins)
            self.conv_phi_bins[epsilon] = self._get_phi_bins(
                self.r_bins, _conv_mass_bins
            )

    def _get_rho_bins(self, r_bins):
        """
        Get the density profile. Must be implemented in the subclass.

        Parameters:
        ----------
        r_bins: array
            Array of the radius bins.

        Returns:
        -------
        rho_bins: array
            Array of the density bins.
        """

        raise NotImplementedError("Not implemented in base class.")  # noqa: EM101

    def _get_convoluted_rho_bins(self, r_bins, epsilon):
        """
        Get the convoluted density profile.

        Parameters:
        ----------
        r_bins: array
            Array of the radius bins.
        epsilon: float
            Softening length.

        Returns:
        -------
        conv_rho_bins: array
            Array of the convoluted density bins.
        """
        _h = epsilon * 2.8

        if _h == 0:
            return self._get_rho_bins(r_bins)

        def _A(u):
            return 1 / 2 * u**2 - 3 / 2 * u**4 + 6 / 5 * u**5

        def _B(u):
            return u**2 - 2 * u**3 + 3 / 2 * u**4 - 2 / 5 * u**5

        def _I(u1, u2):
            if u2 <= 1 / 2:
                return _A(u2) - _A(u1)
            if u1 > 1 / 2:
                return _B(u2) - _B(u1)
            return _A(1 / 2) - _A(u1) + _B(u2) - _B(1 / 2)

        _I = np.vectorize(_I)

        _conv_rho_bins = np.zeros_like(r_bins)

        for i in tqdm(range(r_bins.shape[0]), desc="Softening density:"):
            _r = r_bins[i]

            if _r < self._halo_edge:
                _shell_r_bin_edges = np.linspace(
                    max(0, _r - _h), _r + _h, self._N_bins + 1
                )
                _shell_r_bins = (_shell_r_bin_edges[1:] + _shell_r_bin_edges[:-1]) / 2
                _shell_mass_bins = (
                    4
                    * np.pi
                    / 3
                    * (_shell_r_bin_edges[1:] ** 3 - _shell_r_bin_edges[:-1] ** 3)
                    * self._get_rho_bins(_shell_r_bins)
                )

                _u_low = np.abs(_r - _shell_r_bins) / _h
                _u_high = (_r + _shell_r_bins) / _h
                _u_high[_u_high > 1] = 1

                _conv_rho_bins[i] = np.sum(
                    4
                    * _shell_mass_bins
                    / np.pi
                    / _h
                    / _r
                    / _shell_r_bins
                    * _I(_u_low, _u_high)
                )

            else:
                _conv_rho_bins[i] = self._get_rho_bins(_r)

        return _conv_rho_bins

    def _get_mass_bins(self, r_bins, rho_bins):
        """
        Get the mass profile.

        Parameters:
        ----------
        r_bins: array
            Array of the radius bins.
        rho_bins: array
            Array of the density bins.

        Returns:
        -------
        mass_bins: array
            Array of the mass bins.
        """

        _zero_mass = 4 / 3 * np.pi * r_bins[0] ** 3 * rho_bins[0]
        _mass_integrand = 4 * np.pi * r_bins**2 * rho_bins
        return cumulative_trapezoid(_mass_integrand, r_bins, initial=0) + _zero_mass

    def _get_phi_bins(self, r_bins, mass_bins):
        """
        Get the potential profile.

        Parameters:
        ----------
        r_bins: array
            Array of the radius bins.
        mass_bins: array
            Array of the mass bins.

        Returns:
        -------
        phi_bins: array
            Array of the potential bins.
        """
        _delta_phi_integrand = mass_bins / r_bins**2
        _delta_phi_bins = cumulative_trapezoid(_delta_phi_integrand, r_bins, initial=0)

        return _delta_phi_bins - _delta_phi_bins[-1]


class CollisionlessSingleProfile(BaseSingleProfile):
    def __init__(self, r_bin_min, r_bin_max, N_bins, halo_edge=None, epsilons=[0]):  # noqa: B006
        super().__init__(
            r_bin_min=r_bin_min,
            r_bin_max=r_bin_max,
            N_bins=N_bins,
            halo_edge=halo_edge,
            epsilons=epsilons,
        )

        """
        Initialize the profile.

        Parameters:
        ----------
        r_bin_min: float
            Minimum radius of the profile.
        r_bin_max: float
            Maximum radius of the profile.
        N_bins: int
            Number of bins for the profile.
        halo_edge: float, optional
            Radius of the halo edge. If None, it is set to r_bin_max.
        epsilons: list of float, optional
            List of softening lengths. If empty, no softening is applied.

        Attributes:
        ----------
        r_bins: array
            Array of the radius bins.
        rho_bins: array
            Array of the density bins.
        mass_bins: array
            Array of the mass bins.
        phi_bins: array
            Array of the potential bins.
        conv_phi_bins: array
            Array of the convoluted potential bins.
        """

    def get_potential_dependent_profiles(self, total_phi_bins):
        """
        Get all potential-dependent profiles.

        Parameters:
        ----------
        total_phi_bins: array
            Array of the total potential bins.

        Attributes:
        -------
        total_phi_bins: array
            Array of the total potential bins.
        sigma_r_bins: array
            Array of the radial velocity dispersion bins.
        L_err_scaler: float
            Scaler for the angular momentum dispersion of the halo.
        L_max: float
            The maximum angular momentum for the halo.
        eps_bins: array
            Array of the eps bins, where eps = -total_phi.
        f_eps_bins: array
            Array of the distribution function bins.
        """
        self.total_phi_bins = total_phi_bins

        self.sigma_r_bins = self.get_sigma_r_bins(
            self.r_bins, self.rho_bins, total_phi_bins
        )

        self.L_err_scaler = self.get_L_err_scaler(
            self.r_bins, self.rho_bins, self.sigma_r_bins
        )
        self.L_max = self.get_L_max(self.r_bins, self.rho_bins, self.sigma_r_bins)

        self.eps_bins, self.f_eps_bins = self.get_Eddington_bins(
            self.rho_bins, total_phi_bins
        )

    def _get_beta_bins(self, r_bins):
        """
        Get the velocity anisotropy profile. The default is isotropic, i.e., beta=0.
        Anisotropic profiles can be implemented in the subclass by overriding this method.

        Parameters:
        ----------
        r_bins: array
            Array of the radius bins.

        Returns:
        -------
        beta_bins: array
            Array of the velocity anisotropy bins.
        """
        return np.zeros_like(r_bins)

    def get_sigma_r_bins(self, r_bins, rho_bins, total_phi_bins):
        """
        Get the radial velocity dispersion profile.

        Parameters:
        ----------
        r_bins: array
            Array of the radius bins.
        rho_bins: array
            Array of the density bins.
        total_phi_bins: array
            Array of the total potential bins.

        Returns:
        -------
        sigma_r_bins: array
            Array of the radial velocity dispersion bins.
        """

        _beta_bins = self._get_beta_bins(r_bins)
        _ln_fbeta_bins = 2 * _beta_bins / r_bins
        _fbeta_bins = np.exp(cumulative_trapezoid(_ln_fbeta_bins, r_bins, initial=0))

        _delta_fbeta_rho_sigma_r_sqr_integrand = (
            _fbeta_bins
            * rho_bins
            * np.gradient(total_phi_bins, np.log(r_bins))
            / r_bins
        )
        _delta_fbeta_rho_sigma_r_sqr_bins = cumulative_trapezoid(
            _delta_fbeta_rho_sigma_r_sqr_integrand, r_bins, initial=0
        )
        return np.sqrt(
            (_delta_fbeta_rho_sigma_r_sqr_bins[-1] - _delta_fbeta_rho_sigma_r_sqr_bins)
            / (_fbeta_bins * rho_bins)
        )

    def get_L_err_scaler(self, r_bins, rho_bins, sigma_r_bins):
        """
        Calculate the scaler for the angular momentum dispersion of the halo.
        The actual angular momentum dispersion is given by L_err = sqrt(m) * L_err_scaler.
        """

        _L_err_scaler_sqr_integrand = (
            8 * np.pi / 3 * r_bins**4 * rho_bins * sigma_r_bins**2
        )

        return np.sqrt(np.trapezoid(_L_err_scaler_sqr_integrand, r_bins))

    def get_L_max(self, r_bins, rho_bins, sigma_r_bins):
        """
        Calculate the maximum angular momentum of the halo.

        Parameters:
        ----------
        r_bins: array
            Array of the radius bins.
        rho_bins: array
            Array of the density bins.
        sigma_r_bins: array
            Array of the radial velocity dispersion bins.

        Returns:
        -------
        L_max: float
            The maximum angular momentum for the halo.
        """

        _L_max_integrand = (
            np.sqrt(2 / np.pi) * np.pi**2 * r_bins**3 * rho_bins * sigma_r_bins
        )

        return np.trapezoid(_L_max_integrand, r_bins)

    def get_Eddington_bins(self, rho_bins, total_phi_bins):
        """
        Perform Eddington's inversion to get the distribution function bins.

        Parameters:
        ----------
        rho_bins: array
            Array of the density bins.
        total_phi_bins: array
            Array of the total potential bins.

        Returns:
        -------
        eps_bins: array
            Array of the eps bins, where eps = -total_phi.
        f_eps_bins: array
            Array of the distribution function bins.
        """

        _psi_bins = np.flip(-total_phi_bins)
        _d2rho_dpsi2_bins = np.flip(
            rho_bins
            * (
                np.gradient(np.log(rho_bins), total_phi_bins) ** 2
                + np.gradient(
                    np.gradient(np.log(rho_bins), total_phi_bins), total_phi_bins
                )
            )
        )

        _lin_log_d2rho_dpsi2_interp = get_interp(
            _psi_bins[1:], np.log(_d2rho_dpsi2_bins[1:])
        )

        _eps_bins = _psi_bins
        _f_eps_bins = [0]
        for i in tqdm(range(1, len(total_phi_bins)), desc="Eddington's inversion"):

            def _f_eps_integrand(psi):
                return (
                    1
                    / np.sqrt(_psi_bins[i] - psi)  # noqa: B023
                    * np.exp(_lin_log_d2rho_dpsi2_interp(psi))
                )

            _f_eps_bins.append(
                quad(
                    _f_eps_integrand,
                    0,
                    _psi_bins[i],
                )[0]
            )

        return _eps_bins, 1 / np.sqrt(8) / np.pi**2 * np.array(_f_eps_bins)

    def reconstruct_density(self, total_phi_bins, eps_bins, f_eps_bins):
        """
        Reconstruct the density profile from the Eddington distribution function to check
        for self-consistency. The reconstructed density should match the original density profile.

        Parameters:
        ----------
        total_phi_bins: array
            Array of the total potential bins.
        eps_bins: array
            Array of the eps bins, where eps = -total_phi.
        f_eps_bins: array
            Array of the distribution function bins.

        Returns:
        -------
        reconstructed_rho_bins: array
            Array of the reconstructed density bins.
        """

        _lin_log_eddington_interp = get_interp(eps_bins, np.log(f_eps_bins))

        def _rho_integrand(v, psi):
            return 4 * np.pi * v**2 * np.exp(_lin_log_eddington_interp(psi - v**2 / 2))

        _reconstructed_rho_bins = []
        for phi in tqdm(total_phi_bins, desc="Reconstructing densities"):
            _reconstructed_rho_bins.append(  # noqa: PERF401
                quad(_rho_integrand, 0, np.sqrt(-2 * phi), args=(-phi,))[0]
            )

        return np.array(_reconstructed_rho_bins)


class CollisionalSingleProfile(BaseSingleProfile):
    def __init__(self, r_bin_min, r_bin_max, N_bins, halo_edge=None, epsilons=[0]):  # noqa: B006
        super().__init__(
            r_bin_min=r_bin_min,
            r_bin_max=r_bin_max,
            N_bins=N_bins,
            halo_edge=halo_edge,
            epsilons=epsilons,
        )

        """
        Initialize the profile.

        Parameters:
        ----------
        r_bin_min: float
            Minimum radius of the profile.
        r_bin_max: float
            Maximum radius of the profile.
        N_bins: int
            Number of bins for the profile.
        halo_edge: float, optional
            Radius of the halo edge. If None, it is set to r_bin_max.
        epsilons: list of float, optional
            List of softening lengths. If empty, no softening is applied.

        Attributes:
        ----------
        r_bins: array
            Array of the radius bins.
        rho_bins: array
            Array of the density bins.
        mass_bins: array
            Array of the mass bins.
        phi_bins: array
            Array of the potential bins.
        conv_phi_bins: array
            Array of the convoluted potential bins.
        """

    def get_potential_dependent_profiles(self, total_phi_bins, gamma=5 / 3):
        """
        Get all potential-dependent profiles.

        Parameters:
        ----------
        total_phi_bins: array
            Array of the total potential bins.
        gamma: float
            Adiabatic index for the gas.

        Attributes:
        -------
        total_phi_bins: array
            Array of the total potential bins.
        gamma: float
            Adiabatic index for the gas.
        pressure_bins: array
            Array of the pressure bins.
        internal_energy_bins: array
            Array of the internal energy bins.
        """
        self.total_phi_bins = total_phi_bins
        self.gamma = gamma

        self.pressure_bins = self._get_pressure_bins(
            self.r_bins, self.rho_bins, total_phi_bins
        )

        self.internal_energy_bins = self._get_internal_energy_bins(
            self.rho_bins, self.pressure_bins, gamma
        )

    def _get_pressure_bins(self, r_bins, rho_bins, total_phi_bins):
        """
        Get the pressure profile via the hydrostatic equilibrium equation.

        Parameters:
        ----------
        r_bins: array
            Array of the radius bins.
        rho_bins: array
            Array of the density bins.
        total_phi_bins: array
            Array of the total potential bins.

        Returns:
        -------
        pressure_bins: array
            Array of the pressure bins.
        """
        _delta_pressure_integrand = (
            rho_bins * np.gradient(total_phi_bins, np.log(r_bins)) / r_bins
        )
        _delta_pressure_bins = cumulative_trapezoid(
            _delta_pressure_integrand, r_bins, initial=0
        )

        return _delta_pressure_bins[-1] - _delta_pressure_bins

    def _get_internal_energy_bins(self, rho_bins, pressure_bins, gamma):
        """
        Get the internal energy profile from the pressure profile.

        Parameters:
        ----------
        rho_bins: array
            Array of the density bins.
        pressure_bins: array
            Array of the pressure bins.

        Returns:
        -------
        internal_energy_bins: array
            Array of the internal energy bins.
        """
        return pressure_bins / (gamma - 1) / rho_bins
