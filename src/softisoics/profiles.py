import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from tqdm import tqdm

from softisoics.utils import get_interp


class BaseSingleComponentProfile:
    def __init__(self, r_bin_min, r_bin_max, N_bins, halo_edge=None, epsilon=0):
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
        epsilon: float, optional
            Softening length. If 0, no softening is applied.

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

        self.epsilon = epsilon

        self.r_bins = np.logspace(
            np.log10(self._r_bin_min),
            np.log10(self._r_bin_max),
            self._N_bins,
            dtype=np.float64,
        )
        self.rho_bins = self._get_rho_bins(self.r_bins)
        self.mass_bins = self._get_mass_bins(self.r_bins, self.rho_bins)
        self.phi_bins = self._get_phi_bins(self.r_bins, self.mass_bins)

        _conv_rho_bins = self._get_convoluted_rho_bins(self.r_bins, self.epsilon)
        _conv_mass_bins = self._get_mass_bins(self.r_bins, _conv_rho_bins)
        self.conv_phi_bins = self._get_phi_bins(self.r_bins, _conv_mass_bins)

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


class BaseEddingtonDistribution:
    def __init__(self, DM_profile, gas_profile=None):
        """
        Initialize the Eddington distribution.

        Parameters:
        ----------
        DM_profile: BaseSingleComponentProfile
            The dark matter profile.
        gas_profile: BaseSingleComponentProfile, optional
            The gas profile. If None, only the dark matter profile is used.

        Attributes:
        ----------
        DM_sigma_r_bins: array
            Array of the dark matter radial velocity dispersion bins.
        DM_L_err_scaler: float
            Scaler for the dark matter angular momentum dispersion of the halo.
        DM_L_max: float
            The maximum angular momentum for the dark matter halo.
        gas_sigma_r_bins: array
            Array of the gas radial velocity dispersion bins.
        gas_L_err_scaler: float
            Scaler for the gas angular momentum dispersion of the halo.
        gas_L_max: float
            The maximum angular momentum for the gas halo.
        total_phi_bins: array
            Array of the total potential bins.
        DM_f_eps_bins: array
            Array of the dark matter distribution function bins.
        gas_f_eps_bins: array
            Array of the gas distribution function bins. If gas_profile is None, this is set to
        """

        (
            self._DM_rho_bins,
            self._DM_conv_phi_bins,
            self.DM_sigma_r_bins,
            self.DM_L_err_scaler,
            self.DM_L_max,
            self._gas_rho_bins,
            self._gas_conv_phi_bins,
            self.gas_sigma_r_bins,
            self.gas_L_err_scaler,
            self.gas_L_max,
            self.total_phi_bins,
        ) = self._read_profiles(DM_profile, gas_profile)

        self.eps_bins, self.DM_f_eps_bins = self._get_Eddington_bins(
            self._DM_rho_bins, self.total_phi_bins
        )

        if gas_profile is not None:
            _, self.gas_f_eps_bins = self._get_Eddington_bins(
                self._gas_rho_bins, self.total_phi_bins
            )
        else:
            self.gas_f_eps_bins = np.zeros_like(self.DM_f_eps_bins)

    def _read_profiles(self, DM_profile, gas_profile):
        """
        Read the profiles from the input DM and gas profiles. The DM and gas profiles must have the same r_bins.
        If gas_profile is None, the gas density and potential are set to zero. The total potential is the sum of the DM and gas potentials.

        Parameters:
        ----------
        DM_profile: BaseSingleComponentProfile
            The dark matter profile.
        gas_profile: BaseSingleComponentProfile, optional
            The gas profile. If None, only the dark matter profile is used.

        Returns:
        ----------
        DM_rho_bins: array
            Array of the dark matter density bins.
        DM_conv_phi_bins: array
            Array of the dark matter convoluted potential bins.
        DM_sigma_r_bins: array
            Array of the dark matter radial velocity dispersion bins.
        DM_L_err_scaler: float
            Scaler for the dark matter angular momentum dispersion of the halo.
        DM_L_max: float
            The maximum angular momentum for the dark matter halo.
        gas_rho_bins: array
            Array of the gas density bins.
        gas_conv_phi_bins: array
            Array of the gas convoluted potential bins.
        gas_sigma_r_bins: array
            Array of the gas radial velocity dispersion bins.
        gas_L_err_scaler: float
            Scaler for the gas angular momentum dispersion of the halo.
        gas_L_max: float
            The maximum angular momentum for the gas halo.
        total_phi_bins: array
            Array of the total potential bins.
        """

        _r_bins = DM_profile.r_bins

        DM_rho_bins = DM_profile.rho_bins
        DM_conv_phi_bins = DM_profile.conv_phi_bins

        if gas_profile is None:
            gas_rho_bins = np.zeros_like(_r_bins)
            gas_conv_phi_bins = np.zeros_like(_r_bins)
        else:
            if not np.allclose(gas_profile.r_bins, _r_bins):
                raise ValueError("DM and gas profiles must have the same r_bins.")  # noqa: EM101, TRY003

            gas_rho_bins = gas_profile.rho_bins
            gas_conv_phi_bins = gas_profile.conv_phi_bins

        total_phi_bins = DM_conv_phi_bins + gas_conv_phi_bins

        _DM_beta_bins = self._get_DM_beta_bins(_r_bins)
        DM_sigma_r_bins, DM_L_err_scaler, DM_L_max = self._get_sigma_r_related(
            _r_bins, DM_rho_bins, total_phi_bins, _DM_beta_bins
        )

        _gas_beta_bins = self._get_gas_beta_bins(_r_bins)
        gas_sigma_r_bins, gas_L_err_scaler, gas_L_max = self._get_sigma_r_related(
            _r_bins, gas_rho_bins, total_phi_bins, _gas_beta_bins
        )

        return (
            DM_rho_bins,
            DM_conv_phi_bins,
            DM_sigma_r_bins,
            DM_L_err_scaler,
            DM_L_max,
            gas_rho_bins,
            gas_conv_phi_bins,
            gas_sigma_r_bins,
            gas_L_err_scaler,
            gas_L_max,
            total_phi_bins,
        )

    def _get_DM_beta_bins(self, r_bins):
        """
        Get the velocity anisotropy profile for DM. The default is isotropic, i.e., beta=0.
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

    def _get_gas_beta_bins(self, r_bins):
        """
        Get the velocity anisotropy profile for gas. The default is isotropic, i.e., beta=0.
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

    def _get_sigma_r_bins(self, r_bins, rho_bins, phi_bins, beta_bins):
        """
        Get the radial velocity dispersion profile.

        Parameters:
        ----------
        r_bins: array
            Array of the radius bins.
        rho_bins: array
            Array of the density bins.
        phi_bins: array
            Array of the potential bins.

        Returns:
        -------
        sigma_r_bins: array
            Array of the radial velocity dispersion bins.
        """

        _ln_fbeta_bins = 2 * beta_bins / r_bins
        _fbeta_bins = np.exp(cumulative_trapezoid(_ln_fbeta_bins, r_bins, initial=0))

        _delta_fbeta_rho_sigma_r_sqr_integrand = (
            _fbeta_bins * rho_bins * np.gradient(phi_bins, np.log(r_bins)) / r_bins
        )
        _delta_fbeta_rho_sigma_r_sqr_bins = cumulative_trapezoid(
            _delta_fbeta_rho_sigma_r_sqr_integrand, r_bins, initial=0
        )
        return np.sqrt(
            (_delta_fbeta_rho_sigma_r_sqr_bins[-1] - _delta_fbeta_rho_sigma_r_sqr_bins)
            / (_fbeta_bins * rho_bins)
        )

    def _get_L_err_scaler(self, r_bins, rho_bins, sigma_r_bins):
        """
        Calculate the scaler for the angular momentum dispersion of the halo.
        The actual angular momentum dispersion is given by L_err = sqrt(m) * L_err_scaler.
        """

        _L_err_scaler_sqr_integrand = (
            8 * np.pi / 3 * r_bins**4 * rho_bins * sigma_r_bins**2
        )

        return np.sqrt(np.trapezoid(_L_err_scaler_sqr_integrand, r_bins))

    def _get_L_max(self, r_bins, rho_bins, sigma_r_bins):
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

    def _get_sigma_r_related(self, r_bins, rho_bins, total_phi_bins, beta_bins):
        """
        Get the radial velocity dispersion related profiles and params.

        Parameters:
        ----------
        r_bins: array
            Array of the radius bins.
        rho_bins: array
            Array of the density bins.
        total_phi_bins: array
            Array of the total potential bins.
        beta_bins: array
            Array of the velocity anisotropy bins.

        Returns:
        -------
        sigma_r_bins: array
            Array of the radial velocity dispersion bins.
        L_err_scaler: float
            Scaler for the angular momentum dispersion of the halo.
        L_max: float
            The maximum angular momentum for the halo.
        """
        if np.isclose(rho_bins, 0).all():
            return np.zeros_like(r_bins), 0, 0

        sigma_r_bins = self._get_sigma_r_bins(
            r_bins, rho_bins, total_phi_bins, beta_bins
        )

        L_err_scaler = self._get_L_err_scaler(r_bins, rho_bins, sigma_r_bins)
        L_max = self._get_L_max(r_bins, rho_bins, sigma_r_bins)

        return sigma_r_bins, L_err_scaler, L_max

    def _get_Eddington_bins(self, rho_bins, phi_bins):
        """
        Perform Eddington's inversion to get the distribution function bins.

        Parameters:
        ----------
        rho_bins: array
            Array of the density bins.
        phi_bins: array
            Array of the potential bins.

        Returns:
        -------
        eps_bins: array
            Array of the eps bins, where eps = -phi.
        f_eps_bins: array
            Array of the distribution function bins.
        """

        _psi_bins = np.flip(-phi_bins)
        _d2rho_dpsi2_bins = np.flip(
            rho_bins
            * (
                np.gradient(np.log(rho_bins), phi_bins) ** 2
                + np.gradient(np.gradient(np.log(rho_bins), phi_bins), phi_bins)
            )
        )

        _lin_log_d2rho_dpsi2_interp = get_interp(
            _psi_bins[1:], np.log(_d2rho_dpsi2_bins[1:])
        )

        _eps_bins = _psi_bins
        _f_eps_bins = [0]
        for i in tqdm(range(1, len(phi_bins)), desc="Eddington's inversion"):

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
            Array of the eps bins, where eps = -phi.
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
