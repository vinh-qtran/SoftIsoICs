import numpy as np
from scipy.integrate import cumulative_trapezoid, quad
from tqdm import tqdm

from softisoics.constants import G
from softisoics.ICs_writer import ICsWriter
from softisoics.utils import get_interp


class BaseSingleComponentProfile:
    def __init__(self, r_bin_min, r_bin_max, N_bins, halo_edge=None, epsilon=0):
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
        self.mass_bins, self.phi_bins, self.sigma_r_bins = self._get_all_profiles(self.r_bins, self.rho_bins)

        self.conv_rho_bins = self._get_convoluted_rho_bins(self.r_bins, self.epsilon)
        self.conv_mass_bins, self.conv_phi_bins, self.conv_sigma_r_bins = self._get_all_profiles(self.r_bins, self.conv_rho_bins)

    def _get_rho_bins(self, r_bins):
        raise NotImplementedError("Not implemented in base class.")  # noqa: EM101

    def _get_convoluted_rho_bins(self, r_bins, epsilon):
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

            if _r < self.halo_edge:
                _shell_r_bin_edges = np.linspace(
                    max(0, _r - _h), _r + _h, self.N_bins + 1
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
        _zero_mass = 4 / 3 * np.pi * r_bins[0] ** 3 * rho_bins[0]
        _mass_integrand = 4 * np.pi * r_bins**2 * rho_bins
        return cumulative_trapezoid(_mass_integrand, r_bins, initial=0) + _zero_mass

    def _get_phi_bins(self, r_bins, mass_bins):
        _delta_phi_integrand = G * mass_bins / r_bins**2
        _delta_phi_bins = cumulative_trapezoid(_delta_phi_integrand, r_bins, initial=0)

        return _delta_phi_bins - _delta_phi_bins[-1]
    
    def _get_beta_bins(self, r_bins):
        return np.zeros_like(r_bins)

    def _get_sigma_r_bins(self, r_bins, rho_bins, phi_bins):
        _beta_bins = self._get_beta_bins(r_bins)
        _ln_fbeta_bins = 2 * _beta_bins / r_bins
        _fbeta_bins = np.exp(cumulative_trapezoid(_ln_fbeta_bins, r_bins, initial=0))

        _delta_fbeta_rho_sigma_r_sqr_integrand = (
            _fbeta_bins * rho_bins * np.gradient(phi_bins, np.log(r_bins)) / r_bins
        )
        _delta_fbeta_rho_sigma_r_sqr_bins = cumulative_trapezoid(
            _delta_fbeta_rho_sigma_r_sqr_integrand, r_bins, initial=0
        )
        return np.sqrt((_delta_fbeta_rho_sigma_r_sqr_bins[-1] - _delta_fbeta_rho_sigma_r_sqr_bins) / (_fbeta_bins * rho_bins))

    def _get_all_profiles(self, r_bins, rho_bins):
        mass_bins = self._get_mass_bins(r_bins, rho_bins)
        phi_bins = self._get_phi_bins(r_bins, mass_bins)
        sigma_r_bins = self._get_sigma_r_bins(r_bins, rho_bins, phi_bins)

        return mass_bins, phi_bins, sigma_r_bins