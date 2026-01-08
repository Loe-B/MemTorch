import math

import numpy as np
import torch

import memtorch
from memtorch.utils import clip, convert_range

from .Memristor import Memristor as Memristor


class DomainWallMotion(Memristor):
    """Domain Wall Motion memristor model

    Parameters
    ----------
    time_series_resolution : float
        Time series resolution (s). //BASE
    r_off : float
        Off (maximum) resistance of the device (ohms). //BASE
    r_on : float
        On (minimum) resistance of the device (ohms). //BASE
    x_init : float
        Initial domain wall position (m).
    L : float
        Length of the memristor (m).
    gamma : float
        Gyromagnetic ratio (s^-1 T^-1).
    CPP_factor : float
        current perpendicular to plane factor.
    spin_polarization : float
        The spin polarization.
    Delta : float
        The domain wall width.
    Gilbert_damping : float
        The Gilbert damping constant.
    M_s : float
        The saturation magnetization (A/m).
    t_free : float
        The thickness of the free layer (m).
    w : float
        width of the memristor (m).
    J_c : float
        critical current density (A/m^2).
    """

    def __init__(
        self,
        time_series_resolution=1e-5,
        r_off=1e5,
        r_on=1e3,
        x_init=0.0,
        L=5000e-9,
        gamma=1.761e11,
        CPP_factor=0.4,
        spin_polarization=0.4,
        Delta=10e-9,
        Gilbert_damping=0.006,
        M_s=1050e3,
        t_free=2.2e-9,
        w=50e-9,
        J_c=3e10,
        **kwargs
    ):

        args = memtorch.bh.unpack_parameters(locals())
        super(DomainWallMotion, self).__init__(
            args.r_off, args.r_on, args.time_series_resolution, 0, 0
        )
        self.x_init = args.x_init
        self.L = args.L
        self.gamma = args.gamma
        self.CPP_factor = args.CPP_factor
        self.spin_polarization = args.spin_polarization
        self.Delta = args.Delta
        self.Gilbert_damping = args.Gilbert_damping
        self.M_s = args.M_s
        self.t_free = args.t_free
        self.w = args.w
        self.J_c = args.J_c
        self.x = self.x_init
        self.memristance = self.r_off/(1 + ((self.r_off - self.r_on) * self.x_init)/(self.r_on * self.L))
        self.g = 1/self.memristance
        h_bar = 1.054e-34
        e = 1.602e-19
        self.C = (self.gamma * h_bar * self.CPP_factor * self.spin_polarization * self.Delta)/(2 * self.Gilbert_damping * e * self.M_s * self.t_free)

    def current_density(self, voltage):
        """Method to determine the current density of the model given an applied voltage.

        Parameters
        ----------
        voltage : float
            The applied voltage (V).

        Returns
        -------
        float
            The current density (A/m^2).
        """
        return voltage/(self.memristance * self.w * self.L)

    def get_memristance(self):
        """Method to determine the memristance of the model given a domain wall position.

        Returns
        -------
        float
            The memristance (ohms).
        """
        return self.r_off/(1 + ((self.r_off - self.r_on) * self.x)/(self.r_on * self.L))

    def speed(self, J):
        """Method to determine the speed of the domain wall given a current density.

        Parameters
        ----------
        J : float
            The applied current density (A/m^2).

        Returns
        -------
        float
            The domain wall speed (m/s).
        """
        if J < self.J_c:
            return self.C * J
        else:
            return self.C * self.J_c

    def simulate(self, voltage_signal, return_current=False):
        current_density_signal = self.current_density(voltage_signal)
        if return_current:
            current = np.zeros(len(voltage_signal))
        for t in range(0, len(current_density_signal)):
            self.memristance = self.get_memristance()
            self.memristance = clip(self.memristance, self.r_on, self.r_off)
            v = self.speed(current_density_signal[t])
            self.x += v * self.time_series_resolution
            self.x = clip(self.x, 0, self.L)
            if return_current:
                current[t] = voltage_signal[t]/self.memristance
        if return_current:
            return current

    def set_conductance(self, conductance):
        conductance = clip(conductance, 1 / self.r_off, 1 / self.r_on)
        memristance = 1 / conductance
        self.memristance = clip(memristance, self.r_on, self.r_off)
        self.g = 1 / self.memristance
        self.x = (self.r_on * self.L * (self.r_off/self.memristance - 1))/(self.r_off - self.r_on)
        self.x = clip(self.x, 0, self.L)

    def plot_hysteresis_loop(
        self,
        duration=0.5,
        voltage_signal_amplitude=0.001,
        voltage_signal_frequency=100,
        log_scale=False,
        return_result=False,
    ):
        return super().plot_hysteresis_loop(
            self,
            duration=duration,
            voltage_signal_amplitude=voltage_signal_amplitude,
            voltage_signal_frequency=voltage_signal_frequency,
            log_scale=log_scale,
            return_result=return_result,
        )

    def plot_bipolar_switching_behaviour(
        self,
        voltage_signal_amplitude=0.001,
        voltage_signal_frequency=100,
        log_scale=True,
        return_result=False,
    ):
        return super().plot_bipolar_switching_behaviour(
            self,
            voltage_signal_amplitude=voltage_signal_amplitude,
            voltage_signal_frequency=voltage_signal_frequency,
            log_scale=log_scale,
            return_result=return_result,
        )
