#!/usr/bin/env python
"""
Analytic gravitational potential models in JAX.

This module provides analytic potential components commonly used in galactic dynamics:
- Miyamoto-Nagai disk potential
- NFW halo potential
- Combined AnalyticPotential class

All potentials are implemented as equinox modules with trainable parameters.
These components are used by PhiNN in potential.py when type='Analytic'.
"""

from __future__ import print_function, division

import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy
from jaxtyping import Array
import equinox as eqx

import numpy as np
from typing import Optional


# Physical constants
# Converts [M_sun * G] to [(100km/s)^2 kpc]
G_MSUN_TO_INTERNAL = 4.3e-10


class MiyamotoNagaiDisk(eqx.Module):
    """
    Miyamoto-Nagai disk potential.

    The potential is given by:
        Phi(R, z) = -G*M / sqrt(R^2 + (sqrt(z^2 + b^2) + a)^2)

    where R is the cylindrical radius, z is the height, M is the total mass,
    a is the scale radius, and b is the scale height.

    Reference: https://docs.galpy.org/en/v1.8.1/reference/potentialmiyamoto.html

    Attributes:
        log_amp: Log of the amplitude (mass in M_sun).
        log_a: Log of the scale radius (kpc).
        log_b: Log of the scale height (kpc).
        *_trainable: Flags indicating which parameters are trainable.
    """
    log_amp: Array
    log_a: Array
    log_b: Array

    amp_trainable: bool = eqx.field(static=True)
    a_trainable: bool = eqx.field(static=True)
    b_trainable: bool = eqx.field(static=True)

    def __init__(
        self,
        amp: float = 6.8e10,
        a: float = 3.0,
        b: float = 0.28,
        amp_trainable: bool = False,
        a_trainable: bool = False,
        b_trainable: bool = False,
    ):
        """
        Initialize a Miyamoto-Nagai disk potential.

        Args:
            amp: Mass amplitude in M_sun. Default is MWPotential2014 disk mass.
            a: Scale radius in kpc. Default is MWPotential2014 value.
            b: Scale height in kpc. Default is MWPotential2014 value.
            amp_trainable: Whether amplitude is trainable.
            a_trainable: Whether scale radius is trainable.
            b_trainable: Whether scale height is trainable.
        """
        self.log_amp = jnp.log(jnp.array(amp, dtype=jnp.float32))
        self.log_a = jnp.log(jnp.array(a, dtype=jnp.float32))
        self.log_b = jnp.log(jnp.array(b, dtype=jnp.float32))

        self.amp_trainable = amp_trainable
        self.a_trainable = a_trainable
        self.b_trainable = b_trainable

    @property
    def amp(self) -> Array:
        return jnp.exp(self.log_amp)

    @property
    def a(self) -> Array:
        return jnp.exp(self.log_a)

    @property
    def b(self) -> Array:
        return jnp.exp(self.log_b)

    def __call__(self, R2: Array, z: Array) -> Array:
        """
        Evaluate the potential.

        Args:
            R2: Squared cylindrical radius (R^2 = x^2 + y^2).
            z: Height above/below the disk plane.

        Returns:
            The gravitational potential value.
        """
        a = jnp.exp(self.log_a)
        b = jnp.exp(self.log_b)
        amp = jnp.exp(self.log_amp)

        zterm = jnp.sqrt(z**2 + b**2) + a
        pot = -G_MSUN_TO_INTERNAL * amp / jnp.sqrt(R2 + zterm**2)
        return pot


class NFWHalo(eqx.Module):
    """
    Navarro-Frenk-White (NFW) halo potential.

    The potential is given by:
        Phi(r) = -G*M / r * ln(1 + r/a)

    where r is the spherical radius, M is a characteristic mass, and a is the
    scale radius.

    Reference: https://docs.galpy.org/en/v1.8.1/reference/potentialnfw.html

    Attributes:
        log_amp: Log of the amplitude (characteristic mass in M_sun).
        log_a: Log of the scale radius (kpc).
        *_trainable: Flags indicating which parameters are trainable.
    """
    log_amp: Array
    log_a: Array

    amp_trainable: bool = eqx.field(static=True)
    a_trainable: bool = eqx.field(static=True)

    def __init__(
        self,
        amp: float = 1.0e12,
        a: float = 16.0,
        amp_trainable: bool = False,
        a_trainable: bool = False,
    ):
        """
        Initialize an NFW halo potential.

        Args:
            amp: Characteristic mass in M_sun.
            a: Scale radius in kpc. Default is MWPotential2014 value.
            amp_trainable: Whether amplitude is trainable.
            a_trainable: Whether scale radius is trainable.
        """
        self.log_amp = jnp.log(jnp.array(amp, dtype=jnp.float32))
        self.log_a = jnp.log(jnp.array(a, dtype=jnp.float32))

        self.amp_trainable = amp_trainable
        self.a_trainable = a_trainable

    @property
    def amp(self) -> Array:
        return jnp.exp(self.log_amp)

    @property
    def a(self) -> Array:
        return jnp.exp(self.log_a)

    def __call__(self, r2: Array) -> Array:
        """
        Evaluate the potential.

        Args:
            r2: Squared spherical radius (r^2 = x^2 + y^2 + z^2).

        Returns:
            The gravitational potential value.
        """
        a = jnp.exp(self.log_a)
        amp = jnp.exp(self.log_amp)

        # Use softened radius to avoid division by zero and ensure smooth gradients
        # The softening is small enough not to affect physical results
        eps = 1e-6
        r = jnp.sqrt(r2 + eps**2)
        pot = -G_MSUN_TO_INTERNAL * amp * jnp.log1p(r / a) / r
        return pot


class AnalyticPotential(eqx.Module):
    """
    JAX/Equinox reimplementation of the legacy TensorFlow PhiNNAnalytic class.

    This version explicitly stores `_trainable` booleans for every parameter
    to support custom filtering logic, imitating the metadata storage of the
    original TensorFlow variables.
    """

    # --- Constants ---
    r_c: jnp.ndarray
    dxy: jnp.ndarray
    # We mark these as False statically
    r_c_trainable: bool = eqx.field(static=True, default=False)
    dxy_trainable: bool = eqx.field(static=True, default=False)

    # --- Global Geometry ---
    dz: jnp.ndarray
    dz_trainable: bool

    # --- Disk 1 ---
    mn1_logamp: jnp.ndarray
    mn1_logamp_trainable: bool

    mn1_loga: jnp.ndarray
    mn1_loga_trainable: bool

    mn1_logb: jnp.ndarray
    mn1_logb_trainable: bool

    # --- Disk 2 ---
    mn2_logamp: jnp.ndarray
    mn2_logamp_trainable: bool

    mn2_loga: jnp.ndarray
    mn2_loga_trainable: bool

    mn2_logb: jnp.ndarray
    mn2_logb_trainable: bool

    # --- Disk 3 ---
    mn3_logamp: jnp.ndarray
    mn3_logamp_trainable: bool

    mn3_loga: jnp.ndarray
    mn3_loga_trainable: bool

    mn3_logb: jnp.ndarray
    mn3_logb_trainable: bool

    # --- Halo ---
    halo_logamp: jnp.ndarray
    halo_logamp_trainable: bool

    halo_loga: jnp.ndarray
    halo_loga_trainable: bool

    def __init__(
        self,
        n_dim=3,
        dz=0., dz_trainable=False,
        mn1_amp=1., mn1_amp_trainable=False,
        mn1_a=3., mn1_a_trainable=False,
        mn1_b=0.5, mn1_b_trainable=False,
        mn2_amp=1., mn2_amp_trainable=False,
        mn2_a=3., mn2_a_trainable=False,
        mn2_b=0.5, mn2_b_trainable=False,
        mn3_amp=1., mn3_amp_trainable=False,
        mn3_a=3., mn3_a_trainable=False,
        mn3_b=0.5, mn3_b_trainable=False,
        halo_amp=0., halo_amp_trainable=False,
        halo_a=16., halo_a_trainable=False,
        r_c=8.3,
        **kwargs
    ):
        # Coordinate system
        self.r_c = jnp.array(r_c, dtype=jnp.float32)
        self.dxy = jnp.array([-r_c, 0.], dtype=jnp.float32)

        self.dz = jnp.array(dz, dtype=jnp.float32)
        self.dz_trainable = dz_trainable

        # Helper to log-ify inputs
        def to_log(x):
            return jnp.log(jnp.array(x, dtype=jnp.float32))

        # Disk 1
        self.mn1_logamp = to_log(mn1_amp)
        self.mn1_logamp_trainable = mn1_amp_trainable

        self.mn1_loga = to_log(mn1_a)
        self.mn1_loga_trainable = mn1_a_trainable

        self.mn1_logb = to_log(mn1_b)
        self.mn1_logb_trainable = mn1_b_trainable

        # Disk 2
        self.mn2_logamp = to_log(mn2_amp)
        self.mn2_logamp_trainable = mn2_amp_trainable

        self.mn2_loga = to_log(mn2_a)
        self.mn2_loga_trainable = mn2_a_trainable

        self.mn2_logb = to_log(mn2_b)
        self.mn2_logb_trainable = mn2_b_trainable

        # Disk 3
        self.mn3_logamp = to_log(mn3_amp)
        self.mn3_logamp_trainable = mn3_amp_trainable

        self.mn3_loga = to_log(mn3_a)
        self.mn3_loga_trainable = mn3_a_trainable

        self.mn3_logb = to_log(mn3_b)
        self.mn3_logb_trainable = mn3_b_trainable

        # Halo
        self.halo_logamp = to_log(halo_amp)
        self.halo_logamp_trainable = halo_amp_trainable

        self.halo_loga = to_log(halo_a)
        self.halo_loga_trainable = halo_a_trainable

    def __call__(self, q):
        """
        Returns the gravitational potential.
        Args:
            q: Input coordinates (batch_size, 3) or (3,)
        """
        q_in = jnp.atleast_2d(q)

        # xy is columns 0,1; z is column 2
        xy = q_in[..., :2]
        z = q_in[..., 2:3]

        # Apply shifts
        z = z + self.dz
        xy = xy + self.dxy

        # Radii w.r.t galactic center
        R2 = jnp.sum(xy**2, axis=-1, keepdims=True)
        r2 = R2 + z**2

        # Specific coefficient from original code
        coef = 4.3e-10

        # Helper for Miyamoto-Nagai calculation
        def mn_pot(logamp, loga, logb):
            amp = jnp.exp(logamp)
            a = jnp.exp(loga)
            b = jnp.exp(logb)
            # Formula: -coef * amp / sqrt(R2 + (sqrt(z^2 + b^2) + a)^2)
            denom = jnp.sqrt(R2 + (jnp.sqrt(z**2 + b**2) + a)**2)
            return -coef * amp / denom

        # Calculate components
        pot_mn1 = mn_pot(self.mn1_logamp, self.mn1_loga, self.mn1_logb)
        pot_mn2 = mn_pot(self.mn2_logamp, self.mn2_loga, self.mn2_logb)
        pot_mn3 = mn_pot(self.mn3_logamp, self.mn3_loga, self.mn3_logb)

        # Halo contribution (NFW)
        halo_amp = jnp.exp(self.halo_logamp)
        halo_a = jnp.exp(self.halo_loga)
        r = jnp.sqrt(r2)

        # xlogy(x, y) = x * log(y)
        pot_halo = -coef * halo_amp * xlogy(1./r, 1. + r/halo_a)

        total_pot = pot_mn1 + pot_mn2 + pot_mn3 + pot_halo

        if q.ndim == 1:
            return total_pot.squeeze()
        return total_pot
