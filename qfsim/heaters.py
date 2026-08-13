"""
heaters.py
========================
This programme models the heat dynamics of the PIC currently

Dependencies
------------
NumPy is required.
SciPy is required only if the numerical fallback is used.
"""
from __future__ import annotations

# import argparse
# import csv
# import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from . import misc
from . import one_two_decomposition
from . import two_one_decomposition
from . import core


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

MZI_NAMES: tuple[str, ...] = ("a23", "a12", "a34", "b23", "b12", "b34")
MZI_INDEX = {name: index for index, name in enumerate(MZI_NAMES)}
MZI_PAIRS: dict[str, tuple[int, int]] = {
    "a23": (1, 2),
    "a12": (0, 1),
    "a34": (2, 3),
    "b23": (1, 2),
    "b12": (0, 1),
    "b34": (2, 3),
}
ACTUATOR_NAMES: tuple[str, ...] = tuple(
    [f"theta_{name}" for name in MZI_NAMES] + [f"phi_{name}" for name in MZI_NAMES]
)

N_MZI = len(MZI_NAMES)
N_ACTUATORS = len(ACTUATOR_NAMES)


def PIC_Topology(decomp='1212'):
    if decomp=='1212':
        mzi_names = ("a23", "a12", "a34", "b23", "b12", "b34")
        mzi_index = {name: index for index, name in enumerate(mzi_names)}
        mzi_pairs = {
            "a23": (1, 2),
            "a12": (0, 1),
            "a34": (2, 3),
            "b23": (1, 2),
            "b12": (0, 1),
            "b34": (2, 3),
        }
        actuator_names = tuple(
            [f"theta_{name}" for name in mzi_names] + [f"phi_{name}" for name in mzi_names]
        )
    elif decomp=='2121':
        mzi_names = ("a12", "a34", "a23", "b12", "b34", "b23")
        mzi_index = {name: index for index, name in enumerate(mzi_names)}
        mzi_pairs = {
            "a12": (0, 1),
            "a34": (2, 3),
            "a23": (1, 2),
            "b12": (0, 1),
            "b34": (2, 3),
            "b23": (1, 2),
            
        }
        actuator_names = tuple(
            [f"theta_{name}" for name in mzi_names] + [f"phi_{name}" for name in mzi_names]
        )

    return {MZI_NAMES : mzi_names,
            MZI_INDEX : mzi_index,
            MZI_PAIRS : mzi_pairs,
            ACTUATOR_NAMES : actuator_names}


def mzi_matrix(theta: float, phi: float) -> ComplexArray:
    """
    Alias for givens.
    """
    return misc.Givens(theta, phi)


def real_rotation(angle: float) -> ComplexArray:
    """Return a small real parasitic two-mode rotation."""

    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=complex)


def embed_two_mode(block: ComplexArray, pair: tuple[int, int]) -> ComplexArray:
    """Embed a 2 x 2 transfer block into four modes."""

    block = np.asarray(block, dtype=complex)
    if block.shape != (2, 2):
        raise ValueError("block must have shape (2, 2)")
    i, j = pair
    result = np.eye(4, dtype=complex)
    result[np.ix_([i, j], [i, j])] = block
    return result


def embedded_mzi(theta: float, phi: float, pair: tuple[int, int]) -> ComplexArray:
    """Embed an ideal MZI into the specified pair of four optical rails."""

    return embed_two_mode(mzi_matrix(theta, phi), pair)


def ideal_mirrored_u4(theta: FloatArray, phi: FloatArray) -> ComplexArray:
    r"""Return the exact ideal mirrored U(4) matrix.

    The multiplication order is

        a23, (a12 direct_sum a34), b23, (b12 direct_sum b34).
    """

    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if theta.shape != (N_MZI,) or phi.shape != (N_MZI,):
        raise ValueError("theta and phi must each have shape (6,)")

    matrices = {
        name: embedded_mzi(theta[k], phi[k], MZI_PAIRS[name])
        for k, name in enumerate(MZI_NAMES)
    }
    return (
        matrices["a23"]
        @ matrices["a12"]
        @ matrices["a34"]
        @ matrices["b23"]
        @ matrices["b12"]
        @ matrices["b34"]
    )


def ideal_single_port_powers(
    theta: FloatArray,
    phi: FloatArray,
    input_port: int,
) -> FloatArray:
    """Return ideal output powers for one one-based input port."""

    if input_port not in (1, 2, 3, 4):
        raise ValueError("input_port must be one of 1, 2, 3, 4")
    field = np.zeros(4, dtype=complex)
    field[input_port - 1] = 1.0
    output = field @ ideal_mirrored_u4(theta, phi)
    return np.abs(output) ** 2


def boundary_phase_matrix() -> FloatArray:
    r"""Return C_boundary for the port-1/port-4 ideal boundary experiment.

    C phi = (chi_L, chi_R), where

        chi_L = phi_a12 - phi_b23 + phi_b12,
        chi_R = phi_a34 - phi_b23 + phi_b34.
    """

    return np.array(
        [
            [0.0, 1.0, 0.0, -1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, -1.0, 0.0, 1.0],
        ]
    )


def nominal_heater_positions_um() -> FloatArray:
    r"""Return nominal planar coordinates for the twelve heaters.

    The first six rows are theta heaters and the final six rows are phi
    heaters.  The MZI centres follow the four optical layers.  The two heaters
    belonging to one MZI are displaced vertically by 45 micrometres.
    """

    centres = np.array(
        [
            [0.0, 450.0],  # a23
            [550.0, 150.0],  # a12
            [550.0, 750.0],  # a34
            [1100.0, 450.0],  # b23
            [1650.0, 150.0],  # b12
            [1650.0, 750.0],  # b34
        ]
    )
    theta_positions = centres + np.array([0.0, -22.5])
    phi_positions = centres + np.array([0.0, 22.5])
    return np.vstack([theta_positions, phi_positions])


def thermal_green_matrix(
    positions_um: FloatArray,
    self_resistance_k_per_w: FloatArray,
    crosstalk_fraction: float,
    decay_length_um: float,
) -> FloatArray:
    r"""Construct a reciprocal reduced thermal Green matrix.

    For h != k,

        H_hk = c sqrt(H_hh H_kk) exp(-d_hk / ell_th).

    This is not a finite-element solution.  It is the dense Green-matrix
    reduction of the steady heat equation used by the virtual chip.
    """

    positions_um = np.asarray(positions_um, dtype=float)
    diagonal = np.asarray(self_resistance_k_per_w, dtype=float)
    if positions_um.shape != (N_ACTUATORS, 2):
        raise ValueError("positions_um must have shape (12, 2)")
    if diagonal.shape != (N_ACTUATORS,):
        raise ValueError("self_resistance_k_per_w must have shape (12,)")
    if not (0.0 <= crosstalk_fraction < 1.0):
        raise ValueError("crosstalk_fraction must lie in [0, 1)")
    if decay_length_um <= 0.0:
        raise ValueError("decay_length_um must be positive")

    displacement = positions_um[:, None, :] - positions_um[None, :, :]
    distances = np.linalg.norm(displacement, axis=2)
    scale = np.sqrt(diagonal[:, None] * diagonal[None, :])
    matrix = crosstalk_fraction * scale * np.exp(-distances / decay_length_um)
    np.fill_diagonal(matrix, diagonal)
    return matrix


@dataclass(frozen=True)
class ElectroThermalParameters:
    """Physical parameters of twelve independent electrical heaters."""

    ambient_temperature_k: float
    resistance_0_ohm: FloatArray
    resistance_tcr_per_k: FloatArray
    voltage_max_v: FloatArray
    thermal_green_k_per_w: FloatArray
    positions_um: FloatArray

    def __post_init__(self) -> None:
        vectors = (
            self.resistance_0_ohm,
            self.resistance_tcr_per_k,
            self.voltage_max_v,
        )
        if any(np.asarray(v).shape != (N_ACTUATORS,) for v in vectors):
            raise ValueError("electrical vectors must each have shape (12,)")
        if np.asarray(self.thermal_green_k_per_w).shape != (
            N_ACTUATORS,
            N_ACTUATORS,
        ):
            raise ValueError("thermal_green_k_per_w must have shape (12, 12)")
        if np.asarray(self.positions_um).shape != (N_ACTUATORS, 2):
            raise ValueError("positions_um must have shape (12, 2)")
        if np.any(np.asarray(self.resistance_0_ohm) <= 0.0):
            raise ValueError("all zero-temperature resistances must be positive")
        if np.any(np.asarray(self.voltage_max_v) <= 0.0):
            raise ValueError("all maximum voltages must be positive")


@dataclass(frozen=True)
class ThermalState:
    """One converged steady-state electro-thermal solution."""

    controls: FloatArray
    voltages_v: FloatArray
    currents_a: FloatArray
    powers_w: FloatArray
    temperature_rise_k: FloatArray
    resistance_ohm: FloatArray
    iterations: int
    fixed_point_residual_k: float


def solve_steady_thermal_state(
    controls: FloatArray,
    parameters: ElectroThermalParameters,
    *,
    tolerance_k: float = 1.0e-11,
    max_iterations: int = 300,
    relaxation: float = 0.65,
) -> ThermalState:
    r"""Solve the coupled Joule-heating fixed point.

    The iteration solves

        DeltaT = H [V^2 / (R0 (1 + alpha_R DeltaT))].

    Under-relaxation is used because it is stable over a wider set of thermal
    and resistance parameters than direct Picard iteration.
    """

    u = np.asarray(controls, dtype=float)
    if u.shape != (N_ACTUATORS,):
        raise ValueError("controls must have shape (12,)")
    if np.any((u < 0.0) | (u > 1.0)):
        raise ValueError("every normalized control must lie in [0, 1]")
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("relaxation must lie in (0, 1]")

    voltages = parameters.voltage_max_v * u
    temperature = np.zeros(N_ACTUATORS, dtype=float)
    residual = math.inf

    for iteration in range(1, max_iterations + 1):
        resistance_factor = 1.0 + parameters.resistance_tcr_per_k * temperature
        if np.any(resistance_factor <= 0.05):
            raise RuntimeError("nonphysical heater resistance during iteration")
        resistance = parameters.resistance_0_ohm * resistance_factor
        powers = voltages**2 / resistance
        proposed = parameters.thermal_green_k_per_w @ powers
        updated = (1.0 - relaxation) * temperature + relaxation * proposed
        residual = float(np.max(np.abs(updated - temperature)))
        temperature = updated
        if residual < tolerance_k:
            break
    else:
        raise RuntimeError(
            f"thermal fixed point failed to converge after {max_iterations} iterations"
        )

    resistance = parameters.resistance_0_ohm * (
        1.0 + parameters.resistance_tcr_per_k * temperature
    )
    powers = voltages**2 / resistance
    currents = np.divide(
        voltages,
        resistance,
        out=np.zeros_like(voltages),
        where=resistance > 0.0,
    )
    fixed_point = parameters.thermal_green_k_per_w @ powers
    fixed_point_residual = float(np.max(np.abs(temperature - fixed_point)))
    return ThermalState(
        controls=u.copy(),
        voltages_v=voltages,
        currents_a=currents,
        powers_w=powers,
        temperature_rise_k=temperature,
        resistance_ohm=resistance,
        iterations=iteration,
        fixed_point_residual_k=fixed_point_residual,
    )


@dataclass(frozen=True)
class ThermoOpticParameters:
    r"""Map the twelve thermal sites to six theta and six phi coordinates."""

    offset_rad: FloatArray
    linear_rad_per_k: FloatArray
    quadratic_rad_per_k2: FloatArray

    def __post_init__(self) -> None:
        vectors = (
            self.offset_rad,
            self.linear_rad_per_k,
            self.quadratic_rad_per_k2,
        )
        if any(np.asarray(v).shape != (N_ACTUATORS,) for v in vectors):
            raise ValueError("thermo-optic vectors must each have shape (12,)")


def thermo_optic_coordinates(
    state: ThermalState,
    parameters: ThermoOpticParameters,
) -> tuple[FloatArray, FloatArray]:
    r"""Return unwrapped and wrapped x=(theta_1,...,theta_6,phi_1,...,phi_6)."""

    delta_t = state.temperature_rise_k
    unwrapped = (
        parameters.offset_rad
        + parameters.linear_rad_per_k * delta_t
        + 0.5 * parameters.quadratic_rad_per_k2 * delta_t**2
    )
    return unwrapped, misc.wrap_to_pi(unwrapped)




@dataclass(frozen=True)
class OpticalImperfections:
    r"""Fixed fabrication and propagation imperfections.

    Each physical cell is represented as

        B_j = R(epsilon_pre,j) D_j G(theta_j, phi_j) R(epsilon_post,j),

    where D_j contains field transmissions.  Three diagonal propagation
    matrices are inserted between the four mesh layers.
    """

    pre_rotation_rad: FloatArray
    post_rotation_rad: FloatArray
    arm_power_transmission: FloatArray
    propagation_power_transmission: FloatArray
    propagation_phase_rad: FloatArray

    def __post_init__(self) -> None:
        if np.asarray(self.pre_rotation_rad).shape != (N_MZI,):
            raise ValueError("pre_rotation_rad must have shape (6,)")
        if np.asarray(self.post_rotation_rad).shape != (N_MZI,):
            raise ValueError("post_rotation_rad must have shape (6,)")
        if np.asarray(self.arm_power_transmission).shape != (N_MZI, 2):
            raise ValueError("arm_power_transmission must have shape (6, 2)")
        if np.asarray(self.propagation_power_transmission).shape != (3, 4):
            raise ValueError("propagation_power_transmission must have shape (3, 4)")
        if np.asarray(self.propagation_phase_rad).shape != (3, 4):
            raise ValueError("propagation_phase_rad must have shape (3, 4)")
        if np.any(
            (np.asarray(self.arm_power_transmission) <= 0.0)
            | (np.asarray(self.arm_power_transmission) > 1.0)
        ):
            raise ValueError("arm power transmissions must lie in (0, 1]")


def physical_cell_matrix(
    theta: float,
    phi: float,
    mzi_index: int,
    imperfections: OpticalImperfections,
) -> ComplexArray:
    """Return the non-ideal 2 x 2 transfer matrix of one MZI."""

    field_transmission = np.sqrt(imperfections.arm_power_transmission[mzi_index])
    differential_loss = np.diag(field_transmission.astype(complex))
    return (
        real_rotation(imperfections.pre_rotation_rad[mzi_index])
        @ differential_loss
        @ mzi_matrix(theta, phi)
        @ real_rotation(imperfections.post_rotation_rad[mzi_index])
    )


def propagation_matrix(
    layer_index: int,
    imperfections: OpticalImperfections,
) -> ComplexArray:
    """Return a four-mode diagonal propagation matrix after one mesh layer."""

    amplitude = np.sqrt(imperfections.propagation_power_transmission[layer_index])
    phase = imperfections.propagation_phase_rad[layer_index]
    return np.diag(amplitude * np.exp(1j * phase))


def physical_scattering_matrix(
    theta: FloatArray,
    phi: FloatArray,
    imperfections: OpticalImperfections,
) -> ComplexArray:
    """Return the generally non-unitary transfer matrix of the physical mesh."""

    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if theta.shape != (N_MZI,) or phi.shape != (N_MZI,):
        raise ValueError("theta and phi must each have shape (6,)")

    cells = {}
    for k, name in enumerate(MZI_NAMES):
        local = physical_cell_matrix(theta[k], phi[k], k, imperfections)
        cells[name] = embed_two_mode(local, MZI_PAIRS[name])

    return (
        cells["a23"]
        @ propagation_matrix(0, imperfections)
        @ cells["a12"]
        @ cells["a34"]
        @ propagation_matrix(1, imperfections)
        @ cells["b23"]
        @ propagation_matrix(2, imperfections)
        @ cells["b12"]
        @ cells["b34"]
    )


@dataclass(frozen=True)
class SourceParameters:
    """Classical single-port calibration source."""

    mean_power_w: float
    relative_intensity_noise: float
    additive_field_noise_std: float


@dataclass(frozen=True)
class DetectorParameters:
    """Four independent power-detector channels."""

    gain: FloatArray
    dark_offset_w: FloatArray
    read_noise_std_w: FloatArray
    photons_per_watt_sample: float
    adc_step_w: float

    def __post_init__(self) -> None:
        for value in (self.gain, self.dark_offset_w, self.read_noise_std_w):
            if np.asarray(value).shape != (4,):
                raise ValueError("detector vectors must each have shape (4,)")
        if np.any(np.asarray(self.gain) <= 0.0):
            raise ValueError("detector gains must be positive")
        if self.photons_per_watt_sample <= 0.0:
            raise ValueError("photons_per_watt_sample must be positive")
        if self.adc_step_w < 0.0:
            raise ValueError("adc_step_w cannot be negative")



@dataclass
class SteadyStateU4Chip:
    """Complete hidden steady-state virtual chip."""

    electrothermal: ElectroThermalParameters
    thermo_optic: ThermoOpticParameters
    optical: OpticalImperfections
    source: SourceParameters
    detectors: DetectorParameters

    @classmethod
    def random(
        cls,
        seed: int = 20260727,
        *,
        thermal_crosstalk_fraction: float = 0.11,
        detector_noise_scale: float = 1.0,
    ) -> "SteadyStateU4Chip":
        """Construct one reproducible but fabrication-imperfect virtual device."""

        rng = np.random.default_rng(seed)
        positions = nominal_heater_positions_um()
        positions = positions + rng.normal(0.0, 5.0, size=positions.shape)

        self_thermal = rng.normal(1750.0, 120.0, N_ACTUATORS)
        thermal_green = thermal_green_matrix(
            positions,
            self_thermal,
            thermal_crosstalk_fraction,
            decay_length_um=310.0,
        )
        electrothermal = ElectroThermalParameters(
            ambient_temperature_k=298.15,
            resistance_0_ohm=rng.normal(150, 8.5, N_ACTUATORS),
            resistance_tcr_per_k=rng.normal(2.5e-5, 5e-6, N_ACTUATORS),
            voltage_max_v=rng.normal(9, 0.4, N_ACTUATORS),
            thermal_green_k_per_w=thermal_green,
            positions_um=positions,
        )

        wavelength_um = 1.550
        dn_eff_d_t = 1.85e-4
        heater_lengths_um = np.concatenate(
            [
                rng.normal(1050.0, 45.0, N_MZI),
                rng.normal(900.0, 40.0, N_MZI),
            ]
        )
        optical_overlap = np.concatenate(
            [
                rng.normal(0.34, 0.015, N_MZI),
                rng.normal(0.30, 0.015, N_MZI),
            ]
        )

        ##############################################################
        linear = (
            2.0
            * np.pi
            / wavelength_um
            * dn_eff_d_t
            * heater_lengths_um
            * optical_overlap
        )
        linear *= rng.normal(1.0, 0.025, N_ACTUATORS)
        quadratic = linear * rng.normal(8.0e-4, 1.5e-4, N_ACTUATORS)
        ##############################################################

        thermo_optic = ThermoOpticParameters(
            offset_rad=rng.uniform(-np.pi, np.pi, N_ACTUATORS),
            linear_rad_per_k=linear,
            quadratic_rad_per_k2=quadratic,
        )

        optical = OpticalImperfections(
            pre_rotation_rad=rng.normal(0.0, 0.0, N_MZI),
            post_rotation_rad=rng.normal(0.0, 0.0, N_MZI),
            arm_power_transmission=np.clip(
                rng.normal(0.986, 0.004, size=(N_MZI, 2)),
                0.96,
                0.9995,
            ),
            propagation_power_transmission=np.clip(
                rng.normal(0.992, 0.002, size=(3, 4)),
                0.98,
                0.9995,
            ),
            propagation_phase_rad=rng.uniform(-0.10, 0.10, size=(3, 4)),
        )

        source = SourceParameters(
            mean_power_w=1.0e-3,
            relative_intensity_noise=2.0e-3 * detector_noise_scale,
            additive_field_noise_std=2.0e-4 * detector_noise_scale,
        )
        detectors = DetectorParameters(
            gain=rng.normal(1.0, 0.025, 4),
            dark_offset_w=np.clip(
                rng.normal(2.5e-7, 0.5e-7, 4),
                0.0,
                None,
            ),
            read_noise_std_w=np.full(4, 2.5e-7 * detector_noise_scale),
            photons_per_watt_sample=2.0e9 / max(detector_noise_scale, 1.0e-9),
            adc_step_w=1.0e-8 * detector_noise_scale,
        )
        return cls(electrothermal, thermo_optic, optical, source, detectors)

    def steady_state(self, controls: FloatArray) -> ThermalState:
        """Solve the electro-thermal state for one twelve-control vector."""

        return solve_steady_thermal_state(controls, self.electrothermal)

    def physical_coordinates(
        self,
        controls: FloatArray,
    ) -> tuple[FloatArray, FloatArray, ThermalState]:
        """Return unwrapped x, wrapped x and the corresponding thermal state."""

        state = self.steady_state(controls)
        unwrapped, wrapped = thermo_optic_coordinates(state, self.thermo_optic)
        return unwrapped, wrapped, state

    def scattering_matrix(self, controls: FloatArray) -> ComplexArray:
        """Return the physical 4 x 4 optical transfer matrix."""

        _, wrapped, _ = self.physical_coordinates(controls)
        return physical_scattering_matrix(
            wrapped[:N_MZI],
            wrapped[N_MZI:],
            self.optical,
        )

    def noiseless_output_power(
        self,
        controls: FloatArray,
        input_port: int,
        *,
        input_power_w: float | None = None,
    ) -> FloatArray:
        """Return the pre-detector optical power in the four output waveguides."""

        if input_port not in (1, 2, 3, 4):
            raise ValueError("input_port must be one of 1, 2, 3, 4")
        power = self.source.mean_power_w if input_power_w is None else input_power_w
        field = np.zeros(4, dtype=complex)
        field[input_port - 1] = math.sqrt(power)
        output = field @ self.scattering_matrix(controls)
        return np.abs(output) ** 2

    def detector_reference_shot(
        self,
        reference_power_w: float,
        rng: np.random.Generator,
    ) -> FloatArray:
        """Simulate an external equal-power detector-reference measurement."""

        incident = np.full(4, max(reference_power_w, 0.0))
        return self._detect(incident, rng)

    def electrical_readback(
        self,
        controls: FloatArray,
        rng: np.random.Generator,
        *,
        relative_noise: float = 2.0e-4,
    ) -> tuple[FloatArray, FloatArray]:
        """Return noisy voltage and current readback for all heaters."""

        state = self.steady_state(controls)
        voltage = state.voltages_v * (
            1.0 + rng.normal(0.0, relative_noise, N_ACTUATORS)
        )
        current = state.currents_a * (
            1.0 + rng.normal(0.0, relative_noise, N_ACTUATORS)
        )
        return voltage, current

    def _detect(
        self,
        incident_power_w: FloatArray,
        rng: np.random.Generator,
    ) -> FloatArray:
        """Apply photon statistics, detector gain, offset, read noise and ADC."""

        incident = np.clip(np.asarray(incident_power_w, dtype=float), 0.0, None)
        expected_photons = incident * self.detectors.photons_per_watt_sample
        photons = rng.poisson(expected_photons)
        shot_power = photons / self.detectors.photons_per_watt_sample
        measured = (
            self.detectors.gain * shot_power
            + self.detectors.dark_offset_w
            + rng.normal(0.0, self.detectors.read_noise_std_w, 4)
        )
        measured = np.clip(measured, 0.0, None)
        if self.detectors.adc_step_w > 0.0:
            measured = (
                np.round(measured / self.detectors.adc_step_w)
                * self.detectors.adc_step_w
            )
        return measured

    def measure_once(
        self,
        controls: FloatArray,
        input_port: int,
        rng: np.random.Generator,
    ) -> FloatArray:
        """Perform one noisy optical measurement with one illuminated input."""

        source_scale = max(
            0.0,
            1.0 + rng.normal(0.0, self.source.relative_intensity_noise),
        )
        input_power = self.source.mean_power_w * source_scale
        field = np.zeros(4, dtype=complex)
        field[input_port - 1] = math.sqrt(input_power)
        if self.source.additive_field_noise_std > 0.0:
            field[input_port - 1] *= 1.0 + (
                rng.normal(0.0, self.source.additive_field_noise_std)
                + 1j * rng.normal(0.0, self.source.additive_field_noise_std)
            )
        output = field @ self.scattering_matrix(controls)
        return self._detect(np.abs(output) ** 2, rng)

    def parameter_expressions(self) -> dict[str, str]:
        """Return the twelve explicit numerical thermo-optic response expressions."""

        expressions: dict[str, str] = {}
        for h, name in enumerate(ACTUATOR_NAMES):
            offset = self.thermo_optic.offset_rad[h]
            linear = self.thermo_optic.linear_rad_per_k[h]
            quadratic_half = 0.5 * self.thermo_optic.quadratic_rad_per_k2[h]
            expressions[name] = (
                f"{name}(u) = misc.wrap_to_pi({offset:+.12g} "
                f"{linear:+.12g}*DeltaT_{h} "
                f"{quadratic_half:+.12g}*DeltaT_{h}^2)"
            )
        return expressions