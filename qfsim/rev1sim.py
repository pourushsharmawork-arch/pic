'''
The final file that simulates the complete Quanfluence Rev1 4-qubit photonic quantum computer.
'''

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Optional

from . import heaters
from . import detectors

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]

@dataclass
class rev1machine:
    '''The rev1machine class object.'''

    left_PIC : heaters.SteadyStateU4Chip
    right_PIC : heaters.SteadyStateU4Chip
    detectors : detectors.U4xU4DetectorArray

    
    def steady_state(self, left_controls: FloatArray, right_controls: FloatArray) -> heaters.ThermalState:
        """Solve the electro-thermal state for one twelve-control vector, for left and right PIC."""

        left_steady_state = heaters.solve_steady_thermal_state(left_controls, self.left_PIC.electrothermal)
        right_steady_state = heaters.solve_steady_thermal_state(right_controls, self.right_PIC.electrothermal)
        
        return {"left" : left_steady_state, "right" : right_steady_state}
    
    def physical_coordinates(
        self,
        left_controls: FloatArray,
        right_controls: FloatArray,
        ) -> tuple[FloatArray, FloatArray, heaters.ThermalState]:
        """Return unwrapped x, wrapped x and the corresponding thermal state."""
        state = self.steady_state(left_controls, right_controls)
        left_state = state["left"]
        right_state = state["right"]
        left_unwrapped, left_wrapped = heaters.thermo_optic_coordinates(left_state, self.left_PIC.thermo_optic)
        right_unwrapped, right_wrapped = heaters.thermo_optic_coordinates(right_state, self.right_PIC.thermo_optic)
        return {
          "left"   :  [left_unwrapped, left_wrapped, left_state],
           "right" : [right_unwrapped, right_wrapped, right_state],
        }

    def scattering_matrix(self, left_controls : FloatArray, right_controls : FloatArray) -> ComplexArray :
        Uleft = self.left_PIC.scattering_matrix(left_controls)
        Uright = self.right_PIC.scattering_matrix(right_controls)

        U = np.kron(Uleft, Uright)

        return U

    def propagate_state(self, input_signal, left_controls:FloatArray, right_contorls:FloatArray):
        M = self.scattering_matrix(left_controls, right_contorls)
        output = input_signal @ M

        return output

    def electrical_readback(
            self,
            left_controls: FloatArray,
            right_controls: FloatArray,
            rng: np.random.Generator,
            *,
            relative_noise: float = 2.0e-4,
        ) -> tuple[FloatArray, FloatArray]:
            """Return noisy voltage and current readback for all heaters. For each individual PIC"""
    
            state = self.steady_state(left_controls, right_controls)
            left_state = state["left"]
            right_state = state["right"]

            left_voltage = left_state.voltages_v * (
                1.0 + rng.normal(0.0, relative_noise, heaters.N_ACTUATORS)
            )
            left_current = left_state.currents_a * (
                1.0 + rng.normal(0.0, relative_noise, heaters.N_ACTUATORS)
            )

            right_voltage = right_state.voltages_v * (
                            1.0 + rng.normal(0.0, relative_noise, heaters.N_ACTUATORS)
                        )
            right_current = right_state.currents_a * (
                            1.0 + rng.normal(0.0, relative_noise, heaters.N_ACTUATORS)
                        )
            return {"left" : [left_voltage, left_current],
                    "right": [right_voltage, right_current]}

    def parameter_expressions(self):
        """Return the twelve explicit numerical thermo-optic response expressions."""
    
        left_expressions = self.left_PIC.parameter_expressions()
        right_expressions = self.right_PIC.parameter_expressions()

        return {"left" : left_expressions, "right" : right_expressions}

    def measure(
            self,
            input_state_vector: npt.ArrayLike,
            left_controls : FloatArray, 
            right_controls: FloatArray,
            n_shots: int,
            *,
            rng: Optional[np.random.Generator] = None,
            seed: Optional[int] = None,
            renormalize_state: bool = False,
            reset: bool = True,
            record_shots: bool = False,
        ) -> detectors.DetectionResult:

        output_state_vector = self.propagate_state(input_state_vector, left_controls, right_controls)

        detectors_array = self.detectors

        detection_result = detectors_array.measure(output_state_vector,
                                                n_shots = n_shots,
                                                rng = rng,
                                                seed = seed,
                                                renormalize_state = renormalize_state,
                                                reset = reset,
                                                record_shots = record_shots,)

        return detection_result 