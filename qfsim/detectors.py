"""
Eight-channel photon-counting detector for a U(4) x U(4) output state.

We assume that the  input is a normalised 16-component two-photon state vector

    psi = [c_00, c_01, ..., c_03, c_10, ..., c_33]

in row-major order.  The coefficient c_ab is the probability amplitude for
one photon in output ``a`` of the left U(4) and one photon in output ``b`` of
the right U(4).  The hardware therefore needs eight detectors (physical ports
L1...L4 and R1...R4), while the 16 outcomes are obtained as left-right
coincidences.  Code indices ``a,b = 0,...,3`` correspond to physical ports
``a+1,b+1``.

This module models threshold single-photon detectors.  It includes:

* nonuniform quantum efficiency;
* nonuniform optical transmission between chip and detector;
* Poisson dark counts;
* non-paralyzable dead time;
* Gaussian timing jitter and finite time-tag resolution;
* a finite coincidence window;
* phenomenological afterpulsing;
* optional detector-to-detector crosstalk;
* threshold saturation (at most one registered click per detector per gate).

It does not add analog gain, Gaussian read noise, or ADC
quantization, those can be added by hand if necessary. 
(In fact, I beleive those belong to a linear photodiode/power-meter model,
 not to a time-tagged SPAD/SNSPD coincidence model.)

Example
-------
``output_state`` can be the 16-component state already returned by the pair of
``SteadyStateU4Chip`` objects.  The following detector model is an independent entity, 
it does not assume anything about what's happening inside the chip (i.e. any other file). 

    import numpy as np

    psi_out = np.zeros(16, dtype=complex)
    psi_out[4 * 1 + 2] = 1.0       # certain physical L2-R3 output

    detector = U4xU4DetectorArray(
        DetectorParameters(
            quantum_efficiency=0.25,
            optical_transmission=0.90,
            dark_count_rate_hz=1_000.0,
            dead_time_s=20e-6,
            timing_jitter_std_s=50e-12,
            afterpulse_probability=0.01,
            afterpulse_decay_time_s=5e-6,
        ),
        AcquisitionParameters(
            gate_width_s=2e-9,
            shot_period_s=100e-6,
            coincidence_window_s=500e-12,
            time_tag_resolution_s=10e-12,
        ),
    )

    result = detector.measure(psi_out, n_shots=100_000, seed=7)
    print(result.coincidence_counts)
    print(result.conditional_coincidence_distribution())
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Union

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
ArrayParameter = Union[float, npt.ArrayLike]

N_LEFT = 4
N_RIGHT = 4
N_DETECTORS = N_LEFT + N_RIGHT
N_JOINT_OUTCOMES = N_LEFT * N_RIGHT
# Physical port labels are one-based; numpy indices a,b remain zero-based.
DETECTOR_LABELS = ("L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4")


class EventType(IntEnum):
    """Origin of the click ultimately registered by a threshold detector."""

    NONE = 0
    PHOTON = 1
    DARK = 2
    AFTERPULSE = 3
    CROSSTALK = 4


def _broadcast_detector_parameter(
    value: ArrayParameter,
    name: str,
    *,
    minimum: float = 0.0,
    maximum: Optional[float] = None,
    strictly_positive: bool = False,
) -> FloatArray:
    """Return a validated length-eight detector-parameter array."""

    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(N_DETECTORS, float(array), dtype=float)
    elif array.shape == (N_DETECTORS,):
        array = array.astype(float, copy=True)
    else:
        raise ValueError(
            f"{name} must be a scalar or have shape ({N_DETECTORS},); "
            f"received shape {array.shape}."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if strictly_positive:
        if np.any(array <= minimum):
            raise ValueError(f"Every {name} entry must be > {minimum}.")
    elif np.any(array < minimum):
        raise ValueError(f"Every {name} entry must be >= {minimum}.")
    if maximum is not None and np.any(array > maximum):
        raise ValueError(f"Every {name} entry must be <= {maximum}.")
    return array


@dataclass(frozen=True)
class DetectorParameters:
    """
    Imperfections of detectors ``(L1,L2,L3,L4,R1,R2,R3,R4)``.

    Every field except ``crosstalk_probability`` may be either a scalar, which
    is shared by all eight channels, or an array of length eight. That is if we want 
    we can account for unique imperfections of each detector. 

    ``quantum_efficiency`` is the probability that a photon arriving at the
    active detector produces a primary avalanche, that is it gets clicked.
    
    ``optical_transmission`` describes propagation/coupling loss after the chip. 
    The product of ``quantum_efficiency`` and ``optical_transmission``
    is the effective per-channel signal-detection probability.

    ``crosstalk_probability[i,j]`` is the conditional probability that a click
    in detector i immediately induces a click in detector j. Only one crosstalk generation is simulated,
    so a crosstalk click does not recursively trigger further crosstalk.
    """

    quantum_efficiency: ArrayParameter = 1.0
    optical_transmission: ArrayParameter = 1.0
    dark_count_rate_hz: ArrayParameter = 0.0
    dead_time_s: ArrayParameter = 0.0
    timing_jitter_std_s: ArrayParameter = 0.0
    afterpulse_probability: ArrayParameter = 0.0
    afterpulse_decay_time_s: ArrayParameter = 1.0
    crosstalk_probability: Optional[npt.ArrayLike] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "quantum_efficiency",
            _broadcast_detector_parameter(
                self.quantum_efficiency,
                "quantum_efficiency",
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "optical_transmission",
            _broadcast_detector_parameter(
                self.optical_transmission,
                "optical_transmission",
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "dark_count_rate_hz",
            _broadcast_detector_parameter(
                self.dark_count_rate_hz,
                "dark_count_rate_hz",
            ),
        )
        object.__setattr__(
            self,
            "dead_time_s",
            _broadcast_detector_parameter(self.dead_time_s, "dead_time_s"),
        )
        object.__setattr__(
            self,
            "timing_jitter_std_s",
            _broadcast_detector_parameter(
                self.timing_jitter_std_s,
                "timing_jitter_std_s",
            ),
        )
        object.__setattr__(
            self,
            "afterpulse_probability",
            _broadcast_detector_parameter(
                self.afterpulse_probability,
                "afterpulse_probability",
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "afterpulse_decay_time_s",
            _broadcast_detector_parameter(
                self.afterpulse_decay_time_s,
                "afterpulse_decay_time_s",
                strictly_positive=True,
            ),
        )

        if self.crosstalk_probability is None:
            crosstalk = np.zeros((N_DETECTORS, N_DETECTORS), dtype=float)
        else:
            crosstalk = np.asarray(self.crosstalk_probability, dtype=float)
            if crosstalk.shape != (N_DETECTORS, N_DETECTORS):
                raise ValueError(
                    "crosstalk_probability must have shape "
                    f"({N_DETECTORS}, {N_DETECTORS})."
                )
            crosstalk = crosstalk.copy()
            if not np.all(np.isfinite(crosstalk)):
                raise ValueError(
                    "crosstalk_probability must contain only finite values."
                )
            if np.any((crosstalk < 0.0) | (crosstalk > 1.0)):
                raise ValueError(
                    "Every crosstalk_probability entry must lie in [0, 1]."
                )
        np.fill_diagonal(crosstalk, 0.0)
        object.__setattr__(self, "crosstalk_probability", crosstalk)

    @property
    def effective_signal_efficiency(self) -> FloatArray:
        """Per-channel probability of registering an incident photon."""

        return self.quantum_efficiency * self.optical_transmission


@dataclass(frozen=True)
class AcquisitionParameters:
    """Timing parameters for a sequence of detector gates.

    ``coincidence_window_s`` is the maximum allowed absolute time-tag
    separation, hence, a pair is coincident when ``abs(t_left - t_right) <= window``.
    The photon-pair emission time is at the centre of every gate before detector
    timing jitter is applied.  Coincidences are formed only between clicks in
    the same shot/gate, this is appropriate for a pulsed experiment whose repetition
    period is much longer than the coincidence window.
    """

    gate_width_s: float = 2e-9
    shot_period_s: float = 1e-6
    coincidence_window_s: float = 500e-12
    time_tag_resolution_s: float = 0.0

    def __post_init__(self) -> None:
        scalar_fields = (
            "gate_width_s",
            "shot_period_s",
            "coincidence_window_s",
            "time_tag_resolution_s",
        )
        for name in scalar_fields:
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)

        if self.gate_width_s <= 0.0:
            raise ValueError("gate_width_s must be positive.")
        if self.shot_period_s < self.gate_width_s:
            raise ValueError("shot_period_s must be at least gate_width_s.")
        if self.coincidence_window_s < 0.0:
            raise ValueError("coincidence_window_s cannot be negative.")
        if self.time_tag_resolution_s < 0.0:
            raise ValueError("time_tag_resolution_s cannot be negative.")


@dataclass(frozen=True)
class ShotRecords:
    """Optional per-shot records; allocated only when explicitly requested."""

    true_output_pairs: IntArray
    clicks: BoolArray
    time_tags_s: FloatArray
    event_types: npt.NDArray[np.int8]


@dataclass(frozen=True)
class DetectionResult:
    """Aggregate output of an eight-detector acquisition."""

    n_shots: int
    ideal_joint_probabilities: FloatArray
    true_pair_counts: IntArray
    singles_counts: IntArray
    coincidence_counts: IntArray
    unique_coincidence_counts: IntArray
    true_signal_coincidence_counts: IntArray
    accidental_coincidence_counts: IntArray
    photon_click_counts: IntArray
    dark_click_counts: IntArray
    afterpulse_click_counts: IntArray
    crosstalk_click_counts: IntArray
    missed_efficiency_counts: IntArray
    dead_time_blocked_photon_counts: IntArray
    zero_click_shots: int
    no_coincidence_shots: int
    ambiguous_coincidence_shots: int
    records: Optional[ShotRecords] = None

    @property
    def left_singles_counts(self) -> IntArray:
        return self.singles_counts[:N_LEFT]

    @property
    def right_singles_counts(self) -> IntArray:
        return self.singles_counts[N_LEFT:]

    @property
    def ideal_left_marginal(self) -> FloatArray:
        """Ideal probability that the left photon exits each left port."""

        return np.sum(self.ideal_joint_probabilities, axis=1)

    @property
    def ideal_right_marginal(self) -> FloatArray:
        """Ideal probability that the right photon exits each right port."""

        return np.sum(self.ideal_joint_probabilities, axis=0)

    @property
    def singles_probability_per_shot(self) -> FloatArray:
        return self.singles_counts / self.n_shots

    @property
    def coincidence_probability_per_shot(self) -> FloatArray:
        """Raw coincidence counts per emitted-pair shot.

        Entries need not sum to one: missed shots contribute zero and a shot
        with multiple dark/afterpulse clicks can contribute more than one pair.
        """

        return self.coincidence_counts / self.n_shots

    def conditional_coincidence_distribution(
        self,
        *,
        unique_only: bool = False,
    ) -> FloatArray:
        """Normalize observed coincidences to a 4 x 4 distribution.

        This is a *postselected raw* distribution.  It is not automatically
        corrected for channel efficiency, dark counts, or dead time.
        """

        counts = (
            self.unique_coincidence_counts
            if unique_only
            else self.coincidence_counts
        )
        total = int(np.sum(counts))
        if total == 0:
            return np.full((N_LEFT, N_RIGHT), np.nan, dtype=float)
        return counts / total


def state_vector_to_joint_probabilities(
    output_state_vector: npt.ArrayLike,
    *,
    renormalize: bool = False,
    normalization_tolerance: float = 1e-8,
) -> FloatArray:
    """Map a 16-component state to its row-major 4 x 4 joint distribution.

    By default the input must already be normalized.  Setting ``renormalize``
    conditions on the state represented by the nonzero vector.  It must not be
    used to hide physical loss when singles statistics matter: a nonunit norm
    two-photon vector does not specify the one-photon loss branches.
    """

    psi = np.asarray(output_state_vector, dtype=complex)
    if psi.size != N_JOINT_OUTCOMES:
        raise ValueError(
            "output_state_vector must contain exactly 16 coefficients; "
            f"received {psi.size}."
        )
    psi = np.ravel(psi, order="C").astype(complex, copy=False)
    if not np.all(np.isfinite(psi.real)) or not np.all(np.isfinite(psi.imag)):
        raise ValueError("output_state_vector must contain only finite values.")

    norm_squared = float(np.vdot(psi, psi).real)
    if norm_squared <= 0.0:
        raise ValueError("output_state_vector must have nonzero norm.")
    if not renormalize and not np.isclose(
        norm_squared,
        1.0,
        rtol=normalization_tolerance,
        atol=normalization_tolerance,
    ):
        raise ValueError(
            "output_state_vector is not normalized: "
            f"sum(abs(psi)**2)={norm_squared:.12g}. Pass renormalize=True only "
            "if conditioning on pair survival is physically intended."
        )

    probabilities = np.abs(psi.reshape(N_LEFT, N_RIGHT)) ** 2
    probabilities /= norm_squared
    # Force exact normalization for numpy.random.Generator.choice.
    probabilities /= np.sum(probabilities)
    return probabilities.astype(float, copy=False)


class U4xU4DetectorArray:
    """Stateful eight-channel threshold-detector and coincidence simulator."""

    def __init__(
        self,
        detectors: Optional[DetectorParameters] = None,
        acquisition: Optional[AcquisitionParameters] = None,
    ) -> None:
        self.detectors = detectors or DetectorParameters()
        self.acquisition = acquisition or AcquisitionParameters()
        self.reset()

    def reset(self) -> None:
        """Clear dead-time/afterpulse history and restart acquisition at t=0."""

        self._next_live_time_s = np.full(N_DETECTORS, -np.inf, dtype=float)
        self._last_avalanche_time_s = np.full(N_DETECTORS, -np.inf, dtype=float)
        self._next_gate_start_s = 0.0

    def measure(
        self,
        output_state_vector: npt.ArrayLike,
        n_shots: int,
        *,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        renormalize_state: bool = False,
        reset: bool = True,
        record_shots: bool = False,
    ) -> DetectionResult:
        """Simulate sequential emitted-pair shots.

        Each shot first samples exactly one physical output pair ``(a,b)`` from
        ``abs(c_ab)**2``.  It then simulates the eight detectors during one
        time gate.  Set ``reset=False`` only when this call should continue the
        dead-time and afterpulse history of a preceding call.
        """

        if isinstance(n_shots, bool) or int(n_shots) != n_shots:
            raise ValueError("n_shots must be a positive integer.")
        n_shots = int(n_shots)
        if n_shots <= 0:
            raise ValueError("n_shots must be a positive integer.")
        if rng is not None and seed is not None:
            raise ValueError("Pass either rng or seed, not both.")
        if rng is None:
            rng = np.random.default_rng(seed)
        if reset:
            self.reset()

        joint_probabilities = state_vector_to_joint_probabilities(
            output_state_vector,
            renormalize=renormalize_state,
        )
        sampled_flat_pairs = rng.choice(
            N_JOINT_OUTCOMES,
            size=n_shots,
            p=joint_probabilities.ravel(),
        )
        sampled_left = sampled_flat_pairs // N_RIGHT
        sampled_right = sampled_flat_pairs % N_RIGHT

        true_pair_counts = np.zeros((N_LEFT, N_RIGHT), dtype=np.int64)
        singles_counts = np.zeros(N_DETECTORS, dtype=np.int64)
        coincidence_counts = np.zeros((N_LEFT, N_RIGHT), dtype=np.int64)
        unique_coincidence_counts = np.zeros_like(coincidence_counts)
        true_signal_coincidence_counts = np.zeros_like(coincidence_counts)
        accidental_coincidence_counts = np.zeros_like(coincidence_counts)

        photon_click_counts = np.zeros(N_DETECTORS, dtype=np.int64)
        dark_click_counts = np.zeros(N_DETECTORS, dtype=np.int64)
        afterpulse_click_counts = np.zeros(N_DETECTORS, dtype=np.int64)
        crosstalk_click_counts = np.zeros(N_DETECTORS, dtype=np.int64)
        missed_efficiency_counts = np.zeros(N_DETECTORS, dtype=np.int64)
        blocked_photon_counts = np.zeros(N_DETECTORS, dtype=np.int64)

        zero_click_shots = 0
        no_coincidence_shots = 0
        ambiguous_coincidence_shots = 0

        if record_shots:
            record_pairs = np.column_stack((sampled_left, sampled_right)).astype(
                np.int64,
                copy=False,
            )
            record_clicks = np.zeros((n_shots, N_DETECTORS), dtype=bool)
            record_times = np.full((n_shots, N_DETECTORS), np.nan, dtype=float)
            record_types = np.zeros((n_shots, N_DETECTORS), dtype=np.int8)

        efficiency = self.detectors.effective_signal_efficiency
        gate_width = self.acquisition.gate_width_s
        period = self.acquisition.shot_period_s
        coincidence_window = self.acquisition.coincidence_window_s
        resolution = self.acquisition.time_tag_resolution_s

        for shot in range(n_shots):
            left_output = int(sampled_left[shot])
            right_output = int(sampled_right[shot])
            true_pair_counts[left_output, right_output] += 1

            gate_start = self._next_gate_start_s + shot * period
            gate_end = gate_start + gate_width
            emission_time = gate_start + 0.5 * gate_width

            raw_times = np.full(N_DETECTORS, np.nan, dtype=float)
            event_types = np.zeros(N_DETECTORS, dtype=np.int8)
            photon_targets = (left_output, N_LEFT + right_output)

            for detector in range(N_DETECTORS):
                live_start = max(gate_start, self._next_live_time_s[detector])
                candidates: list[tuple[float, EventType]] = []

                if detector in photon_targets:
                    if emission_time < live_start:
                        blocked_photon_counts[detector] += 1
                    elif rng.random() < efficiency[detector]:
                        candidates.append((emission_time, EventType.PHOTON))
                    else:
                        missed_efficiency_counts[detector] += 1

                live_duration = gate_end - live_start
                if live_duration > 0.0:
                    dark_rate = self.detectors.dark_count_rate_hz[detector]
                    if dark_rate > 0.0:
                        dark_wait = rng.exponential(1.0 / dark_rate)
                        if dark_wait <= live_duration:
                            candidates.append(
                                (live_start + dark_wait, EventType.DARK)
                            )

                    last_avalanche = self._last_avalanche_time_s[detector]
                    if np.isfinite(last_avalanche):
                        elapsed = max(0.0, live_start - last_avalanche)
                        afterpulse_probability = (
                            self.detectors.afterpulse_probability[detector]
                            * np.exp(
                                -elapsed
                                / self.detectors.afterpulse_decay_time_s[detector]
                            )
                        )
                        if rng.random() < afterpulse_probability:
                            afterpulse_time = (
                                live_start + rng.random() * live_duration
                            )
                            candidates.append(
                                (afterpulse_time, EventType.AFTERPULSE)
                            )

                if candidates:
                    event_time, event_type = min(
                        candidates,
                        key=lambda item: item[0],
                    )
                    raw_times[detector] = event_time
                    event_types[detector] = int(event_type)

            # One nonrecursive generation of prompt detector-array crosstalk.
            primary_detectors = np.flatnonzero(np.isfinite(raw_times))
            for source in primary_detectors:
                for target in range(N_DETECTORS):
                    if np.isfinite(raw_times[target]):
                        continue
                    probability = self.detectors.crosstalk_probability[source, target]
                    if probability <= 0.0 or rng.random() >= probability:
                        continue
                    induced_time = raw_times[source]
                    if induced_time < self._next_live_time_s[target]:
                        continue
                    raw_times[target] = induced_time
                    event_types[target] = int(EventType.CROSSTALK)

            clicks = np.isfinite(raw_times)
            if not np.any(clicks):
                zero_click_shots += 1

            # Dead time depends on the physical avalanche time, before electronic
            # time-tag quantization.
            for detector in np.flatnonzero(clicks):
                avalanche_time = raw_times[detector]
                self._last_avalanche_time_s[detector] = avalanche_time
                self._next_live_time_s[detector] = (
                    avalanche_time + self.detectors.dead_time_s[detector]
                )

            # Timing jitter is an uncertainty in the recorded avalanche time,
            # not a displacement of the physical photon-arrival time.  It is
            # therefore applied after dead-time evolution and to every kind of
            # registered avalanche, including dark counts and afterpulses.
            time_tags = raw_times.copy()
            time_tags[clicks] += rng.normal(
                0.0,
                self.detectors.timing_jitter_std_s[clicks],
            )
            if resolution > 0.0:
                time_tags[clicks] = gate_start + resolution * np.round(
                    (time_tags[clicks] - gate_start) / resolution
                )

            singles_counts += clicks
            photon_click_counts += event_types == int(EventType.PHOTON)
            dark_click_counts += event_types == int(EventType.DARK)
            afterpulse_click_counts += event_types == int(EventType.AFTERPULSE)
            crosstalk_click_counts += event_types == int(EventType.CROSSTALK)

            coincident_pairs: list[tuple[int, int]] = []
            left_clicks = np.flatnonzero(clicks[:N_LEFT])
            right_clicks = np.flatnonzero(clicks[N_LEFT:])
            for left_detector in left_clicks:
                for right_detector in right_clicks:
                    right_global = N_LEFT + right_detector
                    if (
                        abs(
                            time_tags[left_detector]
                            - time_tags[right_global]
                        )
                        <= coincidence_window
                    ):
                        pair = (int(left_detector), int(right_detector))
                        coincident_pairs.append(pair)
                        coincidence_counts[pair] += 1
                        is_true_signal = (
                            pair == (left_output, right_output)
                            and event_types[left_detector] == int(EventType.PHOTON)
                            and event_types[right_global] == int(EventType.PHOTON)
                        )
                        if is_true_signal:
                            true_signal_coincidence_counts[pair] += 1
                        else:
                            accidental_coincidence_counts[pair] += 1

            if len(coincident_pairs) == 0:
                no_coincidence_shots += 1
            elif len(coincident_pairs) == 1:
                unique_coincidence_counts[coincident_pairs[0]] += 1
            else:
                ambiguous_coincidence_shots += 1

            if record_shots:
                record_clicks[shot] = clicks
                record_times[shot] = time_tags
                record_types[shot] = event_types

        self._next_gate_start_s += n_shots * period

        records = None
        if record_shots:
            records = ShotRecords(
                true_output_pairs=record_pairs,
                clicks=record_clicks,
                time_tags_s=record_times,
                event_types=record_types,
            )

        return DetectionResult(
            n_shots=n_shots,
            ideal_joint_probabilities=joint_probabilities,
            true_pair_counts=true_pair_counts,
            singles_counts=singles_counts,
            coincidence_counts=coincidence_counts,
            unique_coincidence_counts=unique_coincidence_counts,
            true_signal_coincidence_counts=true_signal_coincidence_counts,
            accidental_coincidence_counts=accidental_coincidence_counts,
            photon_click_counts=photon_click_counts,
            dark_click_counts=dark_click_counts,
            afterpulse_click_counts=afterpulse_click_counts,
            crosstalk_click_counts=crosstalk_click_counts,
            missed_efficiency_counts=missed_efficiency_counts,
            dead_time_blocked_photon_counts=blocked_photon_counts,
            zero_click_shots=zero_click_shots,
            no_coincidence_shots=no_coincidence_shots,
            ambiguous_coincidence_shots=ambiguous_coincidence_shots,
            records=records,
        )


__all__ = [
    "AcquisitionParameters",
    "DETECTOR_LABELS",
    "DetectionResult",
    "DetectorParameters",
    "EventType",
    "ShotRecords",
    "U4xU4DetectorArray",
    "state_vector_to_joint_probabilities",
]
