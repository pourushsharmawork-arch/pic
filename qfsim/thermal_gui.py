"""Live thermal dashboard for :mod:`heaters`.

This module is deliberately separate from ``heaters.py``.  It only uses the
public data structures and steady-state solver already provided by that module;
the original heater model does not need to be edited.

The dashboard has one persistent Matplotlib figure.  Slider changes,
decomposition changes, heat maps, bars, and time traces update existing artists
in place; no frame recreates the figure or its axes.

The heater package defines only a steady-state relation.  To visualize a time
evolution, this module adds the reduced first-order model

    d(Delta T_h)/dt = (T_proposed_h(Delta T, u) - Delta T_h) / tau_h,

where

    T_proposed = H @ [V(u)^2 / (R0 * (1 + alpha_R * Delta T))].

Its fixed points are exactly the nonlinear steady states calculated by
``solve_steady_thermal_state``.  The time constants are visualization/model
parameters because they cannot be inferred from the static Green matrix H.

Typical use
-----------

Place this file beside ``heaters.py`` inside the same package, then run

    from your_package.thermal_gui import launch_thermal_gui
    launch_thermal_gui()

Existing chip instances can be supplied instead:

    launch_thermal_gui(chip_1212=chip_a, chip_2121=chip_b)

Dependencies
------------

NumPy and Matplotlib are required.  No GUI framework beyond the active
Matplotlib backend is used.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Final

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button, RadioButtons, Slider
import numpy as np
from numpy.typing import NDArray

from .heaters import N_ACTUATORS, PIC_Structure, SteadyStateU4Chip


FloatArray = NDArray[np.float64]

DECOMPOSITIONS: Final[tuple[str, str]] = ("1212", "2121")


class LiveThermalGUI:
    """Persistent, live dashboard for the two U4 heater decompositions.

    Parameters
    ----------
    chips:
        Mapping containing one ``SteadyStateU4Chip`` for ``"1212"`` and one
        for ``"2121"``.
    initial_decomposition:
        Decomposition displayed when the window opens.
    initial_controls:
        Optional normalized control vector with shape ``(12,)`` in the
        actuator ordering of ``initial_decomposition``.
    thermal_time_constant_s:
        Either one positive time constant shared by all thermal sites or a
        mapping from actuator name to positive time constant.  This parameter
        governs the transient visualization, not the steady-state chip model.
    timer_interval_ms:
        Interval between live updates.
    history_seconds:
        Width of the scrolling temperature-history window.
    field_width_um:
        Gaussian width used only to interpolate the twelve discrete thermal
        sites into the displayed two-dimensional heat map.
    """

    def __init__(
        self,
        chips: Mapping[str, SteadyStateU4Chip],
        *,
        initial_decomposition: str = "1212",
        initial_controls: FloatArray | None = None,
        thermal_time_constant_s: float | Mapping[str, float] = 0.20,
        timer_interval_ms: int = 40,
        history_seconds: float = 12.0,
        field_width_um: float = 190.0,
    ) -> None:
        self._chips = dict(chips)
        self._validate_chips()

        if initial_decomposition not in DECOMPOSITIONS:
            raise ValueError(
                f"initial_decomposition must be one of {DECOMPOSITIONS}"
            )
        if timer_interval_ms <= 0:
            raise ValueError("timer_interval_ms must be positive")
        if history_seconds <= 0.0:
            raise ValueError("history_seconds must be positive")
        if field_width_um <= 0.0:
            raise ValueError("field_width_um must be positive")

        self._time_constant_spec = thermal_time_constant_s
        self._validate_time_constants()
        self._timer_interval_ms = int(timer_interval_ms)
        self._dt_s = self._timer_interval_ms / 1000.0
        self._history_seconds = float(history_seconds)
        self._field_width_um = float(field_width_um)

        self._decomposition = initial_decomposition
        self._chip = self._chips[self._decomposition]
        self._actuator_names = self._names_for(self._decomposition)

        if initial_controls is None:
            controls = np.zeros(N_ACTUATORS, dtype=float)
        else:
            controls = np.asarray(initial_controls, dtype=float)
            if controls.shape != (N_ACTUATORS,):
                raise ValueError("initial_controls must have shape (12,)")
            if np.any((controls < 0.0) | (controls > 1.0)):
                raise ValueError("initial controls must lie in [0, 1]")

        self._controls = controls.copy()
        self._temperature_k = np.zeros(N_ACTUATORS, dtype=float)
        self._target_state = self._chip.steady_state(self._controls)
        self._elapsed_s = 0.0
        self._running = True
        self._suspend_callbacks = False

        history_length = max(
            2,
            int(np.ceil(self._history_seconds / self._dt_s)) + 1,
        )
        self._history_t: deque[float] = deque(maxlen=history_length)
        self._history_temperature: deque[FloatArray] = deque(
            maxlen=history_length
        )
        self._history_t.append(0.0)
        self._history_temperature.append(self._temperature_k.copy())

        self._build_figure()
        self._configure_decomposition_artists()
        self._update_artists(force_rescale=True)

        self._timer = self.figure.canvas.new_timer(
            interval=self._timer_interval_ms
        )
        self._timer.add_callback(self._on_timer)

    @property
    def figure(self):
        """The persistent Matplotlib figure owned by the dashboard."""

        return self._figure

    @property
    def controls(self) -> FloatArray:
        """A copy of the currently requested normalized controls."""

        return self._controls.copy()

    @property
    def temperature_rise_k(self) -> FloatArray:
        """A copy of the current transient temperature-rise vector."""

        return self._temperature_k.copy()

    @property
    def decomposition(self) -> str:
        """The decomposition currently displayed by the dashboard."""

        return self._decomposition

    def show(self, *, block: bool = True) -> "LiveThermalGUI":
        """Start live updates and display the existing dashboard window."""

        self._timer.start()
        plt.show(block=block)
        return self

    def set_controls(self, controls: FloatArray) -> None:
        """Update all controls without rebuilding any interface component."""

        requested = np.asarray(controls, dtype=float)
        if requested.shape != (N_ACTUATORS,):
            raise ValueError("controls must have shape (12,)")
        if np.any((requested < 0.0) | (requested > 1.0)):
            raise ValueError("controls must lie in [0, 1]")

        self._controls = requested.copy()
        self._suspend_callbacks = True
        try:
            for slider, value in zip(self._sliders, self._controls):
                slider.set_val(float(value))
        finally:
            self._suspend_callbacks = False

        self._target_state = self._chip.steady_state(self._controls)
        self._update_artists()
        self.figure.canvas.draw_idle()

    def close(self) -> None:
        """Stop live updates and close the dashboard figure."""

        self._timer.stop()
        plt.close(self.figure)

    def _validate_chips(self) -> None:
        for decomposition in DECOMPOSITIONS:
            if decomposition not in self._chips:
                raise ValueError(
                    f"chips must contain a '{decomposition}' device"
                )
            chip = self._chips[decomposition]
            if chip.decomp != decomposition:
                raise ValueError(
                    f"chip stored as '{decomposition}' has decomp="
                    f"'{chip.decomp}'"
                )

    def _validate_time_constants(self) -> None:
        specification = self._time_constant_spec
        if np.isscalar(specification):
            if float(specification) <= 0.0:
                raise ValueError("thermal_time_constant_s must be positive")
            return

        expected = set(self._names_for("1212"))
        supplied = set(specification)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise ValueError(
                "time-constant mapping must contain every actuator name; "
                f"missing={missing}, extra={extra}"
            )
        if any(float(value) <= 0.0 for value in specification.values()):
            raise ValueError("all thermal time constants must be positive")

    @staticmethod
    def _names_for(decomposition: str) -> tuple[str, ...]:
        return tuple(PIC_Structure(decomposition)["ACTUATOR_NAMES"])

    def _time_constants(self) -> FloatArray:
        specification = self._time_constant_spec
        if np.isscalar(specification):
            return np.full(N_ACTUATORS, float(specification))
        return np.array(
            [float(specification[name]) for name in self._actuator_names],
            dtype=float,
        )

    def _build_figure(self) -> None:
        self._figure = plt.figure(figsize=(15.5, 9.2))
        self._figure.patch.set_facecolor("#10151d")
        manager = self._figure.canvas.manager
        if hasattr(manager, "set_window_title"):
            manager.set_window_title("U4 live electro-thermal dashboard")

        self._ax_heat = self._figure.add_axes([0.055, 0.52, 0.415, 0.405])
        self._ax_colorbar = self._figure.add_axes([0.482, 0.52, 0.012, 0.405])
        self._ax_bars = self._figure.add_axes([0.545, 0.52, 0.235, 0.405])
        self._ax_history = self._figure.add_axes([0.055, 0.085, 0.725, 0.335])
        self._ax_radio = self._figure.add_axes([0.815, 0.825, 0.145, 0.09])
        self._ax_pause = self._figure.add_axes([0.815, 0.772, 0.068, 0.035])
        self._ax_settle = self._figure.add_axes([0.892, 0.772, 0.068, 0.035])
        self._ax_reset = self._figure.add_axes([0.815, 0.727, 0.145, 0.035])

        for axis in (
            self._ax_heat,
            self._ax_bars,
            self._ax_history,
            self._ax_colorbar,
            self._ax_radio,
            self._ax_pause,
            self._ax_settle,
            self._ax_reset,
        ):
            axis.set_facecolor("#17202b")

        self._heat_image = self._ax_heat.imshow(
            np.zeros((100, 180), dtype=float),
            origin="lower",
            aspect="auto",
            interpolation="bilinear",
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
        )
        self._connection_lines = LineCollection(
            [], colors="#91a4b7", linewidths=1.2, alpha=0.65
        )
        self._ax_heat.add_collection(self._connection_lines)
        self._heater_scatter = self._ax_heat.scatter(
            np.zeros(N_ACTUATORS),
            np.zeros(N_ACTUATORS),
            c=np.zeros(N_ACTUATORS),
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
            edgecolors="#f4f7fb",
            linewidths=0.8,
            s=72,
            zorder=4,
        )
        self._heater_labels = [
            self._ax_heat.text(
                0.0,
                0.0,
                "",
                color="#f4f7fb",
                fontsize=7,
                ha="left",
                va="bottom",
                zorder=5,
            )
            for _ in range(N_ACTUATORS)
        ]
        self._colorbar = self._figure.colorbar(
            self._heat_image, cax=self._ax_colorbar
        )
        self._colorbar.ax.set_title("ΔT (K)", color="#d8e0e8", fontsize=8, pad=7)
        self._colorbar.ax.tick_params(colors="#d8e0e8")

        x = np.arange(N_ACTUATORS)
        colors = ["#50b7ff"] * 6 + ["#f48fb1"] * 6
        self._bars = self._ax_bars.bar(x, np.zeros(N_ACTUATORS), color=colors)
        (self._target_markers,) = self._ax_bars.plot(
            x,
            np.zeros(N_ACTUATORS),
            linestyle="none",
            marker="_",
            markersize=12,
            markeredgewidth=2.0,
            color="#ffffff",
            label="steady target",
        )
        self._ax_bars.legend(
            loc="upper right", frameon=False, labelcolor="#d8e0e8", fontsize=8
        )

        history_colors = plt.cm.turbo(np.linspace(0.04, 0.96, N_ACTUATORS))
        self._history_lines = []
        for index in range(N_ACTUATORS):
            (line,) = self._ax_history.plot(
                [],
                [],
                color=history_colors[index],
                linewidth=1.2,
                alpha=0.85,
            )
            self._history_lines.append(line)

        self._radio = RadioButtons(
            self._ax_radio,
            DECOMPOSITIONS,
            active=DECOMPOSITIONS.index(self._decomposition),
            activecolor="#50b7ff",
        )
        self._radio.on_clicked(self._on_decomposition_selected)
        for label in self._radio.labels:
            label.set_color("#d8e0e8")

        self._pause_button = Button(
            self._ax_pause, "Pause", color="#263545", hovercolor="#36506a"
        )
        self._settle_button = Button(
            self._ax_settle, "Settle", color="#263545", hovercolor="#36506a"
        )
        self._reset_button = Button(
            self._ax_reset,
            "Set controls to zero",
            color="#263545",
            hovercolor="#36506a",
        )
        self._pause_button.on_clicked(self._toggle_running)
        self._settle_button.on_clicked(self._settle_immediately)
        self._reset_button.on_clicked(self._reset_controls)
        for button in (
            self._pause_button,
            self._settle_button,
            self._reset_button,
        ):
            button.label.set_color("#e7edf3")

        self._sliders: list[Slider] = []
        slider_top = 0.685
        slider_step = 0.0475
        for index in range(N_ACTUATORS):
            slider_axis = self._figure.add_axes(
                [0.835, slider_top - index * slider_step, 0.125, 0.019]
            )
            slider_axis.set_facecolor("#263545")
            slider = Slider(
                slider_axis,
                "",
                0.0,
                1.0,
                valinit=float(self._controls[index]),
                valstep=0.001,
                color="#50b7ff" if index < 6 else "#f48fb1",
            )
            slider.label.set_fontsize(7.5)
            slider.label.set_color("#d8e0e8")
            slider.valtext.set_fontsize(7.0)
            slider.valtext.set_color("#d8e0e8")
            slider.on_changed(
                lambda value, actuator_index=index: self._on_slider(
                    actuator_index, value
                )
            )
            self._sliders.append(slider)

        self._status_text = self._figure.text(
            0.805,
            0.035,
            "",
            color="#d8e0e8",
            fontsize=8,
            family="monospace",
            ha="left",
            va="bottom",
        )

        self._style_data_axis(self._ax_heat)
        self._style_data_axis(self._ax_bars)
        self._style_data_axis(self._ax_history)
        self._ax_radio.tick_params(colors="#d8e0e8", labelsize=9)
        self._ax_radio.set_title(
            "Decomposition", color="#d8e0e8", fontsize=9, pad=5
        )

        self._ax_heat.set_title("Interpolated live thermal field")
        self._ax_heat.set_xlabel("Chip coordinate x, µm")
        self._ax_heat.set_ylabel("Chip coordinate y, µm")
        self._ax_bars.set_title("Thermal sites")
        self._ax_bars.set_ylabel("Temperature rise, K")
        self._ax_history.set_title("Temperature history")
        self._ax_history.set_xlabel("Simulation time, s")
        self._ax_history.set_ylabel("Temperature rise, K")

    @staticmethod
    def _style_data_axis(axis) -> None:
        axis.tick_params(colors="#b9c6d3", labelsize=8)
        axis.xaxis.label.set_color("#d8e0e8")
        axis.yaxis.label.set_color("#d8e0e8")
        axis.title.set_color("#f4f7fb")
        axis.grid(color="#6b7f92", alpha=0.16, linewidth=0.7)
        for spine in axis.spines.values():
            spine.set_color("#56697c")

    def _configure_decomposition_artists(self) -> None:
        positions = np.asarray(self._chip.electrothermal.positions_um, dtype=float)
        x_min, y_min = np.min(positions, axis=0)
        x_max, y_max = np.max(positions, axis=0)
        padding = 150.0

        self._grid_x = np.linspace(x_min - padding, x_max + padding, 180)
        self._grid_y = np.linspace(y_min - padding, y_max + padding, 100)
        self._mesh_x, self._mesh_y = np.meshgrid(self._grid_x, self._grid_y)
        extent = (
            self._grid_x[0],
            self._grid_x[-1],
            self._grid_y[0],
            self._grid_y[-1],
        )
        self._heat_image.set_extent(extent)
        self._ax_heat.set_xlim(extent[0], extent[1])
        self._ax_heat.set_ylim(extent[2], extent[3])
        self._heater_scatter.set_offsets(positions)

        # Index i and i+6 are the theta and phi heaters of the same MZI.
        segments = [
            np.vstack([positions[i], positions[i + 6]]) for i in range(6)
        ]
        self._connection_lines.set_segments(segments)

        for index, (label, position, name) in enumerate(
            zip(self._heater_labels, positions, self._actuator_names)
        ):
            if index < 6:
                offset = np.array([10.0, -11.0])
                label.set_verticalalignment("top")
            else:
                offset = np.array([10.0, 11.0])
                label.set_verticalalignment("bottom")
            label.set_position(position + offset)
            label.set_text(name.replace("theta_", "θ ").replace("phi_", "φ "))

        short_names = [
            name.replace("theta_", "θ-").replace("phi_", "φ-")
            for name in self._actuator_names
        ]
        self._ax_bars.set_xticks(np.arange(N_ACTUATORS))
        self._ax_bars.set_xticklabels(short_names, rotation=70, ha="right")

        for slider, name in zip(self._sliders, short_names):
            slider.label.set_text(name)

        for line, name in zip(self._history_lines, short_names):
            line.set_label(name)

        self._figure.suptitle(
            f"U4 live electro-thermal dashboard — {self._decomposition}",
            color="#f4f7fb",
            fontsize=15,
            y=0.975,
        )

    def _on_slider(self, index: int, value: float) -> None:
        if self._suspend_callbacks:
            return
        self._controls[index] = float(value)
        self._target_state = self._chip.steady_state(self._controls)

    def _on_decomposition_selected(self, label: str) -> None:
        if self._suspend_callbacks or label == self._decomposition:
            return

        old_names = self._actuator_names
        control_by_name = dict(zip(old_names, self._controls))
        temperature_by_name = dict(zip(old_names, self._temperature_k))

        self._decomposition = label
        self._chip = self._chips[label]
        self._actuator_names = self._names_for(label)
        self._controls = np.array(
            [control_by_name[name] for name in self._actuator_names], dtype=float
        )
        self._temperature_k = np.array(
            [temperature_by_name[name] for name in self._actuator_names],
            dtype=float,
        )

        self._suspend_callbacks = True
        try:
            for slider, value in zip(self._sliders, self._controls):
                slider.set_val(float(value))
        finally:
            self._suspend_callbacks = False

        self._target_state = self._chip.steady_state(self._controls)
        self._clear_history()
        self._configure_decomposition_artists()
        self._update_artists(force_rescale=True)
        self.figure.canvas.draw_idle()

    def _toggle_running(self, _event) -> None:
        self._running = not self._running
        self._pause_button.label.set_text("Pause" if self._running else "Resume")
        self.figure.canvas.draw_idle()

    def _settle_immediately(self, _event) -> None:
        self._temperature_k = self._target_state.temperature_rise_k.copy()
        self._append_history()
        self._update_artists(force_rescale=True)
        self.figure.canvas.draw_idle()

    def _reset_controls(self, _event) -> None:
        self._suspend_callbacks = True
        try:
            for slider in self._sliders:
                slider.set_val(0.0)
        finally:
            self._suspend_callbacks = False
        self._controls.fill(0.0)
        self._target_state = self._chip.steady_state(self._controls)
        self.figure.canvas.draw_idle()

    def _clear_history(self) -> None:
        self._history_t.clear()
        self._history_temperature.clear()
        self._history_t.append(self._elapsed_s)
        self._history_temperature.append(self._temperature_k.copy())

    def _append_history(self) -> None:
        self._history_t.append(self._elapsed_s)
        self._history_temperature.append(self._temperature_k.copy())

    def _advance_temperature(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        parameters = self._chip.electrothermal
        voltage = parameters.voltage_max_v * self._controls
        resistance_factor = 1.0 + (
            parameters.resistance_tcr_per_k * self._temperature_k
        )
        if np.any(resistance_factor <= 0.05):
            raise RuntimeError("nonphysical heater resistance in live dynamics")

        resistance = parameters.resistance_0_ohm * resistance_factor
        power = voltage**2 / resistance
        proposed_temperature = parameters.thermal_green_k_per_w @ power

        # Exponential relaxation is stable for any timer interval and has the
        # same fixed point as the package's nonlinear steady-state equation.
        fraction = 1.0 - np.exp(-self._dt_s / self._time_constants())
        self._temperature_k += fraction * (
            proposed_temperature - self._temperature_k
        )
        return voltage, power, proposed_temperature

    def _current_electrical_values(self) -> tuple[FloatArray, FloatArray]:
        parameters = self._chip.electrothermal
        voltage = parameters.voltage_max_v * self._controls
        resistance = parameters.resistance_0_ohm * (
            1.0 + parameters.resistance_tcr_per_k * self._temperature_k
        )
        power = voltage**2 / resistance
        return voltage, power

    def _interpolated_field(self) -> FloatArray:
        positions = np.asarray(self._chip.electrothermal.positions_um, dtype=float)
        dx = self._mesh_x[None, :, :] - positions[:, 0, None, None]
        dy = self._mesh_y[None, :, :] - positions[:, 1, None, None]
        radius_squared = dx**2 + dy**2
        weights = np.exp(
            -0.5 * radius_squared / self._field_width_um**2
        )

        numerator = np.sum(
            weights * self._temperature_k[:, None, None], axis=0
        )
        denominator = np.maximum(1.0, np.sum(weights, axis=0))
        return numerator / denominator

    def _update_artists(self, *, force_rescale: bool = False) -> None:
        field = self._interpolated_field()
        target_temperature = self._target_state.temperature_rise_k
        maximum = max(
            0.5,
            float(np.max(field)),
            float(np.max(self._temperature_k)),
            float(np.max(target_temperature)),
        )

        self._heat_image.set_data(field)
        self._heater_scatter.set_array(self._temperature_k)
        if force_rescale or maximum > self._heat_image.norm.vmax:
            self._heat_image.set_clim(0.0, 1.08 * maximum)
            self._heater_scatter.set_clim(0.0, 1.08 * maximum)
            self._colorbar.update_normal(self._heat_image)

        for rectangle, temperature in zip(self._bars, self._temperature_k):
            rectangle.set_height(float(temperature))
        self._target_markers.set_ydata(target_temperature)
        self._ax_bars.set_ylim(0.0, 1.12 * maximum)

        times = np.asarray(self._history_t, dtype=float)
        history = np.asarray(self._history_temperature, dtype=float)
        if history.ndim == 2 and history.shape[0] > 0:
            for index, line in enumerate(self._history_lines):
                line.set_data(times, history[:, index])

        left = max(0.0, self._elapsed_s - self._history_seconds)
        right = max(self._history_seconds, self._elapsed_s + self._dt_s)
        self._ax_history.set_xlim(left, right)
        history_maximum = maximum
        if history.size:
            history_maximum = max(history_maximum, float(np.max(history)))
        self._ax_history.set_ylim(0.0, 1.12 * max(0.5, history_maximum))

        voltage, power = self._current_electrical_values()
        current_proposed = (
            self._chip.electrothermal.thermal_green_k_per_w @ power
        )
        residual = float(
            np.max(np.abs(current_proposed - self._temperature_k))
        )

        hottest_index = int(np.argmax(self._temperature_k))
        self._status_text.set_text(
            f"decomposition : {self._decomposition}\n"
            f"simulation t : {self._elapsed_s:8.3f} s\n"
            f"total power  : {1e3*np.sum(power):8.3f} mW\n"
            f"max voltage  : {np.max(voltage):8.3f} V\n"
            f"hottest site : {self._actuator_names[hottest_index]}\n"
            f"current max  : {np.max(self._temperature_k):8.3f} K\n"
            f"steady max   : {np.max(target_temperature):8.3f} K\n"
            f"thermal gap  : {residual:8.3e} K"
        )

    def _on_timer(self) -> None:
        if not plt.fignum_exists(self.figure.number):
            self._timer.stop()
            return
        if not self._running:
            return

        self._advance_temperature()
        self._elapsed_s += self._dt_s
        self._append_history()
        self._update_artists()
        self.figure.canvas.draw_idle()


def launch_thermal_gui(
    *,
    chip_1212: SteadyStateU4Chip | None = None,
    chip_2121: SteadyStateU4Chip | None = None,
    seed: int = 20260727,
    initial_decomposition: str = "1212",
    initial_controls: FloatArray | None = None,
    thermal_time_constant_s: float | Mapping[str, float] = 0.20,
    timer_interval_ms: int = 40,
    history_seconds: float = 12.0,
    field_width_um: float = 190.0,
    block: bool = True,
) -> LiveThermalGUI:
    """Construct and show one persistent dashboard for both decompositions.

    Missing chip objects are constructed with
    ``SteadyStateU4Chip.default(seed=seed, decomp=...)``.  Supplying the two
    chips is preferable when the interface must display already-configured
    physical devices rather than the package defaults.
    """

    if chip_1212 is None:
        chip_1212 = SteadyStateU4Chip.default(seed=seed, decomp="1212")
    if chip_2121 is None:
        chip_2121 = SteadyStateU4Chip.default(seed=seed, decomp="2121")

    dashboard = LiveThermalGUI(
        {"1212": chip_1212, "2121": chip_2121},
        initial_decomposition=initial_decomposition,
        initial_controls=initial_controls,
        thermal_time_constant_s=thermal_time_constant_s,
        timer_interval_ms=timer_interval_ms,
        history_seconds=history_seconds,
        field_width_um=field_width_um,
    )
    return dashboard.show(block=block)


def main() -> None:
    """Entry point for ``python -m your_package.thermal_gui``."""

    launch_thermal_gui()


if __name__ == "__main__":
    main()
