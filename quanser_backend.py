from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import exp, fmod, pi, sin
import os
import random
import time

import numpy as np


COUNT_TO_RAD = 2.0 * pi / (512.0 * 4.0)


class QuanserConnectionError(RuntimeError):
    pass


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


@dataclass
class CommandState:
    inicio: int = 0
    modooperacion: int = 0
    A: float = 0.0
    B: float = 0.0
    C: float = 0.0
    D: float = 0.0
    E: float = 0.0
    F: float = 0.0
    G: float = 0.0
    H: float = 0.0
    delay_ms: int = 10
    reference_period_ms: int = 2000
    amplitude: int = 0
    manual_reference: int = 0
    offset: int = 0
    signal_type: int = 0
    active_tab: int = 0
    Kp: float = 0.0
    Ki: float = 0.0
    Kd: float = 0.0
    deadzone: int = 0
    derivative_time_constant: float = 0.01
    pid_type: int = 0
    automatic_reference: int = 1
    reset_time: float = 0.1


class QuanserSerialEmulator:
    """Serial-like adapter that runs the old firmware logic against Qube HIL.

    GUI.py expects a pyserial object that accepts a 25-field command string and
    returns text lines: REF MEAS DT_ms CURR PWM. This class preserves that
    contract while replacing the Teensy/Arduino firmware with Quanser HIL I/O.
    """

    def __init__(
        self,
        card_type: str = "qube_servo3_usb",
        card_identifier: str = "0",
        max_voltage: float | None = None,
        output_sign: float | None = None,
        encoder_sign: float | None = None,
    ) -> None:
        try:
            from quanser.hardware import HIL
        except Exception as exc:
            raise QuanserConnectionError("Quanser HIL Python package is not available") from exc

        self.max_voltage = float(max_voltage if max_voltage is not None else os.environ.get("QUBE_MAX_VOLTAGE", "6.0"))
        self.output_sign = float(output_sign if output_sign is not None else os.environ.get("QUBE_OUTPUT_SIGN", "-1.0"))
        self.encoder_sign = float(encoder_sign if encoder_sign is not None else os.environ.get("QUBE_ENCODER_SIGN", "-1.0"))
        self.card = HIL()
        self.card.open(card_type, card_identifier)
        self.is_open = True

        self.ai_channels = np.array([0], dtype=np.uint32)
        self.ao_channels = np.array([0], dtype=np.uint32)
        self.encoder_channels = np.array([0, 1], dtype=np.uint32)
        self.digital_channels = np.array([0], dtype=np.uint32)
        self.other_input_channels = np.array([14000, 14001], dtype=np.uint32)
        self.led_channels = np.array([11000, 11001, 11002], dtype=np.uint32)
        self.led_red = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.led_green = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self.other_available = True
        self._enabled_led = False

        options = (
            "deadband_compensation=0.3;"
            "pwm_en=0;"
            "enc0_velocity=3.0;"
            "enc1_velocity=3.0;"
            "min_diode_compensation=0.3;"
            "max_diode_compensation=1.5"
        )
        try:
            self.card.set_card_specific_options(options, len(options))
        except Exception:
            pass
        try:
            self.card.set_analog_output_ranges(
                self.ao_channels,
                1,
                np.array([-15.0], dtype=np.float64),
                np.array([15.0], dtype=np.float64),
            )
        except Exception:
            pass

        self.cmd = CommandState()
        self._rx_lines: deque[bytes] = deque()
        self._last_service = time.monotonic()
        self._last_theta = 0.0
        self._theta_zero = 0.0
        self._last_command_text = ""
        self.reset_input_buffer()
        self._reset_control_memory()
        self._set_encoder_counts_zero()
        self._write_voltage(0.0, enabled=False)

    def __enter__(self) -> "QuanserSerialEmulator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def reset_input_buffer(self) -> None:
        self._rx_lines.clear()

    def stop(self) -> None:
        if self.is_open:
            self._reset_control_memory()
            self._write_voltage(0.0, enabled=False)

    def close(self) -> None:
        if not self.is_open:
            return
        try:
            try:
                self.stop()
            finally:
                self.card.close()
        finally:
            self.is_open = False

    def write(self, data: bytes) -> int:
        text = data.decode("utf-8", errors="ignore").strip()
        if not text:
            return 0
        self._parse_command(text)
        return len(data)

    @property
    def in_waiting(self) -> int:
        self._service()
        return len(self._rx_lines)

    def readline(self, size: int = 128) -> bytes:
        self._service()
        if not self._rx_lines:
            return b""
        return self._rx_lines.popleft()[:size]

    def read_encoder_snapshot(self) -> dict[str, float | int | bool]:
        """Return raw Qube instrumentation values for encoder/filtering labs."""
        if not self.is_open:
            raise QuanserConnectionError("Quanser card is closed")

        ai = np.zeros(1, dtype=np.float64)
        enc = np.zeros(2, dtype=np.int32)
        other = np.zeros(2, dtype=np.float64)
        try:
            self.card.read(
                self.ai_channels,
                1,
                self.encoder_channels,
                2,
                None,
                0,
                self.other_input_channels if self.other_available else None,
                2 if self.other_available else 0,
                ai,
                enc,
                None,
                other if self.other_available else None,
            )
        except Exception:
            if not self.other_available:
                raise
            self.other_available = False
            self.card.read(self.ai_channels, 1, self.encoder_channels, 2, None, 0, None, 0, ai, enc, None, None)

        theta = self.encoder_sign * float(enc[0]) * COUNT_TO_RAD
        tach0_rpm = float("nan")
        tach1_rpm = float("nan")
        if self.other_available:
            tach0_rpm = self.encoder_sign * float(other[0]) * COUNT_TO_RAD * 60.0 / (2.0 * pi)
            tach1_rpm = self.encoder_sign * float(other[1]) * COUNT_TO_RAD * 60.0 / (2.0 * pi)

        return {
            "encoder0_count": int(enc[0]),
            "encoder1_count": int(enc[1]),
            "theta_rad": theta,
            "theta_deg": theta * 180.0 / pi,
            "tach0_rpm": tach0_rpm,
            "tach1_rpm": tach1_rpm,
            "current_a": float(ai[0]),
            "enabled": bool(self._enabled_led),
            "counts_per_rev": int(round((2.0 * pi) / COUNT_TO_RAD)),
        }

    def _parse_command(self, text: str) -> None:
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 25:
            return

        def f(idx: int, default: float = 0.0) -> float:
            try:
                return float(parts[idx])
            except ValueError:
                return default

        def i(idx: int, default: int = 0) -> int:
            return int(round(f(idx, default)))

        self.cmd = CommandState(
            inicio=i(0),
            modooperacion=i(1),
            A=f(2),
            B=f(3),
            C=f(4),
            D=f(5),
            E=f(6),
            F=f(7),
            G=f(8),
            H=f(9),
            delay_ms=max(1, i(10, 10)),
            reference_period_ms=max(1, i(11, 2000)),
            amplitude=i(12),
            manual_reference=i(13),
            offset=i(14),
            signal_type=i(15),
            active_tab=i(16),
            Kp=f(17),
            Ki=f(18),
            Kd=f(19),
            deadzone=max(0, min(255, i(20))),
            derivative_time_constant=max(1e-6, f(21, 0.01)),
            pid_type=max(0, min(1, i(22))),
            automatic_reference=1 if i(23, 1) else 0,
            reset_time=max(1e-6, f(24, 0.1)),
        )

        if text != self._last_command_text:
            self._reset_control_memory()
            self._last_command_text = text
            self._last_service = time.monotonic()

    def _reset_control_memory(self) -> None:
        self._reset_controller_state()
        self.time_ms = 0.0
        self.prbs_is_zero = False
        self.prbs_last_switch = 0.0
        self.reference = 0
        self.measurement = 0
        self.rpm_avg: deque[float] = deque(maxlen=1)

    def _reset_controller_state(self) -> None:
        self.x = [0.0] * 5
        self.y = [0.0] * 5
        self.error = 0.0
        self.integral_sum = 0.0
        self.previous_error = 0.0
        self.previous_output = 0.0
        self.controller_output = 0.0
        self.pwm_out = 0
        self.d_y = 0.0
        self.d_out_prev = 0.0
        self.d_b0 = 0.0
        self.d_b1 = 0.0
        self.d_a1 = 0.0
        self._last_pid_coeff_key = None

    def _set_encoder_counts_zero(self) -> None:
        try:
            self.card.set_encoder_counts(self.encoder_channels, 2, np.array([0, 0], dtype=np.int32))
        except Exception:
            pass
        self._theta_zero = 0.0
        self._last_theta = 0.0

    def _service(self) -> None:
        if not self.is_open:
            return

        now = time.monotonic()
        delay_s = max(self.cmd.delay_ms, 1) / 1000.0
        produced = 0
        while now - self._last_service >= delay_s and produced < 5:
            self._last_service += delay_s
            dt_ms = delay_s * 1000.0
            produced += 1
            line = self._step(dt_ms)
            if line is not None:
                self._rx_lines.append(line.encode("utf-8"))

    def _read_qube(self, dt_ms: float) -> tuple[float, float, float, float]:
        ai = np.zeros(1, dtype=np.float64)
        enc = np.zeros(2, dtype=np.int32)
        other = np.zeros(2, dtype=np.float64)
        try:
            self.card.read(
                self.ai_channels,
                1,
                self.encoder_channels,
                2,
                None,
                0,
                self.other_input_channels if self.other_available else None,
                2 if self.other_available else 0,
                ai,
                enc,
                None,
                other if self.other_available else None,
            )
        except Exception:
            if not self.other_available:
                raise
            self.other_available = False
            self.card.read(self.ai_channels, 1, self.encoder_channels, 2, None, 0, None, 0, ai, enc, None, None)

        theta = self.encoder_sign * float(enc[0]) * COUNT_TO_RAD
        if self.other_available:
            theta_dot = self.encoder_sign * float(other[0]) * COUNT_TO_RAD
        else:
            theta_dot = (theta - self._last_theta) / max(dt_ms * 1e-3, 1e-6)
        self._last_theta = theta
        rpm = theta_dot * 60.0 / (2.0 * pi)
        degrees = (theta - self._theta_zero) * 180.0 / pi
        return rpm, degrees, float(ai[0]), theta

    def _write_voltage(self, voltage: float, enabled: bool | None = None) -> None:
        voltage = _clip(voltage, -self.max_voltage, self.max_voltage)
        if enabled is not None:
            self._enabled_led = bool(enabled)
        analog = np.array([self.output_sign * voltage], dtype=np.float64)
        digital = np.array([1 if self._enabled_led else 0], dtype=np.int8)
        leds = self.led_green if self._enabled_led else self.led_red
        self.card.write(
            self.ao_channels,
            1,
            None,
            0,
            self.digital_channels,
            len(self.digital_channels),
            self.led_channels,
            len(self.led_channels),
            analog,
            None,
            digital,
            leds,
        )

    def _pwm_to_voltage(self, pwm: float) -> float:
        return _clip(pwm, -255.0, 255.0) / 255.0 * self.max_voltage

    def _reference_generator(self) -> int:
        c = self.cmd
        T = max(float(c.reference_period_ms), 1.0)
        t = self.time_ms
        if c.signal_type == 0:
            half = T / 2.0
            if (t - self.prbs_last_switch) >= half:
                self.prbs_last_switch = t
                self.prbs_is_zero = not self.prbs_is_zero
                if not self.prbs_is_zero:
                    self.reference = random.randint(40, 240)
            return 0 if self.prbs_is_zero else int(self.reference)
        if c.signal_type == 1:
            return int(c.amplitude + c.offset if fmod(t, T) < T / 2.0 else c.offset)
        if c.signal_type == 2:
            return int(c.amplitude * sin(2.0 * pi * t / T) + c.offset)
        if c.signal_type == 3:
            section = T / 3.0
            tc = fmod(t, T)
            if tc < section:
                return int((c.amplitude / section) * tc)
            if tc < 2.0 * section:
                return int(c.amplitude - (c.amplitude / section) * (tc - section))
            return 0
        if c.signal_type == 4:
            return int(c.amplitude + c.offset if t < T else c.offset)
        if c.signal_type == 5:
            return int(c.amplitude * sin(2.0 * pi * (0.001 * t) * t * 0.0001) + c.offset)
        if c.signal_type == 6:
            return int(c.amplitude * exp(-0.0005 * t) + c.offset)
        if c.signal_type == 7:
            return int(random.random() * c.amplitude + c.offset)
        return 0

    def _update_reference(self) -> int:
        self.reference = self._reference_generator() if self.cmd.automatic_reference else self.cmd.manual_reference
        return int(self.reference)

    def _update_pid_coefficients(self, Ts: float) -> None:
        key = (round(Ts, 9), self.cmd.Kd, self.cmd.derivative_time_constant)
        if key == self._last_pid_coeff_key:
            return
        Tc = max(self.cmd.derivative_time_constant, 1e-6)
        den = 2.0 * Tc + Ts
        self.d_b0 = 2.0 * self.cmd.Kd / den
        self.d_b1 = -2.0 * self.cmd.Kd / den
        self.d_a1 = (Ts - 2.0 * Tc) / den
        self._last_pid_coeff_key = key

    def _pid_positional(self, Ts: float, lo: float, hi: float) -> int:
        self._update_pid_coefficients(Ts)
        e = self.error
        em = self.previous_error
        p_term = self.cmd.Kp * e
        self.d_y = -self.d_a1 * self.d_y + self.d_b0 * e + self.d_b1 * em
        d_term = self.d_y if self.cmd.Kd >= 0 else 0.0
        i_trial = self.integral_sum + 0.5 * self.cmd.Ki * Ts * (e + em)
        u_unsat = p_term + i_trial + d_term
        u_sat = _clip(u_unsat, lo, hi)
        self.integral_sum = i_trial + (Ts / max(self.cmd.reset_time, 1e-6)) * (u_sat - u_unsat)
        self.controller_output = u_sat
        self.previous_error = e
        return int(round(u_sat))

    def _pid_incremental(self, Ts: float, lo: float, hi: float) -> int:
        self._update_pid_coefficients(Ts)
        du_p = self.cmd.Kp * (self.error - self.previous_error)
        du_i = 0.5 * self.cmd.Ki * Ts * (self.error + self.previous_error)
        d_out = -self.d_a1 * self.d_out_prev + self.d_b0 * self.error + self.d_b1 * self.previous_error
        du_d = d_out - self.d_out_prev
        self.d_out_prev = d_out
        u_pre = self.previous_output + du_p + du_i + du_d
        u_sat_pre = _clip(u_pre, lo, hi)
        du_aw = (u_sat_pre - u_pre) * (Ts / max(self.cmd.reset_time, 1e-6))
        u_sat = _clip(u_pre + du_aw, lo, hi)
        self.controller_output = u_sat
        self.previous_output = u_sat
        self.previous_error = self.error
        return int(round(u_sat))

    def _difference_equation(self, lo: float, hi: float) -> int:
        self.x[3], self.x[2], self.x[1] = self.x[2], self.x[1], self.x[0]
        self.x[0] = self.error
        self.y[4], self.y[3], self.y[2], self.y[1] = self.y[3], self.y[2], self.y[1], self.controller_output
        u = (
            self.cmd.A * self.x[0]
            + self.cmd.B * self.x[1]
            + self.cmd.C * self.x[2]
            + self.cmd.D * self.x[3]
            + self.cmd.E * self.y[1]
            + self.cmd.F * self.y[2]
            + self.cmd.G * self.y[3]
            + self.cmd.H * self.y[4]
        )
        self.controller_output = _clip(u, lo, hi)
        return int(round(self.controller_output))

    def _step(self, dt_ms: float) -> str | None:
        c = self.cmd
        self.time_ms += dt_ms
        rpm, degrees, current, _theta = self._read_qube(dt_ms)
        self.rpm_avg.append(rpm)
        rpm_avg = float(sum(self.rpm_avg) / len(self.rpm_avg))
        Ts = max(dt_ms * 1e-3, 1e-6)

        if c.inicio != 1 or c.modooperacion == 0:
            self._reset_control_memory()
            self._write_voltage(0.0, enabled=False)
            return None

        if c.modooperacion == 1:
            ref = self._update_reference()
            self.controller_output = _clip(ref, 0.0, 255.0)
            self.pwm_out = int(round(self.controller_output))
            self._write_voltage(self._pwm_to_voltage(self.pwm_out), enabled=True)
            meas = int(round(rpm_avg))
            return f"{int(ref)} {meas} {int(round(dt_ms))} {current:.2f} {self.pwm_out}\n"

        if c.modooperacion == 2:
            ref = self._update_reference()
            meas = rpm_avg
            self.error = float(ref) - meas
            if c.active_tab == 8:
                raw = self._difference_equation(0.0, 255.0)
            else:
                raw = self._pid_incremental(Ts, 0.0, 255.0) if c.pid_type == 0 else self._pid_positional(Ts, 0.0, 255.0)
            self.pwm_out = int(_clip(raw + c.deadzone, 0, 255)) if ref != 0 else 0
            if ref == 0:
                self._reset_controller_state()
            self._write_voltage(self._pwm_to_voltage(self.pwm_out), enabled=True)
            return f"{int(ref)} {int(round(meas))} {int(round(dt_ms))} {current:.2f} {self.pwm_out}\n"

        if c.modooperacion == 3:
            ref = self._update_reference()
            meas = degrees
            self.error = float(ref) - meas
            if c.active_tab == 8:
                u = self._difference_equation(-255.0, 255.0)
            else:
                u = self._pid_incremental(Ts, -255.0, 255.0) if c.pid_type == 0 else self._pid_positional(Ts, -255.0, 255.0)
            u = _clip(u, -255, 255)
            self.pwm_out = int(round(abs(u)))
            self._write_voltage(self._pwm_to_voltage(u), enabled=True)
            return f"{int(ref)} {int(round(meas))} {int(round(dt_ms))} {current:.2f} {self.pwm_out}\n"

        self._write_voltage(0.0, enabled=False)
        return None
