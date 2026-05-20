from __future__ import annotations

import ast
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import Qt

from quanser_backend import QuanserConnectionError, QuanserSerialEmulator

try:
    from scipy.optimize import least_squares
except Exception:
    least_squares = None


QUADRATURE_COUNTS_PER_REV = 512 * 4


MODE_OPTIONS = [
    ("Disabled", 0),
    ("Open-loop motor command", 1),
    ("Speed control", 2),
    ("Position control", 3),
]

SIGNAL_OPTIONS = [
    ("PRBS", 0),
    ("Square", 1),
    ("Sine", 2),
    ("Triangular", 3),
    ("Pulse", 4),
    ("Chirp", 5),
    ("Exponential decay", 6),
    ("White noise", 7),
]


@dataclass(frozen=True)
class LabPreset:
    title: str
    goal: str
    mode: int
    signal: int
    amplitude: int
    period_ms: int
    offset: int = 0
    manual_reference: int = 0
    automatic_reference: bool = True
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    pid_type: int = 1
    active_tab: int = 5
    coefficients: tuple[float, float, float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0, 0, 0)
    deadzone: int = 0
    delay_ms: int = 20
    duration_s: float = 12.0
    analysis: str = "summary"


LABS = [
    LabPreset(
        "0.1 Interfacing",
        "Use the encoder panel to measure raw counts, select a counting convention, determine counts per revolution, convert counts to angle, and infer motion direction.",
        mode=1,
        signal=1,
        amplitude=90,
        period_ms=2500,
        analysis="interfacing",
    ),
    LabPreset(
        "0.2 Filtering",
        "Use the filtering panel to compare encoder-differentiated speed, filtered speed, motor command, and tachometer speed while varying filter parameters.",
        mode=1,
        signal=1,
        amplitude=100,
        period_ms=2500,
        analysis="filtering",
    ),
    LabPreset(
        "1.1 Step Response Modeling",
        "Capture an open-loop step response and estimate steady-state gain, time constant, rise time, and settling time.",
        mode=1,
        signal=4,
        amplitude=120,
        period_ms=4000,
        analysis="step",
    ),
    LabPreset(
        "1.2 Frequency Response Modeling",
        "Excite the motor with a sinusoidal command and estimate the dominant gain and phase shift from measured data.",
        mode=1,
        signal=2,
        amplitude=70,
        period_ms=2000,
        offset=100,
        duration_s=16.0,
        analysis="frequency",
    ),
    LabPreset(
        "1.3 Parameter Estimation",
        "Estimate a first-order motor model from measured input-output data using the recorded Quanser response.",
        mode=1,
        signal=1,
        amplitude=115,
        period_ms=3000,
        duration_s=15.0,
        analysis="estimate",
    ),
    LabPreset(
        "1.4 Block / State-Space Modeling",
        "Convert the identified motor model to transfer-function and state-space forms for later controller design.",
        mode=1,
        signal=4,
        amplitude=120,
        period_ms=4000,
        analysis="model",
    ),
    LabPreset(
        "2.1 Stability and Routh-Hurwitz",
        "Use the identified model to check closed-loop stability conditions for proportional control.",
        mode=2,
        signal=1,
        amplitude=110,
        period_ms=3000,
        kp=0.45,
        analysis="stability",
    ),
    LabPreset(
        "2.2 Root Locus Design",
        "Sweep proportional gain on the identified rotary model and visualize closed-loop pole migration.",
        mode=2,
        signal=1,
        amplitude=110,
        period_ms=3000,
        kp=0.45,
        analysis="root_locus",
    ),
    LabPreset(
        "3.1 Proportional Speed Control",
        "Close the speed loop with proportional control and measure transient response and steady-state error.",
        mode=2,
        signal=1,
        amplitude=120,
        period_ms=3000,
        kp=0.55,
        duration_s=14.0,
        analysis="control",
    ),
    LabPreset(
        "3.2 Proportional Position Control",
        "Close the position loop with proportional control and evaluate tracking, overshoot, and oscillation.",
        mode=3,
        signal=1,
        amplitude=75,
        period_ms=3500,
        kp=4.0,
        duration_s=14.0,
        analysis="control",
    ),
    LabPreset(
        "3.3 PD / PID Position Control",
        "Tune position PID gains and compare rise time, overshoot, settling behavior, and command effort.",
        mode=3,
        signal=1,
        amplitude=75,
        period_ms=3500,
        kp=4.0,
        ki=0.0,
        kd=0.18,
        duration_s=14.0,
        analysis="control",
    ),
    LabPreset(
        "3.4 Steady-State Error",
        "Measure the remaining tracking error under proportional control and compare it with controller gain.",
        mode=2,
        signal=1,
        amplitude=120,
        period_ms=3000,
        kp=0.35,
        duration_s=14.0,
        analysis="sse",
    ),
    LabPreset(
        "3.5 Lead / Discrete Controller",
        "Deploy a discrete difference-equation controller and compare its response with PID control.",
        mode=2,
        signal=1,
        amplitude=120,
        period_ms=3000,
        active_tab=8,
        coefficients=(0.45, -0.20, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0),
        duration_s=14.0,
        analysis="discrete",
    ),
]


class MotorLabsWindow(QtWidgets.QMainWindow):
    def __init__(self, backend_factory=QuanserSerialEmulator):
        super().__init__()
        self.backend_factory = backend_factory
        self.qube = None
        self.running = False
        self.elapsed_s = 0.0
        self.last_model = None
        self.last_analysis_text = ""
        self.data = {key: [] for key in ("t", "ref", "meas", "dt_ms", "current", "pwm")}
        self.instrument = {key: [] for key in ("t", "raw_count", "display_count", "position_deg", "encoder_rpm", "tach_rpm", "filtered_rpm")}
        self.monitor_elapsed_s = 0.0
        self.encoder_zero_raw = 0
        self.encoder_last_raw = None
        self.encoder_last_time = None
        self.marked_counts_per_rev = None

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(20)
        self.timer.timeout.connect(self._poll_hardware)

        self.setWindowTitle("Quanser Rotary Motor Labs")
        self.resize(1160, 700)
        self.setMinimumSize(900, 560)
        self._build_ui()
        self._apply_style()
        self._connect_signals()
        self.lab_list.setCurrentRow(0)
        self._select_lab(0)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)
        self.setCentralWidget(central)

        left = QtWidgets.QWidget()
        left.setMinimumWidth(300)
        left.setMaximumWidth(370)
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        title = QtWidgets.QLabel("Rotary Motor Labs")
        title.setObjectName("Title")
        left_layout.addWidget(title)

        self.lab_list = QtWidgets.QListWidget()
        for lab in LABS:
            self.lab_list.addItem(lab.title)
        self.lab_list.setMinimumHeight(120)
        self.lab_list.setMaximumHeight(170)
        left_layout.addWidget(self.lab_list)

        hardware_box = QtWidgets.QGroupBox("Hardware")
        hardware_layout = QtWidgets.QGridLayout(hardware_box)
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.release_button = QtWidgets.QPushButton("Release")
        self.release_button.setEnabled(False)
        self.start_button = QtWidgets.QPushButton("Start")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)
        hardware_layout.addWidget(self.connect_button, 0, 0)
        hardware_layout.addWidget(self.release_button, 0, 1)
        hardware_layout.addWidget(self.start_button, 1, 0)
        hardware_layout.addWidget(self.stop_button, 1, 1)
        left_layout.addWidget(hardware_box)

        actions = QtWidgets.QHBoxLayout()
        self.analyze_button = QtWidgets.QPushButton("Analyze")
        self.export_csv_button = QtWidgets.QPushButton("Export CSV")
        self.export_report_button = QtWidgets.QPushButton("Export Report")
        actions.addWidget(self.analyze_button)
        actions.addWidget(self.export_csv_button)
        actions.addWidget(self.export_report_button)
        left_layout.addLayout(actions)

        self.settings_tabs = QtWidgets.QTabWidget()
        self.settings_tabs.setDocumentMode(True)
        self.settings_tabs.setMinimumHeight(260)
        left_layout.addWidget(self.settings_tabs, 1)

        preset_box = QtWidgets.QGroupBox("Experiment")
        preset_layout = QtWidgets.QGridLayout(preset_box)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems([label for label, _value in MODE_OPTIONS])
        self.signal_combo = QtWidgets.QComboBox()
        self.signal_combo.addItems([label for label, _value in SIGNAL_OPTIONS])
        self.auto_reference = QtWidgets.QCheckBox("Automatic reference")
        self.auto_reference.setChecked(True)
        self.duration = self._spin(1, 120, 12, suffix=" s")
        self.delay = self._spin(1, 100, 20, suffix=" ms")
        self.period = self._spin(100, 30000, 3000, suffix=" ms")
        self.amplitude = self._spin(-255, 255, 120)
        self.offset = self._spin(-255, 255, 0)
        self.manual_reference = self._spin(-255, 255, 0)
        self.deadzone = self._spin(0, 255, 0)
        rows = [
            ("Mode", self.mode_combo),
            ("Signal", self.signal_combo),
            ("Duration", self.duration),
            ("Sample time", self.delay),
            ("Period", self.period),
            ("Amplitude", self.amplitude),
            ("Offset", self.offset),
            ("Manual ref", self.manual_reference),
            ("Dead zone", self.deadzone),
        ]
        for row, (label, widget) in enumerate(rows):
            preset_layout.addWidget(QtWidgets.QLabel(label), row, 0)
            preset_layout.addWidget(widget, row, 1)
        preset_layout.addWidget(self.auto_reference, len(rows), 0, 1, 2)
        self.settings_tabs.addTab(self._scroll_page(preset_box), "Experiment")

        controller_box = QtWidgets.QGroupBox("Controller")
        controller_layout = QtWidgets.QGridLayout(controller_box)
        self.kp = self._double_spin(-1000, 1000, 0, decimals=5, step=0.05)
        self.ki = self._double_spin(-1000, 1000, 0, decimals=5, step=0.05)
        self.kd = self._double_spin(-1000, 1000, 0, decimals=5, step=0.01)
        self.derivative_filter = self._double_spin(0.000001, 10, 0.2, decimals=5, step=0.01)
        self.reset_time = self._double_spin(0.000001, 10, 0.5, decimals=5, step=0.01)
        self.pid_form = QtWidgets.QComboBox()
        self.pid_form.addItems(["Incremental", "Positional"])
        for row, (label, widget) in enumerate(
            [
                ("Kp", self.kp),
                ("Ki", self.ki),
                ("Kd", self.kd),
                ("D filter", self.derivative_filter),
                ("Reset time", self.reset_time),
                ("PID form", self.pid_form),
            ]
        ):
            controller_layout.addWidget(QtWidgets.QLabel(label), row, 0)
            controller_layout.addWidget(widget, row, 1)
        self.settings_tabs.addTab(self._scroll_page(controller_box), "Controller")

        coeff_box = QtWidgets.QGroupBox("Discrete A-H")
        coeff_layout = QtWidgets.QGridLayout(coeff_box)
        self.coeff_spins = []
        for idx, name in enumerate("ABCDEFGH"):
            spin = self._double_spin(-1000, 1000, 0, decimals=6, step=0.01)
            self.coeff_spins.append(spin)
            coeff_layout.addWidget(QtWidgets.QLabel(name), idx // 4, (idx % 4) * 2)
            coeff_layout.addWidget(spin, idx // 4, (idx % 4) * 2 + 1)
        self.settings_tabs.addTab(self._scroll_page(coeff_box), "Discrete")

        self.instrument_tabs = QtWidgets.QTabWidget()
        self.instrument_tabs.addTab(self._build_encoder_tab(), "Encoder")
        self.instrument_tabs.addTab(self._build_filter_tab(), "Filtering")
        self.settings_tabs.addTab(self._scroll_page(self.instrument_tabs), "Instrument")

        self.goal_text = QtWidgets.QTextEdit()
        self.goal_text.setReadOnly(True)
        self.settings_tabs.addTab(self._scroll_page(self.goal_text), "Goal")

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        self.status_label = QtWidgets.QLabel("Hardware released")
        self.status_label.setObjectName("Status")
        right_layout.addWidget(self.status_label)

        plot_splitter = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        plot_splitter.setChildrenCollapsible(False)
        right_layout.addWidget(plot_splitter, 1)

        self.response_plot = pg.PlotWidget()
        self.response_plot.addLegend()
        self.response_plot.setLabel("bottom", "Time", units="s")
        self.response_plot.setLabel("left", "Response")
        self.response_plot.showGrid(x=True, y=True, alpha=0.25)
        self.response_plot.setMinimumHeight(150)
        self.reference_curve = self.response_plot.plot(name="Reference", pen=pg.mkPen("#1d4ed8", width=2))
        self.measured_curve = self.response_plot.plot(name="Measured", pen=pg.mkPen("#dc2626", width=2))
        plot_splitter.addWidget(self.response_plot)

        self.command_plot = pg.PlotWidget()
        self.command_plot.addLegend()
        self.command_plot.setLabel("bottom", "Time", units="s")
        self.command_plot.setLabel("left", "Command", units="PWM")
        self.command_plot.showGrid(x=True, y=True, alpha=0.25)
        self.command_plot.setMinimumHeight(90)
        self.pwm_curve = self.command_plot.plot(name="PWM", pen=pg.mkPen("#15803d", width=2))
        self.current_curve = self.command_plot.plot(name="Current x 50", pen=pg.mkPen("#7c3aed", width=1))
        plot_splitter.addWidget(self.command_plot)

        self.analysis_plot = pg.PlotWidget()
        self.analysis_plot.addLegend()
        self.analysis_plot.setLabel("bottom", "Time", units="s")
        self.analysis_plot.setLabel("left", "Analysis")
        self.analysis_plot.showGrid(x=True, y=True, alpha=0.25)
        self.analysis_plot.setMinimumHeight(120)
        plot_splitter.addWidget(self.analysis_plot)

        self.results = QtWidgets.QTextEdit()
        self.results.setReadOnly(True)
        self.results.setMinimumHeight(90)
        plot_splitter.addWidget(self.results)
        plot_splitter.setSizes([330, 150, 180, 120])

        root.addWidget(left)
        root.addWidget(right, 1)

    def _build_encoder_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        controls = QtWidgets.QGridLayout()
        self.encoder_edge_mode = QtWidgets.QComboBox()
        self.encoder_edge_mode.addItems([
            "Quadrature x4 (HIL raw)",
            "Channel change x2",
            "Rising edge x1",
            "Falling edge x1",
            "Custom counts/rev",
        ])
        self.encoder_direction = QtWidgets.QComboBox()
        self.encoder_direction.addItems(["Hardware sign", "Invert sign"])
        self.encoder_expected_cpr = self._spin(1, 100000, QUADRATURE_COUNTS_PER_REV)
        self.encoder_custom_cpr = self._spin(1, 100000, QUADRATURE_COUNTS_PER_REV)
        self.encoder_formula = QtWidgets.QLineEdit("counts * 360 / counts_per_rev")
        self.encoder_formula.setToolTip("Safe expression. Variables: counts, raw_counts, counts_per_rev, pi, rpm.")
        rows = [
            ("Count mode", self.encoder_edge_mode),
            ("Direction", self.encoder_direction),
            ("Expected CPR", self.encoder_expected_cpr),
            ("Custom CPR", self.encoder_custom_cpr),
            ("Angle formula", self.encoder_formula),
        ]
        for row, (label, widget) in enumerate(rows):
            controls.addWidget(QtWidgets.QLabel(label), row, 0)
            controls.addWidget(widget, row, 1)
        layout.addLayout(controls)

        buttons = QtWidgets.QHBoxLayout()
        self.encoder_zero_button = QtWidgets.QPushButton("Zero")
        self.encoder_mark_button = QtWidgets.QPushButton("Mark 1 rev")
        self.encoder_eval_button = QtWidgets.QPushButton("Eval formula")
        buttons.addWidget(self.encoder_zero_button)
        buttons.addWidget(self.encoder_mark_button)
        buttons.addWidget(self.encoder_eval_button)
        layout.addLayout(buttons)

        readouts = QtWidgets.QGridLayout()
        self.encoder_readouts = {}
        for row, key in enumerate([
            "raw_count",
            "display_count",
            "angle_deg",
            "angle_rad",
            "encoder_rpm",
            "tach_rpm",
            "measured_cpr",
            "direction",
            "student_angle",
        ]):
            label = QtWidgets.QLabel(key.replace("_", " ").title())
            value = self._readout_label()
            self.encoder_readouts[key] = value
            readouts.addWidget(label, row, 0)
            readouts.addWidget(value, row, 1)
        layout.addLayout(readouts)
        return self._scroll_page(tab)

    def _build_filter_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        controls = QtWidgets.QGridLayout()
        self.filter_source = QtWidgets.QComboBox()
        self.filter_source.addItems(["Encoder diff speed", "Tachometer speed"])
        self.filter_type = QtWidgets.QComboBox()
        self.filter_type.addItems(["Moving average", "Exponential IIR", "First-order low-pass", "No filter"])
        self.filter_window = self._spin(1, 201, 9)
        self.filter_alpha = self._double_spin(0.001, 1.0, 0.2, decimals=4, step=0.01)
        self.filter_cutoff = self._double_spin(0.01, 1000.0, 50.0, decimals=3, step=1.0)
        rows = [
            ("Source", self.filter_source),
            ("Filter", self.filter_type),
            ("MA window", self.filter_window),
            ("IIR alpha", self.filter_alpha),
            ("Cutoff rad/s", self.filter_cutoff),
        ]
        for row, (label, widget) in enumerate(rows):
            controls.addWidget(QtWidgets.QLabel(label), row, 0)
            controls.addWidget(widget, row, 1)
        layout.addLayout(controls)

        self.cutoff_hz_label = self._readout_label()
        self.raw_noise_label = self._readout_label()
        self.filtered_noise_label = self._readout_label()
        self.noise_reduction_label = self._readout_label()
        readouts = QtWidgets.QGridLayout()
        for row, (label, widget) in enumerate([
            ("Cutoff Hz", self.cutoff_hz_label),
            ("Raw std", self.raw_noise_label),
            ("Filtered std", self.filtered_noise_label),
            ("Reduction", self.noise_reduction_label),
        ]):
            readouts.addWidget(QtWidgets.QLabel(label), row, 0)
            readouts.addWidget(widget, row, 1)
        layout.addLayout(readouts)
        return self._scroll_page(tab)

    def _scroll_page(self, widget: QtWidgets.QWidget) -> QtWidgets.QScrollArea:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(widget)
        return scroll

    def _readout_label(self) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel("--")
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #f8fafc;
                color: #0f172a;
                font-size: 12px;
            }
            QLabel#Title {
                font-size: 22px;
                font-weight: 700;
                padding-bottom: 4px;
            }
            QLabel#Status {
                background: #e2e8f0;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 7px 10px;
                font-weight: 600;
            }
            QGroupBox {
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                font-weight: 600;
            }
            QPushButton {
                border: 1px solid #94a3b8;
                border-radius: 5px;
                padding: 6px 9px;
                background: #ffffff;
            }
            QPushButton:hover {
                background: #eef2ff;
            }
            QPushButton:disabled {
                color: #94a3b8;
                background: #f1f5f9;
            }
            QListWidget, QTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 5px;
                padding: 3px;
            }
            """
        )

    def _connect_signals(self) -> None:
        self.lab_list.currentRowChanged.connect(self._select_lab)
        self.connect_button.clicked.connect(self.connect_hardware)
        self.release_button.clicked.connect(self.release_hardware)
        self.start_button.clicked.connect(self.start_lab)
        self.stop_button.clicked.connect(self.stop_lab)
        self.analyze_button.clicked.connect(self.analyze_current_lab)
        self.export_csv_button.clicked.connect(self.export_csv)
        self.export_report_button.clicked.connect(self.export_report)
        self.encoder_zero_button.clicked.connect(self.zero_encoder_counter)
        self.encoder_mark_button.clicked.connect(self.mark_one_revolution)
        self.encoder_eval_button.clicked.connect(self.evaluate_encoder_formula)
        self.encoder_edge_mode.currentIndexChanged.connect(self._on_encoder_count_mode_changed)
        self.encoder_direction.currentIndexChanged.connect(self._refresh_instrumentation_display)
        self.encoder_expected_cpr.valueChanged.connect(self._refresh_instrumentation_display)
        self.encoder_custom_cpr.valueChanged.connect(self._refresh_instrumentation_display)
        self.filter_source.currentIndexChanged.connect(self._refresh_filter_display)
        self.filter_type.currentIndexChanged.connect(self._refresh_filter_display)
        self.filter_window.valueChanged.connect(self._refresh_filter_display)
        self.filter_alpha.valueChanged.connect(self._refresh_filter_display)
        self.filter_cutoff.valueChanged.connect(self._refresh_filter_display)

    def _spin(self, minimum: int, maximum: int, value: int, suffix: str = "") -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSuffix(suffix)
        return spin

    def _double_spin(self, minimum: float, maximum: float, value: float, decimals: int, step: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        return spin

    def _select_lab(self, row: int) -> None:
        if row < 0:
            return
        lab = LABS[row]
        self.goal_text.setPlainText(lab.goal)
        self.mode_combo.setCurrentIndex(self._index_for_value(MODE_OPTIONS, lab.mode))
        self.signal_combo.setCurrentIndex(self._index_for_value(SIGNAL_OPTIONS, lab.signal))
        self.duration.setValue(int(round(lab.duration_s)))
        self.delay.setValue(lab.delay_ms)
        self.period.setValue(lab.period_ms)
        self.amplitude.setValue(lab.amplitude)
        self.offset.setValue(lab.offset)
        self.manual_reference.setValue(lab.manual_reference)
        self.auto_reference.setChecked(lab.automatic_reference)
        self.kp.setValue(lab.kp)
        self.ki.setValue(lab.ki)
        self.kd.setValue(lab.kd)
        self.pid_form.setCurrentIndex(lab.pid_type)
        self.deadzone.setValue(lab.deadzone)
        for spin, value in zip(self.coeff_spins, lab.coefficients):
            spin.setValue(value)
        if lab.analysis == "filtering":
            self.settings_tabs.setCurrentIndex(3)
            self.instrument_tabs.setCurrentIndex(1)
        elif lab.title.startswith("0.1"):
            self.settings_tabs.setCurrentIndex(3)
            self.instrument_tabs.setCurrentIndex(0)
        else:
            self.settings_tabs.setCurrentIndex(0)
        self.results.setPlainText("Preset loaded. Connect the Qube and press Start when ready.")
        self._update_axis_labels()

    def _index_for_value(self, options, value: int) -> int:
        for idx, (_label, item_value) in enumerate(options):
            if item_value == value:
                return idx
        return 0

    def current_lab(self) -> LabPreset:
        row = max(self.lab_list.currentRow(), 0)
        return LABS[row]

    def current_mode(self) -> int:
        return MODE_OPTIONS[self.mode_combo.currentIndex()][1]

    def current_signal(self) -> int:
        return SIGNAL_OPTIONS[self.signal_combo.currentIndex()][1]

    def connect_hardware(self) -> None:
        if self.qube is not None and getattr(self.qube, "is_open", False):
            self.status_label.setText("Quanser Qube already connected")
            return
        try:
            self.qube = self.backend_factory()
            self.qube.reset_input_buffer()
        except QuanserConnectionError as exc:
            self.qube = None
            self.status_label.setText(f"Quanser backend unavailable: {exc}")
            return
        except Exception as exc:
            self.qube = None
            self.status_label.setText(f"Could not open Qube: {exc}")
            return
        self.connect_button.setEnabled(False)
        self.release_button.setEnabled(True)
        self.timer.setInterval(max(5, self.delay.value()))
        self.timer.start()
        self.status_label.setText("Quanser Qube connected; motor output is disabled")

    def release_hardware(self) -> None:
        self.stop_lab()
        self.timer.stop()
        if self.qube is not None:
            try:
                self.qube.close()
            except Exception as exc:
                self.status_label.setText(f"Qube close reported an error: {exc}")
            finally:
                self.qube = None
        self.connect_button.setEnabled(True)
        self.release_button.setEnabled(False)
        self.status_label.setText("Hardware released")

    def start_lab(self) -> None:
        if self.qube is None or not getattr(self.qube, "is_open", False):
            self.connect_hardware()
        if self.qube is None:
            return
        self.clear_data()
        self.elapsed_s = 0.0
        self.running = True
        self.stop_button.setEnabled(True)
        self.start_button.setEnabled(False)
        self._update_axis_labels()
        self._send_command(start=True)
        self.timer.setInterval(max(5, self.delay.value()))
        self.timer.start()
        self.status_label.setText("Running lab; Qube output is enabled")

    def stop_lab(self) -> None:
        self.running = False
        if self.qube is not None and getattr(self.qube, "is_open", False):
            try:
                self._send_command(start=False)
                if hasattr(self.qube, "stop"):
                    self.qube.stop()
            except Exception as exc:
                self.status_label.setText(f"Stop command reported an error: {exc}")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self.qube is not None:
            if not self.timer.isActive():
                self.timer.setInterval(max(5, self.delay.value()))
                self.timer.start()
            self.status_label.setText("Stopped; motor output is disabled")

    def clear_data(self) -> None:
        for values in self.data.values():
            values.clear()
        for values in self.instrument.values():
            values.clear()
        self.monitor_elapsed_s = 0.0
        self.encoder_last_raw = None
        self.encoder_last_time = None
        self.last_analysis_text = ""
        self.reference_curve.setData([], [])
        self.measured_curve.setData([], [])
        self.pwm_curve.setData([], [])
        self.current_curve.setData([], [])
        self.analysis_plot.clear()
        self.results.clear()

    def _command_values(self, start: bool) -> list[float | int]:
        lab = self.current_lab()
        coeffs = [spin.value() for spin in self.coeff_spins]
        active_tab = 8 if lab.active_tab == 8 else 5
        return [
            1 if start else 0,
            self.current_mode(),
            *coeffs,
            self.delay.value(),
            self.period.value(),
            self.amplitude.value(),
            self.manual_reference.value(),
            self.offset.value(),
            self.current_signal(),
            active_tab,
            self.kp.value(),
            self.ki.value(),
            self.kd.value(),
            self.deadzone.value(),
            self.derivative_filter.value(),
            self.pid_form.currentIndex(),
            1 if self.auto_reference.isChecked() else 0,
            self.reset_time.value(),
        ]

    def _send_command(self, start: bool) -> None:
        if self.qube is None or not getattr(self.qube, "is_open", False):
            return
        data = ",".join(f"{value:.9g}" if isinstance(value, float) else str(value) for value in self._command_values(start))
        self.qube.write((data + "\n").encode("utf-8"))

    def _poll_hardware(self) -> None:
        if self.qube is None or not getattr(self.qube, "is_open", False):
            self.status_label.setText("Qube is not available")
            self.timer.stop()
            self.stop_lab()
            return

        try:
            self._send_command(start=self.running)
            while self.qube.in_waiting:
                line = self.qube.readline().decode("utf-8", errors="ignore").strip()
                self._append_line(line)
            if hasattr(self.qube, "read_encoder_snapshot"):
                self._append_encoder_snapshot(self.qube.read_encoder_snapshot())
        except Exception as exc:
            self.status_label.setText(f"Qube communication error: {exc}")
            self.release_hardware()
            return

        self._update_plots()
        self._refresh_instrumentation_display()
        if self.running and self.elapsed_s >= self.duration.value():
            self.stop_lab()
            self.analyze_current_lab()

    def _append_line(self, line: str) -> None:
        if not line:
            return
        parts = line.split()
        if len(parts) < 5:
            return
        try:
            ref, meas, dt_ms, current, pwm = map(float, parts[:5])
        except ValueError:
            return
        self.elapsed_s += max(dt_ms, 0.0) * 1e-3
        self.data["t"].append(self.elapsed_s)
        self.data["ref"].append(ref)
        self.data["meas"].append(meas)
        self.data["dt_ms"].append(dt_ms)
        self.data["current"].append(current)
        self.data["pwm"].append(pwm)

    def _append_encoder_snapshot(self, snapshot: dict[str, float | int | bool]) -> None:
        raw_count = int(snapshot.get("encoder0_count", 0))
        interval_s = max(self.timer.interval(), 1) * 1e-3
        if self.encoder_last_raw is None:
            encoder_rpm = 0.0
            self.encoder_last_time = self.monitor_elapsed_s
        else:
            dt = max(self.monitor_elapsed_s - float(self.encoder_last_time), interval_s, 1e-6)
            encoder_rpm = (raw_count - int(self.encoder_last_raw)) / QUADRATURE_COUNTS_PER_REV * 60.0 / dt
        self.monitor_elapsed_s += interval_s
        self.encoder_last_raw = raw_count
        self.encoder_last_time = self.monitor_elapsed_s

        display_count = self._display_count_from_raw(raw_count)
        position_deg = display_count * 360.0 / max(float(self.encoder_expected_cpr.value()), 1.0)
        tach_rpm = float(snapshot.get("tach0_rpm", float("nan")))

        self.instrument["t"].append(self.monitor_elapsed_s)
        self.instrument["raw_count"].append(raw_count)
        self.instrument["display_count"].append(display_count)
        self.instrument["position_deg"].append(position_deg)
        self.instrument["encoder_rpm"].append(self._direction_sign() * encoder_rpm)
        self.instrument["tach_rpm"].append(tach_rpm)
        self.instrument["filtered_rpm"].append(float("nan"))

        max_samples = 6000
        if len(self.instrument["t"]) > max_samples:
            for values in self.instrument.values():
                del values[: len(values) - max_samples]

    def _direction_sign(self) -> float:
        return -1.0 if self.encoder_direction.currentIndex() == 1 else 1.0

    def _display_scale(self) -> float:
        mode = self.encoder_edge_mode.currentIndex()
        if mode == 0:
            return 1.0
        if mode == 1:
            return 0.5
        if mode in (2, 3):
            return 0.25
        return float(self.encoder_custom_cpr.value()) / float(QUADRATURE_COUNTS_PER_REV)

    def _display_count_from_raw(self, raw_count: int) -> float:
        return self._direction_sign() * (float(raw_count) - float(self.encoder_zero_raw)) * self._display_scale()

    def _on_encoder_count_mode_changed(self) -> None:
        defaults = {
            0: QUADRATURE_COUNTS_PER_REV,
            1: QUADRATURE_COUNTS_PER_REV // 2,
            2: QUADRATURE_COUNTS_PER_REV // 4,
            3: QUADRATURE_COUNTS_PER_REV // 4,
            4: self.encoder_custom_cpr.value(),
        }
        self.encoder_expected_cpr.setValue(defaults.get(self.encoder_edge_mode.currentIndex(), QUADRATURE_COUNTS_PER_REV))
        self._refresh_instrumentation_display()

    def zero_encoder_counter(self) -> None:
        if self.instrument["raw_count"]:
            self.encoder_zero_raw = int(self.instrument["raw_count"][-1])
        else:
            self.encoder_zero_raw = 0
        self.marked_counts_per_rev = None
        self._refresh_instrumentation_display()

    def mark_one_revolution(self) -> None:
        if not self.instrument["raw_count"]:
            self.results.setPlainText("Connect the Qube first, zero the encoder, rotate one revolution, then press Mark 1 rev.")
            return
        self.marked_counts_per_rev = abs(self._display_count_from_raw(int(self.instrument["raw_count"][-1])))
        self._refresh_instrumentation_display()

    def evaluate_encoder_formula(self) -> None:
        try:
            result = self._student_angle_value()
            self.encoder_readouts["student_angle"].setText(f"{result:.6g}")
        except Exception as exc:
            self.encoder_readouts["student_angle"].setText(f"error: {exc}")

    def _student_angle_value(self) -> float:
        raw_count = int(self.instrument["raw_count"][-1]) if self.instrument["raw_count"] else 0
        counts = self._display_count_from_raw(raw_count)
        rpm = float(self.instrument["encoder_rpm"][-1]) if self.instrument["encoder_rpm"] else 0.0
        variables = {
            "counts": counts,
            "raw_counts": float(raw_count - self.encoder_zero_raw),
            "counts_per_rev": float(self.encoder_expected_cpr.value()),
            "pi": math.pi,
            "rpm": rpm,
        }
        return float(self._safe_eval_expression(self.encoder_formula.text(), variables))

    def _safe_eval_expression(self, expression: str, variables: dict[str, float]) -> float:
        tree = ast.parse(expression, mode="eval")

        def eval_node(node):
            if isinstance(node, ast.Expression):
                return eval_node(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.Name):
                if node.id not in variables:
                    raise ValueError(f"unknown name '{node.id}'")
                return float(variables[node.id])
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = eval_node(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)):
                left = eval_node(node.left)
                right = eval_node(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.Pow):
                    return left ** right
                return left % right
            raise ValueError("only arithmetic expressions are allowed")

        return eval_node(tree)

    def _instrument_arrays(self):
        return (
            np.asarray(self.instrument["t"], dtype=float),
            np.asarray(self.instrument["encoder_rpm"], dtype=float),
            np.asarray(self.instrument["tach_rpm"], dtype=float),
        )

    def _selected_filter_source(self) -> tuple[np.ndarray, str]:
        _t, encoder_rpm, tach_rpm = self._instrument_arrays()
        if self.filter_source.currentIndex() == 1:
            return tach_rpm, "Tachometer"
        return encoder_rpm, "Encoder diff"

    def _apply_selected_filter(self, t: np.ndarray, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return values
        clean = np.nan_to_num(values, nan=0.0)
        filter_name = self.filter_type.currentText()
        if filter_name == "No filter":
            return clean.copy()
        if filter_name == "Moving average":
            return self._moving_average(clean, self.filter_window.value())
        if filter_name == "Exponential IIR":
            alpha = float(self.filter_alpha.value())
            out = np.empty_like(clean)
            out[0] = clean[0]
            for idx in range(1, clean.size):
                out[idx] = alpha * clean[idx] + (1.0 - alpha) * out[idx - 1]
            return out

        cutoff = max(float(self.filter_cutoff.value()), 1e-6)
        out = np.empty_like(clean)
        out[0] = clean[0]
        for idx in range(1, clean.size):
            dt = max(float(t[idx] - t[idx - 1]), 1e-6)
            alpha = 1.0 - math.exp(-cutoff * dt)
            out[idx] = out[idx - 1] + alpha * (clean[idx] - out[idx - 1])
        return out

    def _refresh_instrumentation_display(self) -> None:
        if not hasattr(self, "encoder_readouts"):
            return
        raw_count = int(self.instrument["raw_count"][-1]) if self.instrument["raw_count"] else self.encoder_zero_raw
        display_count = self._display_count_from_raw(raw_count)
        angle_deg = display_count * 360.0 / max(float(self.encoder_expected_cpr.value()), 1.0)
        angle_rad = angle_deg * math.pi / 180.0
        encoder_rpm = float(self.instrument["encoder_rpm"][-1]) if self.instrument["encoder_rpm"] else 0.0
        tach_rpm = float(self.instrument["tach_rpm"][-1]) if self.instrument["tach_rpm"] else float("nan")
        direction = "positive" if display_count > 0 else "negative" if display_count < 0 else "zero"
        measured = "--"
        if self.marked_counts_per_rev is not None:
            expected = max(float(self.encoder_expected_cpr.value()), 1.0)
            error_pct = 100.0 * (self.marked_counts_per_rev - expected) / expected
            measured = f"{self.marked_counts_per_rev:.1f} ({error_pct:+.2f}%)"
        values = {
            "raw_count": f"{raw_count:d}",
            "display_count": f"{display_count:.1f}",
            "angle_deg": f"{angle_deg:.3f}",
            "angle_rad": f"{angle_rad:.5f}",
            "encoder_rpm": f"{encoder_rpm:.2f}",
            "tach_rpm": f"{tach_rpm:.2f}" if np.isfinite(tach_rpm) else "--",
            "measured_cpr": measured,
            "direction": direction,
        }
        for key, text in values.items():
            self.encoder_readouts[key].setText(text)
        try:
            self.encoder_readouts["student_angle"].setText(f"{self._student_angle_value():.6g}")
        except Exception:
            pass
        self._refresh_filter_display()

    def _refresh_filter_display(self) -> None:
        if not hasattr(self, "cutoff_hz_label"):
            return
        t, _encoder_rpm, _tach_rpm = self._instrument_arrays()
        values, source_name = self._selected_filter_source()
        if t.size < 2 or values.size < 2:
            cutoff_hz = float(self.filter_cutoff.value()) / (2.0 * math.pi)
            self.cutoff_hz_label.setText(f"{cutoff_hz:.4g}")
            return
        filtered = self._apply_selected_filter(t, values)
        self.instrument["filtered_rpm"][-filtered.size :] = filtered.tolist()
        finite_values = values[np.isfinite(values)]
        finite_filtered = filtered[np.isfinite(filtered)]
        raw_std = float(np.std(finite_values)) if finite_values.size else 0.0
        filtered_std = float(np.std(finite_filtered)) if finite_filtered.size else 0.0
        reduction = 100.0 * (1.0 - filtered_std / raw_std) if raw_std > 1e-9 else 0.0
        cutoff_hz = float(self.filter_cutoff.value()) / (2.0 * math.pi)
        self.cutoff_hz_label.setText(f"{cutoff_hz:.4g}")
        self.raw_noise_label.setText(f"{raw_std:.3f} RPM")
        self.filtered_noise_label.setText(f"{filtered_std:.3f} RPM")
        self.noise_reduction_label.setText(f"{reduction:.1f}%")
        if self.current_lab().analysis == "filtering":
            self.analysis_plot.clear()
            self.analysis_plot.setLabel("left", "Speed", units="RPM")
            self.analysis_plot.setLabel("bottom", "Time", units="s")
            self.analysis_plot.plot(t, values, name=f"{source_name} raw", pen=pg.mkPen("#94a3b8", width=1))
            self.analysis_plot.plot(t, filtered, name=self.filter_type.currentText(), pen=pg.mkPen("#2563eb", width=2))
            if source_name != "Tachometer":
                tach = np.asarray(self.instrument["tach_rpm"], dtype=float)
                if tach.size == t.size and np.any(np.isfinite(tach)):
                    self.analysis_plot.plot(t, tach, name="Tachometer", pen=pg.mkPen("#f97316", width=2))

    def _update_axis_labels(self) -> None:
        mode = self.current_mode()
        if mode == 3:
            label = "Position"
            units = "deg"
        elif mode == 2:
            label = "Speed"
            units = "RPM"
        else:
            label = "Input / Speed"
            units = "PWM / RPM"
        self.response_plot.setLabel("left", label, units=units)
        self.analysis_plot.setLabel("bottom", "Time", units="s")
        self.analysis_plot.setLabel("left", "Analysis")

    def _update_plots(self) -> None:
        t = np.asarray(self.data["t"], dtype=float)
        if t.size == 0:
            return
        ref = np.asarray(self.data["ref"], dtype=float)
        meas = np.asarray(self.data["meas"], dtype=float)
        pwm = np.asarray(self.data["pwm"], dtype=float)
        current = np.asarray(self.data["current"], dtype=float)
        self.reference_curve.setData(t, ref)
        self.measured_curve.setData(t, meas)
        self.pwm_curve.setData(t, pwm)
        self.current_curve.setData(t, current * 50.0)

    def _arrays(self):
        return (
            np.asarray(self.data["t"], dtype=float),
            np.asarray(self.data["ref"], dtype=float),
            np.asarray(self.data["meas"], dtype=float),
            np.asarray(self.data["pwm"], dtype=float),
        )

    def analyze_current_lab(self) -> None:
        lab = self.current_lab()
        t, ref, meas, pwm = self._arrays()
        if lab.analysis == "interfacing":
            text = self._analyze_interfacing()
            self.last_analysis_text = text
            self.results.setPlainText(text)
            return
        if lab.analysis == "filtering" and len(self.instrument["t"]) >= 5:
            text = self._analyze_filtering()
            self.last_analysis_text = text
            self.results.setPlainText(text)
            return
        if t.size < 5:
            self.results.setPlainText("Not enough samples yet. Run the lab before analyzing.")
            return

        self.analysis_plot.clear()
        analysis = lab.analysis
        if analysis == "step":
            text = self._analyze_step(t, ref, meas)
        elif analysis == "frequency":
            text = self._analyze_frequency(t, ref, meas)
        elif analysis == "estimate":
            text = self._analyze_estimation(t, pwm if np.ptp(pwm) > 1 else ref, meas)
        elif analysis == "model":
            text = self._analyze_model(t, ref, meas)
        elif analysis == "stability":
            text = self._analyze_stability(t, ref, meas)
        elif analysis == "root_locus":
            text = self._analyze_root_locus(t, ref, meas)
        elif analysis == "sse":
            text = self._analyze_sse(t, ref, meas, pwm)
        elif analysis in ("control", "discrete"):
            text = self._analyze_control(t, ref, meas, pwm)
        else:
            text = self._analyze_summary(t, ref, meas, pwm)
        self.last_analysis_text = text
        self.results.setPlainText(text)

    def _analyze_summary(self, t, ref, meas, pwm) -> str:
        return "\n".join(
            [
                f"Samples: {t.size}",
                f"Duration: {t[-1] - t[0]:.3f} s",
                f"Reference range: {np.min(ref):.3f} to {np.max(ref):.3f}",
                f"Measured range: {np.min(meas):.3f} to {np.max(meas):.3f}",
                f"Final measured value: {meas[-1]:.3f}",
                f"Final command: {pwm[-1]:.3f} PWM",
            ]
        )

    def _moving_average(self, values, window: int):
        window = max(1, min(window, values.size))
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="same")

    def _analyze_interfacing(self) -> str:
        raw_count = int(self.instrument["raw_count"][-1]) if self.instrument["raw_count"] else self.encoder_zero_raw
        display_count = self._display_count_from_raw(raw_count)
        expected = max(float(self.encoder_expected_cpr.value()), 1.0)
        deg_gain = 360.0 / expected
        rad_gain = 2.0 * math.pi / expected
        measured = "not marked"
        if self.marked_counts_per_rev is not None:
            error_pct = 100.0 * (self.marked_counts_per_rev - expected) / expected
            measured = f"{self.marked_counts_per_rev:.1f} counts/rev ({error_pct:+.2f}% vs expected)"
        positive_direction = "positive count direction" if display_count >= 0 else "negative count direction"
        return "\n".join(
            [
                "Interfacing analysis",
                f"Selected count mode: {self.encoder_edge_mode.currentText()}",
                f"Raw HIL encoder count: {raw_count}",
                f"Displayed count since Zero: {display_count:.1f}",
                f"Expected counts/rev: {expected:.1f}",
                f"Measured one-revolution count: {measured}",
                f"Gain to degrees: {deg_gain:.9g} deg/count",
                f"Gain to radians: {rad_gain:.9g} rad/count",
                f"Current sign observation: {positive_direction}",
                "When the Quanser backend is reopened, the HIL encoder count is zeroed by software; students should notice that restart changes the reference origin.",
            ]
        )

    def _analyze_filtering(self) -> str:
        t, _encoder_rpm, _tach_rpm = self._instrument_arrays()
        values, source_name = self._selected_filter_source()
        if t.size < 5 or values.size < 5:
            return "Need more instrumentation samples. Connect the Qube and run or manually rotate the disk."
        filtered = self._apply_selected_filter(t, values)
        finite_values = values[np.isfinite(values)]
        finite_filtered = filtered[np.isfinite(filtered)]
        raw_std = float(np.std(finite_values)) if finite_values.size else 0.0
        filtered_std = float(np.std(finite_filtered)) if finite_filtered.size else 0.0
        residual_std = float(np.std(np.nan_to_num(values - filtered))) if values.size == filtered.size else 0.0
        reduction = 100.0 * (1.0 - filtered_std / raw_std) if raw_std > 1e-9 else 0.0
        cutoff_hz = float(self.filter_cutoff.value()) / (2.0 * math.pi)
        self.analysis_plot.clear()
        self.analysis_plot.setLabel("left", "Speed", units="RPM")
        self.analysis_plot.setLabel("bottom", "Time", units="s")
        self.analysis_plot.plot(t, values, name=f"{source_name} raw", pen=pg.mkPen("#94a3b8", width=1))
        self.analysis_plot.plot(t, filtered, name=self.filter_type.currentText(), pen=pg.mkPen("#2563eb", width=2))
        tach = np.asarray(self.instrument["tach_rpm"], dtype=float)
        if tach.size == t.size and np.any(np.isfinite(tach)) and source_name != "Tachometer":
            self.analysis_plot.plot(t, tach, name="Tachometer", pen=pg.mkPen("#f97316", width=2))
        return "\n".join(
            [
                "Filtering analysis",
                f"Source: {source_name}",
                f"Filter: {self.filter_type.currentText()}",
                f"Moving-average window: {self.filter_window.value()} samples",
                f"Exponential alpha: {self.filter_alpha.value():.4f}",
                f"Low-pass cutoff: {self.filter_cutoff.value():.4g} rad/s = {cutoff_hz:.4g} Hz",
                f"Raw standard deviation: {raw_std:.3f} RPM",
                f"Filtered standard deviation: {filtered_std:.3f} RPM",
                f"Raw-filter residual std: {residual_std:.3f} RPM",
                f"Std reduction estimate: {reduction:.1f}%",
                "Lower cutoff/windowed filters reduce noise but add lag and attenuate fast changes; higher cutoff follows motion faster but leaves more quantization noise.",
            ]
        )

    def _step_segment(self, t, ref, meas):
        if t.size < 5:
            return t, ref, meas
        change = np.where(np.abs(np.diff(ref)) > max(1.0, 0.1 * np.ptp(ref)))[0]
        if change.size:
            start = max(0, int(change[0]) + 1)
        else:
            start = 0
        end = t.size
        if change.size > 1:
            end = max(start + 5, int(change[1]) + 1)
        return t[start:end] - t[start], ref[start:end], meas[start:end]

    def _estimate_first_order_step(self, t, ref, meas):
        ts, rs, ys = self._step_segment(t, ref, meas)
        if ts.size < 5:
            return None
        y0 = float(ys[0])
        yss = float(np.median(ys[max(1, int(0.8 * ys.size)) :]))
        u0 = float(rs[0])
        du = u0 if abs(u0) > 1e-9 else float(np.max(np.abs(rs)))
        gain = (yss - y0) / du if abs(du) > 1e-9 else 0.0
        target = y0 + 0.632 * (yss - y0)
        if yss >= y0:
            idx = np.where(ys >= target)[0]
        else:
            idx = np.where(ys <= target)[0]
        tau = float(ts[idx[0]]) if idx.size else float(max(ts[-1] / 3.0, 1e-3))
        return {"K": gain, "tau": max(tau, 1e-3), "y0": y0, "yss": yss, "t": ts, "y": ys}

    def _analyze_step(self, t, ref, meas) -> str:
        model = self._estimate_first_order_step(t, ref, meas)
        if model is None:
            return "Could not estimate a step model from the recorded data."
        self.last_model = model
        ts = model["t"]
        y0 = model["y0"]
        yfit = y0 + (model["yss"] - y0) * (1.0 - np.exp(-ts / model["tau"]))
        self.analysis_plot.setLabel("left", "Step response", units="RPM")
        self.analysis_plot.plot(ts, model["y"], name="Measured", pen=pg.mkPen("#dc2626", width=2))
        self.analysis_plot.plot(ts, yfit, name="First-order fit", pen=pg.mkPen("#0f766e", width=2, style=Qt.PenStyle.DashLine))
        rise = self._rise_time(ts, model["y"])
        settling = self._settling_time(ts, model["y"], model["yss"])
        return "\n".join(
            [
                "First-order step estimate",
                f"DC gain K: {model['K']:.5g} RPM/PWM",
                f"Time constant tau: {model['tau']:.5g} s",
                f"Initial value: {model['y0']:.3f} RPM",
                f"Steady-state value: {model['yss']:.3f} RPM",
                f"10-90 rise time: {rise:.3f} s" if rise is not None else "10-90 rise time: not found",
                f"2 percent settling time: {settling:.3f} s" if settling is not None else "2 percent settling time: not found",
            ]
        )

    def _rise_time(self, t, y):
        y0 = float(y[0])
        yss = float(np.median(y[max(1, int(0.8 * y.size)) :]))
        span = yss - y0
        if abs(span) < 1e-9:
            return None
        low = y0 + 0.1 * span
        high = y0 + 0.9 * span
        if span > 0:
            lo_idx = np.where(y >= low)[0]
            hi_idx = np.where(y >= high)[0]
        else:
            lo_idx = np.where(y <= low)[0]
            hi_idx = np.where(y <= high)[0]
        if lo_idx.size and hi_idx.size:
            return float(t[hi_idx[0]] - t[lo_idx[0]])
        return None

    def _settling_time(self, t, y, yss):
        band = max(abs(yss) * 0.02, 1.0)
        err = np.abs(y - yss)
        for idx in range(err.size):
            if np.all(err[idx:] <= band):
                return float(t[idx])
        return None

    def _analyze_frequency(self, t, ref, meas) -> str:
        if t.size < 16:
            return "Need more samples for frequency-response analysis."
        dt = float(np.median(np.diff(t)))
        u = ref - np.mean(ref)
        y = meas - np.mean(meas)
        freqs = np.fft.rfftfreq(t.size, dt)
        U = np.fft.rfft(u)
        Y = np.fft.rfft(y)
        valid = freqs > 0
        if not np.any(valid):
            return "No nonzero excitation frequency found."
        idxs = np.where(valid)[0]
        idx = idxs[int(np.argmax(np.abs(U[idxs])))]
        gain = abs(Y[idx]) / max(abs(U[idx]), 1e-9)
        phase = math.degrees(np.angle(Y[idx]) - np.angle(U[idx]))
        phase = (phase + 180.0) % 360.0 - 180.0
        self.analysis_plot.setLabel("bottom", "Frequency", units="Hz")
        self.analysis_plot.setLabel("left", "Magnitude")
        self.analysis_plot.plot(freqs[valid], np.abs(Y[valid]) / np.maximum(np.abs(U[valid]), 1e-9), name="Gain", pen=pg.mkPen("#2563eb", width=2))
        return "\n".join(
            [
                "Frequency-response estimate",
                f"Dominant excitation frequency: {freqs[idx]:.4f} Hz",
                f"Magnitude ratio: {gain:.5g}",
                f"Phase shift: {phase:.2f} deg",
            ]
        )

    def _analyze_estimation(self, t, u, y) -> str:
        if least_squares is None:
            return self._analyze_step(t, u, y) + "\n\nscipy is not available; step estimate used instead."
        if t.size < 8:
            return "Need more samples for parameter estimation."
        dt = np.diff(t, prepend=t[0])
        dt[0] = float(np.median(dt[1:])) if dt.size > 1 else 0.02
        u = np.asarray(u, dtype=float)
        y = np.asarray(y, dtype=float)

        def simulate(params):
            gain, tau, y0 = params
            tau = max(abs(tau), 1e-4)
            out = np.empty_like(y)
            out[0] = y0
            for k in range(1, y.size):
                alpha = min(max(dt[k] / tau, 0.0), 1.0)
                out[k] = out[k - 1] + alpha * (gain * u[k - 1] - out[k - 1])
            return out

        step_guess = self._estimate_first_order_step(t, u, y)
        if step_guess is None:
            x0 = np.array([0.5, 0.25, y[0]], dtype=float)
        else:
            x0 = np.array([step_guess["K"], step_guess["tau"], step_guess["y0"]], dtype=float)
        result = least_squares(lambda p: simulate(p) - y, x0=x0, max_nfev=3000)
        gain, tau, y0 = result.x
        tau = abs(float(tau))
        model = {"K": float(gain), "tau": tau, "y0": float(y0), "yss": float(gain * np.median(u[-max(3, u.size // 5) :]))}
        self.last_model = model
        yhat = simulate(result.x)
        rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
        self.analysis_plot.setLabel("left", "Model fit", units="RPM")
        self.analysis_plot.plot(t, y, name="Measured", pen=pg.mkPen("#dc2626", width=2))
        self.analysis_plot.plot(t, yhat, name="Model", pen=pg.mkPen("#0f766e", width=2, style=Qt.PenStyle.DashLine))
        return "\n".join(
            [
                "Least-squares first-order estimate",
                f"Model: G(s) = {gain:.5g} / ({tau:.5g}s + 1)",
                f"Initial condition y0: {y0:.5g}",
                f"Fit RMSE: {rmse:.5g} RPM",
                f"Optimizer status: {result.message}",
            ]
        )

    def _ensure_model(self, t, ref, meas):
        if self.last_model is not None:
            return self.last_model
        return self._estimate_first_order_step(t, ref, meas)

    def _analyze_model(self, t, ref, meas) -> str:
        model = self._ensure_model(t, ref, meas)
        if model is None:
            return "No usable model found. Run a step or parameter-estimation lab first."
        self.last_model = model
        k = model["K"]
        tau = model["tau"]
        return "\n".join(
            [
                "Model forms",
                f"Transfer function: G(s) = {k:.5g} / ({tau:.5g}s + 1)",
                "State-space with x = speed:",
                f"A = {-1.0 / tau:.5g}",
                f"B = {k / tau:.5g}",
                "C = 1",
                "D = 0",
                f"Open-loop pole: {-1.0 / tau:.5g} rad/s",
            ]
        )

    def _analyze_stability(self, t, ref, meas) -> str:
        model = self._ensure_model(t, ref, meas)
        if model is None:
            return "No usable model found. Run a modeling lab first."
        self.last_model = model
        k = model["K"]
        tau = model["tau"]
        kp = self.kp.value()
        a0 = 1.0 + k * kp
        stable = tau > 0 and a0 > 0
        return "\n".join(
            [
                "Routh-Hurwitz check for speed proportional loop",
                f"Characteristic polynomial: {tau:.5g}s + {a0:.5g}",
                f"Closed-loop pole: {-a0 / tau:.5g} rad/s",
                f"Stable: {'yes' if stable else 'no'}",
                "For this first-order approximation, all polynomial coefficients must have the same sign.",
            ]
        )

    def _analyze_root_locus(self, t, ref, meas) -> str:
        model = self._ensure_model(t, ref, meas)
        if model is None:
            return "No usable model found. Run a modeling lab first."
        self.last_model = model
        k = model["K"]
        tau = model["tau"]
        gains = np.linspace(0.0, 5.0, 101)
        poles = -(1.0 + k * gains) / tau
        self.analysis_plot.setLabel("bottom", "Real axis", units="rad/s")
        self.analysis_plot.setLabel("left", "Imag axis", units="rad/s")
        self.analysis_plot.plot(poles, np.zeros_like(poles), name="Speed-loop locus", pen=None, symbol="o", symbolSize=5, symbolBrush="#2563eb")
        selected_pole = -(1.0 + k * self.kp.value()) / tau
        self.analysis_plot.plot([selected_pole], [0.0], name="Current Kp", pen=None, symbol="x", symbolSize=12, symbolBrush="#dc2626")
        return "\n".join(
            [
                "Root-locus sweep for first-order speed model",
                f"Plant model: G(s) = {k:.5g} / ({tau:.5g}s + 1)",
                "Gain sweep: Kp = 0 to 5",
                f"Current Kp: {self.kp.value():.5g}",
                f"Current closed-loop pole: {selected_pole:.5g} rad/s",
            ]
        )

    def _analyze_control(self, t, ref, meas, pwm) -> str:
        error = ref - meas
        final_count = max(3, int(0.2 * error.size))
        final_error = float(np.mean(error[-final_count:]))
        ref_span = float(np.max(ref) - np.min(ref))
        final_ref = float(np.mean(ref[-final_count:]))
        final_meas = float(np.mean(meas[-final_count:]))
        overshoot = 0.0
        if abs(ref_span) > 1e-9:
            peak = float(np.max(meas))
            target = float(np.max(ref))
            overshoot = max(0.0, (peak - target) / abs(ref_span) * 100.0)
        settling = self._settling_time(t - t[0], meas, final_meas)
        return "\n".join(
            [
                "Closed-loop response",
                f"Final reference mean: {final_ref:.3f}",
                f"Final measured mean: {final_meas:.3f}",
                f"Final tracking error mean: {final_error:.3f}",
                f"Peak command: {np.max(np.abs(pwm)):.3f} PWM",
                f"Overshoot estimate: {overshoot:.2f} %",
                f"Settling time estimate: {settling:.3f} s" if settling is not None else "Settling time estimate: not found",
            ]
        )

    def _analyze_sse(self, t, ref, meas, pwm) -> str:
        base = self._analyze_control(t, ref, meas, pwm)
        error = ref - meas
        final_count = max(3, int(0.2 * error.size))
        sse = float(np.mean(error[-final_count:]))
        return base + "\n" + "\n".join(
            [
                "",
                "Steady-state error focus",
                f"Mean steady-state error: {sse:.3f}",
                f"Mean absolute steady-state error: {np.mean(np.abs(error[-final_count:])):.3f}",
            ]
        )

    def export_csv(self) -> None:
        if not self.data["t"] and not self.instrument["t"]:
            self.results.setPlainText("No samples to export.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export motor lab CSV",
            f"{self.current_lab().title.replace(' ', '_').replace('/', '-')}.csv",
            "CSV files (*.csv)",
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["section", "lab", "time_s", "reference", "measured", "dt_ms", "current_a", "pwm"])
            for row in zip(self.data["t"], self.data["ref"], self.data["meas"], self.data["dt_ms"], self.data["current"], self.data["pwm"]):
                writer.writerow(["control", self.current_lab().title, *row])
            writer.writerow([])
            writer.writerow(["section", "lab", "time_s", "raw_count", "display_count", "position_deg", "encoder_rpm", "tach_rpm", "filtered_rpm"])
            for row in zip(
                self.instrument["t"],
                self.instrument["raw_count"],
                self.instrument["display_count"],
                self.instrument["position_deg"],
                self.instrument["encoder_rpm"],
                self.instrument["tach_rpm"],
                self.instrument["filtered_rpm"],
            ):
                writer.writerow(["instrumentation", self.current_lab().title, *row])
        self.status_label.setText(f"CSV exported: {Path(path).name}")

    def export_report(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export motor lab report",
            f"{self.current_lab().title.replace(' ', '_').replace('/', '-')}_report.txt",
            "Text files (*.txt)",
        )
        if not path:
            return
        lines = [
            self.current_lab().title,
            "",
            "Goal:",
            self.current_lab().goal,
            "",
            "Settings:",
            f"Mode: {self.mode_combo.currentText()}",
            f"Signal: {self.signal_combo.currentText()}",
            f"Amplitude: {self.amplitude.value()}",
            f"Period: {self.period.value()} ms",
            f"Kp/Ki/Kd: {self.kp.value()} / {self.ki.value()} / {self.kd.value()}",
            f"Encoder count mode: {self.encoder_edge_mode.currentText()}",
            f"Expected counts/rev: {self.encoder_expected_cpr.value()}",
            f"Encoder formula: {self.encoder_formula.text()}",
            f"Filter source/type: {self.filter_source.currentText()} / {self.filter_type.currentText()}",
            f"Filter window/alpha/cutoff: {self.filter_window.value()} / {self.filter_alpha.value()} / {self.filter_cutoff.value()} rad/s",
            "",
            "Analysis:",
            self.last_analysis_text or self.results.toPlainText() or "No analysis has been run.",
        ]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        self.status_label.setText(f"Report exported: {Path(path).name}")

    def closeEvent(self, event) -> None:
        self.release_hardware()
        super().closeEvent(event)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = MotorLabsWindow()
    app.aboutToQuit.connect(window.release_hardware)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
