# Larynx Force Sensor - Serial data acquisition and plotting GUI
# Connects to ESP-32 via USB serial, plots A/D values vs time, records to CSV.
#(c) Kai Kuck, 2026


"""****************************************
HOW TO CREATE A STAND-ALONE EXE FILE FROM THIS:

Use PyInstaller, which bundles the Python script and everything it needs 
(including the Python interpreter itself) into a standalone executable.

1. Install PyInstaller: pip install pyinstaller

2. Run PyInstaller: pyinstaller --onefile larynx_serial_gui.py
The --onefile option creates a single executable file that includes all dependencies.
The output file will be in the dist/ directory.

To run the standalone executable:
./dist/larynx_serial_gui

To create a Windows executable:
pyinstaller --onefile --windowed larynx_serial_gui.py

*****************************************"""

import json
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import AutoLocator, FuncFormatter, MaxNLocator
import matplotlib.animation as animation


# ---------------------------------------------------------------------------
# Paths and defaults
# ---------------------------------------------------------------------------
SETTINGS_FOLDER_PREFERRED = (
    r"C:\KaisDir\Dropbox\K0_Work\0_actualwork\Projects\Intubation_Monitoring"
    r"\Larynx_ForceSensor\Sw\0_Settings"
)
SETTINGS_FOLDER_FALLBACK = r"C:\\"
SETTINGS_FILENAME = "LarynxSettings"
DEFAULT_DATA_FOLDER = (
    r"C:\KaisDir\Dropbox\K0_Work\0_actualwork\Projects\Intubation_Monitoring"
    r"\Larynx_ForceSensor\Data"
)
DEFAULT_COM_PORT = "COM3"
BAUD_OPTIONS = [115200, 230400, 460800, 921600]
DEFAULT_BAUD = 115200
PLOT_TIME_MIN_SEC = 60
PLOT_TIME_MAX_SEC = 420
DEFAULT_PLOT_TIME_SEC = 180
RECORD_CHUNK_DURATION_SEC = 15
SAMPLES_PER_SEC = 16  # ESP32 sampling rate
# Serial line order is ch4;ch3;ch2;ch1 — index in split parts for logical ch1..ch4
SERIAL_PART_INDEX_FOR_LOGICAL_CH = (3, 2, 1, 0)
Y_AXIS_RANGE_MIN = 0
Y_AXIS_RANGE_MAX = 50000
DEFAULT_Y_AXIS_MIN = 0
DEFAULT_Y_AXIS_MAX = 2500
DEFAULT_Y_AXIS_MODE = "auto"  # "auto" or "fixed"
PRIMARY_Y_MAX_FLOOR_MIN = 0.0
PRIMARY_Y_MAX_FLOOR_MAX = 500.0
DEFAULT_PRIMARY_Y_MAX_FLOOR = 50.0
CALIB_DURATION_MIN_SEC = 1
CALIB_DURATION_MAX_SEC = 60
DEFAULT_CALIB_DURATION_SEC = 10
MOVAVG_LENGTH_MIN_SEC = 0
MOVAVG_LENGTH_MAX_SEC = 30
DEFAULT_MOVAVG_LENGTH_SEC = 3
CALIB_Y_AXIS_INITIAL_MIN = 0
CALIB_Y_AXIS_INITIAL_MAX = 27000
CALIB_Y_AXIS_MIN_SPAN = 500
DEFAULT_CALIB_CHANNEL = 1
CALIB_CHANNEL_MIN = 1
CALIB_CHANNEL_MAX = 4
# Channel plot selection: which channels to show (default Ch1 only)
DEFAULT_PLOT_CHANNEL_1 = True
DEFAULT_PLOT_CHANNEL_2 = False
DEFAULT_PLOT_CHANNEL_3 = False
DEFAULT_PLOT_CHANNEL_4 = False
CHANNEL_COLORS = ("blue", "red", "green", "orange")  # Ch1..Ch4
CALIB_VALUE_FONT = ("Consolas", 16, "bold")
MAIN_VALUE_FONT = CALIB_VALUE_FONT
CALIB_DAC_VALUE_WIDTH = 8  # fixed character width for DAC / avg values (no layout shift)
MAIN_LATEST_VALUE_WIDTH = 8  # numeric part for toolbar (DAC ints or ####.#)
MAIN_LATEST_UNIT_WIDTH = 2   # "g", "N", or empty
MAIN_LATEST_INNER_PAD = 2    # pixels between "ChX:" and its value
MAIN_LATEST_GROUP_PAD = 14   # pixels between channel groups
MAIN_MIN_WIDTH = 880         # floor so toolbar Latest row is not clipped
CALIB_DAC_COLOR = "blue"
CALIB_MOVAVG_COLOR = "orange"
CALIB_SLIDER_LEFT_LABELS = ("Collection time:", "Moving average:")
CALIB_SLIDER_LABEL_WIDTH = max(len(t) for t in CALIB_SLIDER_LEFT_LABELS)
CALIB_SLIDER_GAP = 4  # pixels between label / slider / value label
PREF_VALUE_LABEL_GAP = 4  # pixels between slider/entry and value label to the right
PREF_MIN_WIDTH = 820
PREF_SLOPE_LABEL_WIDTH = 16  # chars; fits "slope: 999.9999"
DEFAULT_DATA_FILE_PREFIX = "LarynxData"
DATA_FILE_PREFIX_MAX_LEN = 30
# Characters not allowed in Windows filenames: \ / : * ? " < > |
WINDOWS_FILENAME_FORBIDDEN = set('\\/:*?"<>|')
# DAC -> mass model: mass (g) = (DAC / a)^(-1/b); Newtons = mass * GRAMS_TO_NEWTONS
DEFAULT_CHANNEL_MODEL_A = 1.0
DEFAULT_CHANNEL_MODEL_B = 1.0
DEFAULT_CHANNEL_TARE_VALUE = 0.0
DEFAULT_CHANNEL_SLOPE_FACTOR = 1.0
DEFAULT_CHANNEL_TARGET_MASS = 0.0
DEFAULT_CHANNEL_DISPLAY_MODE = "dac"  # "dac", "grams", or "newtons"
CHANNEL_DISPLAY_MODES = ("dac", "grams", "newtons")
CHANNEL_DISPLAY_MODE_LABELS = {
    "dac": "Raw DAC",
    "grams": "Mass (g)",
    "newtons": "Force (N)",
}
DISPLAY_MODE_LABEL_TO_KEY = {v: k for k, v in CHANNEL_DISPLAY_MODE_LABELS.items()}
GRAMS_TO_NEWTONS = 0.00981
MODEL_FORMULA_TEXT = "mass (g) = (DAC / a)^(-1/b)   |   Force (N) = mass × 0.00981"





# *****************************************************************************
def sanitize_windows_filename_prefix(text):
    """
    Returns a string safe for use as a Windows filename prefix: max DATA_FILE_PREFIX_MAX_LEN
    characters, with any character in \\ / : * ? " < > | removed.
    """
    if not text or not isinstance(text, str):
        return ""
    s = "".join(c for c in text if c not in WINDOWS_FILENAME_FORBIDDEN)
    return s[:DATA_FILE_PREFIX_MAX_LEN].strip()


# *****************************************************************************
def _channel_defaults_for_settings():
    """Default model/display keys for channels 1..4."""
    d = {}
    for n in range(1, 5):
        d[f"channel_{n}_model_a"] = DEFAULT_CHANNEL_MODEL_A
        d[f"channel_{n}_model_b"] = DEFAULT_CHANNEL_MODEL_B
        d[f"channel_{n}_tare_value"] = DEFAULT_CHANNEL_TARE_VALUE
        d[f"channel_{n}_slope_factor"] = DEFAULT_CHANNEL_SLOPE_FACTOR
        d[f"channel_{n}_target_mass"] = DEFAULT_CHANNEL_TARGET_MASS
        d[f"channel_{n}_display_mode"] = DEFAULT_CHANNEL_DISPLAY_MODE
    return d


# *****************************************************************************
def dac_to_mass_grams(dac, a, b):
    """
    mass (g) = (DAC / a)^(-1/b). Returns None if conversion is invalid.
    """
    try:
        dac_f = float(dac)
        a_f = float(a)
        b_f = float(b)
    except (TypeError, ValueError):
        return None
    if a_f <= 0 or b_f == 0:
        return None
    ratio = dac_f / a_f
    if ratio <= 0:
        return None
    try:
        return ratio ** (-1.0 / b_f)
    except (ValueError, OverflowError):
        return None


# *****************************************************************************
def mass_zeroed_from_dac(dac, settings, channel_index, a=None, b=None):
    """mass_zeroed = mass_raw - tare for channel_index (0..3)."""
    ch = channel_index + 1
    if a is None:
        a = settings.get(f"channel_{ch}_model_a", DEFAULT_CHANNEL_MODEL_A)
    if b is None:
        b = settings.get(f"channel_{ch}_model_b", DEFAULT_CHANNEL_MODEL_B)
    mass_raw = dac_to_mass_grams(dac, a, b)
    if mass_raw is None:
        return None
    try:
        tare = float(settings.get(f"channel_{ch}_tare_value", DEFAULT_CHANNEL_TARE_VALUE))
    except (TypeError, ValueError):
        tare = DEFAULT_CHANNEL_TARE_VALUE
    return mass_raw - tare


# *****************************************************************************
SCALE_APPLY_BUTTON_TEXT = "Click here when known weight is applied"


# *****************************************************************************
def convert_dac_for_display(dac, settings, channel_index):
    """
    Convert raw DAC to the value shown/plotted for channel_index (0..3).
    display_mode: dac | grams | newtons
    """
    ch = channel_index + 1
    mode = settings.get(f"channel_{ch}_display_mode", DEFAULT_CHANNEL_DISPLAY_MODE)
    if mode not in CHANNEL_DISPLAY_MODES:
        mode = DEFAULT_CHANNEL_DISPLAY_MODE
    if mode == "dac":
        try:
            return float(dac)
        except (TypeError, ValueError):
            return None
    a = settings.get(f"channel_{ch}_model_a", DEFAULT_CHANNEL_MODEL_A)
    b = settings.get(f"channel_{ch}_model_b", DEFAULT_CHANNEL_MODEL_B)
    mass_raw = dac_to_mass_grams(dac, a, b)
    if mass_raw is None:
        return None
    try:
        tare = float(settings.get(f"channel_{ch}_tare_value", DEFAULT_CHANNEL_TARE_VALUE))
    except (TypeError, ValueError):
        tare = DEFAULT_CHANNEL_TARE_VALUE
    mass_zeroed = mass_raw - tare
    try:
        slope = float(settings.get(f"channel_{ch}_slope_factor", DEFAULT_CHANNEL_SLOPE_FACTOR))
    except (TypeError, ValueError):
        slope = DEFAULT_CHANNEL_SLOPE_FACTOR
    mass_zeroedscaled = mass_zeroed * slope
    if mode == "grams":
        return mass_zeroedscaled
    if mode == "newtons":
        return mass_zeroedscaled * GRAMS_TO_NEWTONS
    return None


# *****************************************************************************
def format_display_value_parts(value, settings, channel_index):
    """Return (numeric_text, unit_suffix) for toolbar labels; unit is '' for DAC."""
    if value is None:
        return "—", ""
    ch = channel_index + 1
    mode = settings.get(f"channel_{ch}_display_mode", DEFAULT_CHANNEL_DISPLAY_MODE)
    if mode == "dac":
        return str(int(round(value))), ""
    if mode == "grams":
        return f"{value:.1f}", "g"
    if mode == "newtons":
        return f"{value:.1f}", "N"
    return str(int(round(value))), ""


# *****************************************************************************
def format_display_value(value, settings, channel_index):
    """Format a converted value for toolbar labels."""
    num, unit = format_display_value_parts(value, settings, channel_index)
    if unit:
        return f"{num} {unit}"
    return num


# *****************************************************************************
CSV_NA = "N/A"
CSV_RECORD_HEADER = (
    "datetime; seconds; "
    "ch1-DAC; ch2-DAC; ch3-DAC; ch4-DAC; "
    "ch1-grams; ch2-grams; ch3-grams; ch4-grams; "
    "ch1-Newtons; ch2-Newtons; ch3-Newtons; ch4-Newtons; "
    "ch1-calib-a; ch2-calib-a; ch3-calib-a; ch4-calib-a; "
    "ch1-calib-b; ch2-calib-b; ch3-calib-b; ch4-calib-b; "
    "ch1-calib-tarevalue; ch2-calib-tarevalue; ch3-calib-tarevalue; ch4-calib-tarevalue; "
    "ch1-calib-slope; ch2-calib-slope; ch3-calib-slope; ch4-calib-slope; "
    "ch1-Events; ch2-Events; ch3-Events; ch4-Events"
)


# *****************************************************************************
def channel_calib_snapshot(settings, channel_index):
    """Current calibration parameters for one channel (0..3)."""
    ch = channel_index + 1
    return {
        "a": settings.get(f"channel_{ch}_model_a", DEFAULT_CHANNEL_MODEL_A),
        "b": settings.get(f"channel_{ch}_model_b", DEFAULT_CHANNEL_MODEL_B),
        "tare": settings.get(f"channel_{ch}_tare_value", DEFAULT_CHANNEL_TARE_VALUE),
        "slope": settings.get(f"channel_{ch}_slope_factor", DEFAULT_CHANNEL_SLOPE_FACTOR),
    }


# *****************************************************************************
def channel_mass_grams_csv(dac, settings, channel_index):
    """Zeroed, slope-scaled mass (g) for CSV; None if calibration unavailable."""
    ch = channel_index + 1
    a = settings.get(f"channel_{ch}_model_a", DEFAULT_CHANNEL_MODEL_A)
    b = settings.get(f"channel_{ch}_model_b", DEFAULT_CHANNEL_MODEL_B)
    mass_raw = dac_to_mass_grams(dac, a, b)
    if mass_raw is None:
        return None
    try:
        tare = float(settings.get(f"channel_{ch}_tare_value", DEFAULT_CHANNEL_TARE_VALUE))
    except (TypeError, ValueError):
        tare = DEFAULT_CHANNEL_TARE_VALUE
    try:
        slope = float(settings.get(f"channel_{ch}_slope_factor", DEFAULT_CHANNEL_SLOPE_FACTOR))
    except (TypeError, ValueError):
        slope = DEFAULT_CHANNEL_SLOPE_FACTOR
    return (mass_raw - tare) * slope


# *****************************************************************************
def channel_newtons_csv(dac, settings, channel_index):
    """Force (N) for CSV; None if calibration unavailable."""
    grams = channel_mass_grams_csv(dac, settings, channel_index)
    if grams is None:
        return None
    return grams * GRAMS_TO_NEWTONS


# *****************************************************************************
def _format_csv_float(value):
    """Format a numeric calibration or converted value for CSV."""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return CSV_NA


# *****************************************************************************
def _format_csv_dac(value):
    if value is None:
        return ""
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return ""


# *****************************************************************************
def build_csv_record_row(
    dt,
    seconds_since_start,
    dac_values,
    settings,
    events=None,
    calib_channels=None,
):
    """
    Build one semicolon-separated CSV data row for recording.

    calib_channels: set of channel indices (0..3) for which calib-a/b/tare/slope
    columns are filled on this row; other channels leave those columns empty.
    """
    if events is None:
        events = ["", "", "", ""]
    if calib_channels is None:
        calib_channels = set()
    ts = dt.strftime("%m/%d/%Y %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"
    sec_str = f"{seconds_since_start:.3f}"
    v = list(dac_values) if dac_values else []
    if len(v) < 4:
        v = v + [None] * (4 - len(v))
    parts = [ts, sec_str]
    for ch in range(4):
        parts.append(_format_csv_dac(v[ch]))
    for ch in range(4):
        grams = channel_mass_grams_csv(v[ch], settings, ch)
        parts.append(_format_csv_float(grams) if grams is not None else CSV_NA)
    for ch in range(4):
        newtons = channel_newtons_csv(v[ch], settings, ch)
        parts.append(_format_csv_float(newtons) if newtons is not None else CSV_NA)
    snap = [channel_calib_snapshot(settings, ch) for ch in range(4)]
    for key in ("a", "b", "tare", "slope"):
        for ch in range(4):
            if ch in calib_channels:
                parts.append(_format_csv_float(snap[ch][key]))
            else:
                parts.append("")
    for ch in range(4):
        parts.append(events[ch] if ch < len(events) else "")
    return f"{ts}; {sec_str};{';'.join(parts[2:])}\n"


# *****************************************************************************
def plot_axis_labels(settings, plot_channel_flags):
    """
    Y-axis labels for dual-axis plot: left = mass/force, right = raw DAC.
    Returns (left_ylabel, right_ylabel); empty string if that axis has no traces.
    """
    left_modes = set()
    right_has_dac = False
    for ch in range(4):
        if not plot_channel_flags[ch]:
            continue
        m = settings.get(f"channel_{ch + 1}_display_mode", DEFAULT_CHANNEL_DISPLAY_MODE)
        if m not in CHANNEL_DISPLAY_MODES:
            m = DEFAULT_CHANNEL_DISPLAY_MODE
        if m == "dac":
            right_has_dac = True
        else:
            left_modes.add(m)
    if not left_modes:
        left_label = ""
    elif left_modes == {"grams"}:
        left_label = "Mass (g)"
    elif left_modes == {"newtons"}:
        left_label = "Force (N)"
    else:
        left_label = "Mass (g),   Force (N)"
    right_label = "A/D value" if right_has_dac else ""
    return left_label, right_label


def _refresh_axis_y_ticks(ax, side):
    """Restore major tick marks and numeric labels on an axis."""
    ax.yaxis.set_major_locator(AutoLocator())
    if side == "left":
        ax.yaxis.set_tick_params(which="major", left=True, labelleft=True)
    else:
        ax.yaxis.set_tick_params(which="major", right=True, labelright=True)


def _refresh_dac_axis_y_ticks(ax):
    """Right Y-axis (DAC): integer labels, no scientific notation."""
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))

    def _fmt(value, _pos):
        return f"{int(round(value))}"

    ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.yaxis.set_tick_params(which="major", right=True, labelright=True)


def _autoscale_ylim(ax, values, margin_frac=0.02, ymax_floor=None, min_span=None):
    """Set y limits from a list of numeric values with a small margin."""
    if not values:
        return
    data_min = min(values)
    data_max = max(values)
    if data_max <= data_min:
        margin = 1.0
        data_min -= margin
        data_max += margin
    else:
        margin = (data_max - data_min) * margin_frac or 1.0
        data_min -= margin
        data_max += margin
    if ymax_floor is not None and ymax_floor > 0:
        data_max = max(data_max, ymax_floor)
    if min_span is not None and min_span > 0 and (data_max - data_min) < min_span:
        mid = (data_max + data_min) / 2.0
        half = min_span / 2.0
        data_min = mid - half
        data_max = mid + half
    ax.set_ylim(data_min, data_max)


# *****************************************************************************
def get_settings_path():
    """
    ---------------------------------------------------------------
    get_settings_path()
    Returns the path to the LarynxSettings file (preferred or fallback folder).

    INPUTS
    (none)

    OUTPUTS
    returns: tuple (folder_path_str, file_path_str). Folder is the one to use
             for saving; file_path_str is full path to LarynxSettings file.

   (c) Kai Kuck, 2026
    ---------------------------------------------------------------
    """
    folder = SETTINGS_FOLDER_PREFERRED if os.path.isdir(SETTINGS_FOLDER_PREFERRED) else SETTINGS_FOLDER_FALLBACK
    file_path = os.path.join(folder, SETTINGS_FILENAME)
    return folder, file_path


# *****************************************************************************
def load_settings():
    """
    ---------------------------------------------------------------
    load_settings()
    Loads application settings from LarynxSettings JSON file.

    INPUTS
    (none)

    OUTPUTS
    returns: dict with keys: com_port, baud_rate_bps, plot_time_sec, data_folder.
             If no file exists or folder missing, returns default values.

   (c) Kai Kuck, 2026
    ---------------------------------------------------------------
    """
    _, path = get_settings_path()
    defaults = {
        "com_port": DEFAULT_COM_PORT,
        "baud_rate_bps": DEFAULT_BAUD,
        "plot_time_sec": DEFAULT_PLOT_TIME_SEC,
        "data_folder": DEFAULT_DATA_FOLDER,
        "y_axis_mode": DEFAULT_Y_AXIS_MODE,
        "y_axis_min": DEFAULT_Y_AXIS_MIN,
        "y_axis_max": DEFAULT_Y_AXIS_MAX,
        "primary_y_max_floor": DEFAULT_PRIMARY_Y_MAX_FLOOR,
        "plot_channel_1": DEFAULT_PLOT_CHANNEL_1,
        "plot_channel_2": DEFAULT_PLOT_CHANNEL_2,
        "plot_channel_3": DEFAULT_PLOT_CHANNEL_3,
        "plot_channel_4": DEFAULT_PLOT_CHANNEL_4,
        "data_file_prefix": DEFAULT_DATA_FILE_PREFIX,
        "calibduration__sec": DEFAULT_CALIB_DURATION_SEC,
        "movgavglength__sec": DEFAULT_MOVAVG_LENGTH_SEC,
        "calib_channel": DEFAULT_CALIB_CHANNEL,
    }
    defaults.update(_channel_defaults_for_settings())
    if not os.path.isfile(path):
        return defaults
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        y_mode = data.get("y_axis_mode", defaults["y_axis_mode"])
        if y_mode not in ("auto", "fixed"):
            y_mode = defaults["y_axis_mode"]
        y_min = int(data.get("y_axis_min", defaults["y_axis_min"]))
        y_max = int(data.get("y_axis_max", defaults["y_axis_max"]))
        y_min = max(Y_AXIS_RANGE_MIN, min(Y_AXIS_RANGE_MAX, y_min))
        y_max = max(Y_AXIS_RANGE_MIN, min(Y_AXIS_RANGE_MAX, y_max))
        if y_min >= y_max:
            y_min, y_max = DEFAULT_Y_AXIS_MIN, DEFAULT_Y_AXIS_MAX
        try:
            primary_y_max_floor = float(data.get(
                "primary_y_max_floor", defaults["primary_y_max_floor"]))
        except (TypeError, ValueError):
            primary_y_max_floor = DEFAULT_PRIMARY_Y_MAX_FLOOR
        primary_y_max_floor = max(
            PRIMARY_Y_MAX_FLOOR_MIN,
            min(PRIMARY_Y_MAX_FLOOR_MAX, primary_y_max_floor),
        )
        calibduration__sec = int(data.get("calibduration__sec", defaults["calibduration__sec"]))
        calibduration__sec = max(CALIB_DURATION_MIN_SEC, min(CALIB_DURATION_MAX_SEC, calibduration__sec))
        movgavglength__sec = int(data.get("movgavglength__sec", defaults["movgavglength__sec"]))
        movgavglength__sec = max(MOVAVG_LENGTH_MIN_SEC, min(MOVAVG_LENGTH_MAX_SEC, movgavglength__sec))
        try:
            calib_channel = int(data.get("calib_channel", defaults["calib_channel"]))
        except (TypeError, ValueError):
            calib_channel = DEFAULT_CALIB_CHANNEL
        calib_channel = max(CALIB_CHANNEL_MIN, min(CALIB_CHANNEL_MAX, calib_channel))
        result = {
            "com_port": data.get("com_port", defaults["com_port"]),
            "baud_rate_bps": int(data.get("baud_rate_bps", defaults["baud_rate_bps"])),
            "plot_time_sec": int(data.get("plot_time_sec", defaults["plot_time_sec"])),
            "data_folder": data.get("data_folder", defaults["data_folder"]),
            "y_axis_mode": y_mode,
            "y_axis_min": y_min,
            "y_axis_max": y_max,
            "primary_y_max_floor": primary_y_max_floor,
            "plot_channel_1": bool(data.get("plot_channel_1", defaults["plot_channel_1"])),
            "plot_channel_2": bool(data.get("plot_channel_2", defaults["plot_channel_2"])),
            "plot_channel_3": bool(data.get("plot_channel_3", defaults["plot_channel_3"])),
            "plot_channel_4": bool(data.get("plot_channel_4", defaults["plot_channel_4"])),
            "data_file_prefix": sanitize_windows_filename_prefix(
                data.get("data_file_prefix", defaults["data_file_prefix"])
            ) or DEFAULT_DATA_FILE_PREFIX,
            "calibduration__sec": calibduration__sec,
            "movgavglength__sec": movgavglength__sec,
            "calib_channel": calib_channel,
        }
        for n in range(1, 5):
            try:
                a_val = float(data.get(f"channel_{n}_model_a", defaults[f"channel_{n}_model_a"]))
            except (TypeError, ValueError):
                a_val = DEFAULT_CHANNEL_MODEL_A
            try:
                b_val = float(data.get(f"channel_{n}_model_b", defaults[f"channel_{n}_model_b"]))
            except (TypeError, ValueError):
                b_val = DEFAULT_CHANNEL_MODEL_B
            mode = data.get(f"channel_{n}_display_mode", defaults[f"channel_{n}_display_mode"])
            if mode not in CHANNEL_DISPLAY_MODES:
                mode = DEFAULT_CHANNEL_DISPLAY_MODE
            try:
                tare_val = float(data.get(f"channel_{n}_tare_value", defaults[f"channel_{n}_tare_value"]))
            except (TypeError, ValueError):
                tare_val = DEFAULT_CHANNEL_TARE_VALUE
            try:
                slope_val = float(data.get(f"channel_{n}_slope_factor", defaults[f"channel_{n}_slope_factor"]))
            except (TypeError, ValueError):
                slope_val = DEFAULT_CHANNEL_SLOPE_FACTOR
            try:
                target_mass_val = float(data.get(f"channel_{n}_target_mass", defaults[f"channel_{n}_target_mass"]))
            except (TypeError, ValueError):
                target_mass_val = DEFAULT_CHANNEL_TARGET_MASS
            result[f"channel_{n}_model_a"] = a_val
            result[f"channel_{n}_model_b"] = b_val
            result[f"channel_{n}_tare_value"] = tare_val
            result[f"channel_{n}_slope_factor"] = slope_val
            result[f"channel_{n}_target_mass"] = target_mass_val
            result[f"channel_{n}_display_mode"] = mode
        return result
    except Exception:
        return defaults


# *****************************************************************************
def save_settings(com_port, baud_rate_bps, plot_time_sec, data_folder,
                  y_axis_mode=None, y_axis_min=None, y_axis_max=None,
                  primary_y_max_floor=None,
                  calibduration__sec=None, movgavglength__sec=None, calib_channel=None,
                  plot_channel_1=None, plot_channel_2=None, plot_channel_3=None, plot_channel_4=None,
                  data_file_prefix=None, channel_settings=None):
    """
    ---------------------------------------------------------------
    save_settings(...)
    Writes current settings to LarynxSettings JSON file.

    INPUTS
    com_port .............. COM port string (e.g. 'COM3').
    baud_rate_bps ........ Baud rate in bits per second.
    plot_time_sec ......... Time length of scrolling plot in seconds.
    data_folder ........... Folder path for saving CSV data files.
    y_axis_mode ........... 'auto' or 'fixed' (optional).
    y_axis_min ............. Y-axis minimum when fixed (optional).
    y_axis_max ............. Y-axis maximum when fixed (optional).
    primary_y_max_floor .... Minimum upper limit for auto-scaled mass/force Y-axis (optional).
    plot_channel_1..4 ...... Whether to plot channel 1..4 (optional).
    data_file_prefix ...... Prefix for recorded CSV filename (optional). If None,
                            the existing value in the file is kept (not updated).

    OUTPUTS
    returns: None. Creates/overwrites LarynxSettings in preferred or fallback folder.

   (c) Kai Kuck, 2026
    ---------------------------------------------------------------
    """
    folder, path = get_settings_path()
    if y_axis_mode is None:
        y_axis_mode = DEFAULT_Y_AXIS_MODE
    if y_axis_min is None:
        y_axis_min = DEFAULT_Y_AXIS_MIN
    if y_axis_max is None:
        y_axis_max = DEFAULT_Y_AXIS_MAX
    if primary_y_max_floor is None:
        primary_y_max_floor = DEFAULT_PRIMARY_Y_MAX_FLOOR
    if plot_channel_1 is None:
        plot_channel_1 = DEFAULT_PLOT_CHANNEL_1
    if plot_channel_2 is None:
        plot_channel_2 = DEFAULT_PLOT_CHANNEL_2
    if plot_channel_3 is None:
        plot_channel_3 = DEFAULT_PLOT_CHANNEL_3
    if plot_channel_4 is None:
        plot_channel_4 = DEFAULT_PLOT_CHANNEL_4
    if data_file_prefix is None:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                data_file_prefix = existing.get("data_file_prefix", DEFAULT_DATA_FILE_PREFIX)
            except Exception:
                data_file_prefix = DEFAULT_DATA_FILE_PREFIX
        else:
            data_file_prefix = DEFAULT_DATA_FILE_PREFIX
    else:
        data_file_prefix = sanitize_windows_filename_prefix(data_file_prefix) or DEFAULT_DATA_FILE_PREFIX
    existing_file = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing_file = json.load(f)
        except Exception:
            existing_file = {}
    if calibduration__sec is None:
        calibduration__sec = existing_file.get("calibduration__sec", DEFAULT_CALIB_DURATION_SEC)
    if movgavglength__sec is None:
        movgavglength__sec = existing_file.get("movgavglength__sec", DEFAULT_MOVAVG_LENGTH_SEC)
    if calib_channel is None:
        calib_channel = existing_file.get("calib_channel", DEFAULT_CALIB_CHANNEL)
    try:
        calibduration__sec = int(calibduration__sec)
    except (TypeError, ValueError):
        calibduration__sec = DEFAULT_CALIB_DURATION_SEC
    try:
        movgavglength__sec = int(movgavglength__sec)
    except (TypeError, ValueError):
        movgavglength__sec = DEFAULT_MOVAVG_LENGTH_SEC
    calibduration__sec = max(CALIB_DURATION_MIN_SEC, min(CALIB_DURATION_MAX_SEC, calibduration__sec))
    movgavglength__sec = max(MOVAVG_LENGTH_MIN_SEC, min(MOVAVG_LENGTH_MAX_SEC, movgavglength__sec))
    try:
        calib_channel = int(calib_channel)
    except (TypeError, ValueError):
        calib_channel = DEFAULT_CALIB_CHANNEL
    calib_channel = max(CALIB_CHANNEL_MIN, min(CALIB_CHANNEL_MAX, calib_channel))
    payload = {
        "com_port": com_port,
        "baud_rate_bps": baud_rate_bps,
        "plot_time_sec": plot_time_sec,
        "data_folder": data_folder,
        "y_axis_mode": y_axis_mode,
        "y_axis_min": int(y_axis_min),
        "y_axis_max": int(y_axis_max),
        "primary_y_max_floor": round(max(
            PRIMARY_Y_MAX_FLOOR_MIN,
            min(PRIMARY_Y_MAX_FLOOR_MAX, float(primary_y_max_floor)),
        ), 1),
        "plot_channel_1": bool(plot_channel_1),
        "plot_channel_2": bool(plot_channel_2),
        "plot_channel_3": bool(plot_channel_3),
        "plot_channel_4": bool(plot_channel_4),
        "data_file_prefix": data_file_prefix,
        "calibduration__sec": calibduration__sec,
        "movgavglength__sec": movgavglength__sec,
        "calib_channel": calib_channel,
    }
    if channel_settings is not None:
        for n in range(1, 5):
            try:
                payload[f"channel_{n}_model_a"] = float(channel_settings.get(
                    f"channel_{n}_model_a", DEFAULT_CHANNEL_MODEL_A))
            except (TypeError, ValueError):
                payload[f"channel_{n}_model_a"] = DEFAULT_CHANNEL_MODEL_A
            try:
                payload[f"channel_{n}_model_b"] = float(channel_settings.get(
                    f"channel_{n}_model_b", DEFAULT_CHANNEL_MODEL_B))
            except (TypeError, ValueError):
                payload[f"channel_{n}_model_b"] = DEFAULT_CHANNEL_MODEL_B
            mode = channel_settings.get(f"channel_{n}_display_mode", DEFAULT_CHANNEL_DISPLAY_MODE)
            if mode not in CHANNEL_DISPLAY_MODES:
                mode = DEFAULT_CHANNEL_DISPLAY_MODE
            try:
                payload[f"channel_{n}_tare_value"] = float(channel_settings.get(
                    f"channel_{n}_tare_value", DEFAULT_CHANNEL_TARE_VALUE))
            except (TypeError, ValueError):
                payload[f"channel_{n}_tare_value"] = DEFAULT_CHANNEL_TARE_VALUE
            try:
                payload[f"channel_{n}_slope_factor"] = float(channel_settings.get(
                    f"channel_{n}_slope_factor", DEFAULT_CHANNEL_SLOPE_FACTOR))
            except (TypeError, ValueError):
                payload[f"channel_{n}_slope_factor"] = DEFAULT_CHANNEL_SLOPE_FACTOR
            try:
                payload[f"channel_{n}_target_mass"] = float(channel_settings.get(
                    f"channel_{n}_target_mass", DEFAULT_CHANNEL_TARGET_MASS))
            except (TypeError, ValueError):
                payload[f"channel_{n}_target_mass"] = DEFAULT_CHANNEL_TARGET_MASS
            payload[f"channel_{n}_display_mode"] = mode
    elif os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            for n in range(1, 5):
                for key in (
                    f"channel_{n}_model_a",
                    f"channel_{n}_model_b",
                    f"channel_{n}_tare_value",
                    f"channel_{n}_slope_factor",
                    f"channel_{n}_target_mass",
                    f"channel_{n}_display_mode",
                ):
                    if key in existing:
                        payload[key] = existing[key]
            for key in ("calibduration__sec", "movgavglength__sec", "calib_channel"):
                if key in existing and key not in payload:
                    payload[key] = existing[key]
        except Exception:
            payload.update(_channel_defaults_for_settings())
    else:
        payload.update(_channel_defaults_for_settings())
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# *****************************************************************************
def save_app_settings(settings):
    """Persist full application settings dict to LarynxSettings JSON."""
    channel_settings = {}
    for n in range(1, 5):
        channel_settings[f"channel_{n}_model_a"] = settings.get(
            f"channel_{n}_model_a", DEFAULT_CHANNEL_MODEL_A)
        channel_settings[f"channel_{n}_model_b"] = settings.get(
            f"channel_{n}_model_b", DEFAULT_CHANNEL_MODEL_B)
        channel_settings[f"channel_{n}_tare_value"] = settings.get(
            f"channel_{n}_tare_value", DEFAULT_CHANNEL_TARE_VALUE)
        channel_settings[f"channel_{n}_slope_factor"] = settings.get(
            f"channel_{n}_slope_factor", DEFAULT_CHANNEL_SLOPE_FACTOR)
        channel_settings[f"channel_{n}_target_mass"] = settings.get(
            f"channel_{n}_target_mass", DEFAULT_CHANNEL_TARGET_MASS)
        channel_settings[f"channel_{n}_display_mode"] = settings.get(
            f"channel_{n}_display_mode", DEFAULT_CHANNEL_DISPLAY_MODE)
    save_settings(
        settings.get("com_port", DEFAULT_COM_PORT),
        settings.get("baud_rate_bps", DEFAULT_BAUD),
        settings.get("plot_time_sec", DEFAULT_PLOT_TIME_SEC),
        settings.get("data_folder", DEFAULT_DATA_FOLDER),
        y_axis_mode=settings.get("y_axis_mode", DEFAULT_Y_AXIS_MODE),
        y_axis_min=settings.get("y_axis_min", DEFAULT_Y_AXIS_MIN),
        y_axis_max=settings.get("y_axis_max", DEFAULT_Y_AXIS_MAX),
        primary_y_max_floor=settings.get("primary_y_max_floor", DEFAULT_PRIMARY_Y_MAX_FLOOR),
        calibduration__sec=settings.get("calibduration__sec", DEFAULT_CALIB_DURATION_SEC),
        movgavglength__sec=settings.get("movgavglength__sec", DEFAULT_MOVAVG_LENGTH_SEC),
        calib_channel=settings.get("calib_channel", DEFAULT_CALIB_CHANNEL),
        plot_channel_1=settings.get("plot_channel_1", DEFAULT_PLOT_CHANNEL_1),
        plot_channel_2=settings.get("plot_channel_2", DEFAULT_PLOT_CHANNEL_2),
        plot_channel_3=settings.get("plot_channel_3", DEFAULT_PLOT_CHANNEL_3),
        plot_channel_4=settings.get("plot_channel_4", DEFAULT_PLOT_CHANNEL_4),
        data_file_prefix=settings.get("data_file_prefix", DEFAULT_DATA_FILE_PREFIX),
        channel_settings=channel_settings,
    )


# *****************************************************************************
def ensure_settings_file_exists():
    """
    ---------------------------------------------------------------
    ensure_settings_file_exists()
    If no LarynxSettings file exists, writes default settings and creates file.

    INPUTS
    (none)

    OUTPUTS
    returns: None.

   (c) Kai Kuck, 2026
    ---------------------------------------------------------------
    """
    _, path = get_settings_path()
    if os.path.isfile(path):
        return
    s = load_settings()
    save_settings(
        s["com_port"],
        s["baud_rate_bps"],
        s["plot_time_sec"],
        s["data_folder"],
        s.get("y_axis_mode", DEFAULT_Y_AXIS_MODE),
        s.get("y_axis_min", DEFAULT_Y_AXIS_MIN),
        s.get("y_axis_max", DEFAULT_Y_AXIS_MAX),
        s.get("primary_y_max_floor", DEFAULT_PRIMARY_Y_MAX_FLOOR),
        s.get("plot_channel_1", DEFAULT_PLOT_CHANNEL_1),
        s.get("plot_channel_2", DEFAULT_PLOT_CHANNEL_2),
        s.get("plot_channel_3", DEFAULT_PLOT_CHANNEL_3),
        s.get("plot_channel_4", DEFAULT_PLOT_CHANNEL_4),
    )


# *****************************************************************************
def list_com_ports():
    """
    ---------------------------------------------------------------
    list_com_ports()
    Returns list of available COM port names on Windows.

    INPUTS
    (none)

    OUTPUTS
    returns: list of strings, e.g. ['COM3', 'COM4']. Empty if pyserial not available.

   (c) Kai Kuck, 2026
    ---------------------------------------------------------------
    """
    if serial is None:
        return []
    ports = serial.tools.list_ports.comports()
    return sorted([p.device for p in ports])


# *****************************************************************************
def format_csv_timestamp(dt=None):
    """
    ---------------------------------------------------------------
    format_csv_timestamp(dt=None)
    Formats a datetime for CSV row: DD-MM-YYYY;HH:MM:SS.SSS

    INPUTS
    dt ... datetime to format. If None, uses current time.

    OUTPUTS
    returns: str in form DD-MM-YYYY;HH:MM:SS.SSS

   (c) Kai Kuck, 2026
    ---------------------------------------------------------------
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%d-%m-%Y;%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


# *****************************************************************************
def format_record_row(dt, seconds_since_start, values, settings, events=None, calib_channels=None):
    """
    ---------------------------------------------------------------
    format_record_row(...)
    Formats one CSV row for recording with DAC, mass, force, calibration, and events.

    INPUTS
    dt ...................... datetime for the sample (system date/time).
    seconds_since_start ..... seconds since first value in current file (float).
    values .................. list of 4 A/D values [ch1, ch2, ch3, ch4].
    settings ................ application settings dict (calibration, tare, slope).
    events .................. optional list of 4 event strings per channel.
    calib_channels .......... optional set of channel indices to include calib columns.

    OUTPUTS
    returns: str — one CSV data line (including newline).

   (c) Kai Kuck, 2026
    ---------------------------------------------------------------
    """
    return build_csv_record_row(
        dt, seconds_since_start, values, settings,
        events=events, calib_channels=calib_channels,
    )


# *****************************************************************************
def data_filename_datetime():
    """
    ---------------------------------------------------------------
    data_filename_datetime()
    Returns system date/time for filename: DDMMYYYY_HHMM.

    INPUTS
    (none)

    OUTPUTS
    returns: str, e.g. '24022026_1430'

   (c) Kai Kuck, 2026
    ---------------------------------------------------------------
    """
    return datetime.now().strftime("%d%m%Y_%H%M")


# *****************************************************************************
class SerialReaderThread(threading.Thread):
    """Background thread that reads lines from serial and puts (timestamp, value) into queue."""

    def __init__(self, ser, queue_out, stop_event):
        super().__init__(daemon=True)
        self.ser = ser
        self.queue_out = queue_out
        self.stop_event = stop_event

    def run(self):
        while not self.stop_event.is_set():
            try:
                if self.ser.in_waiting:
                    line = self.ser.readline()
                    if line:
                        try:
                            s = line.decode("ascii", errors="ignore").strip()
                            if s:
                                # Line on wire: ch4;ch3;ch2;ch1 — remap to logical ch1..ch4
                                parts = [p.strip() for p in s.split(";")]
                                if len(parts) >= 1 and any(parts):
                                    values = []
                                    for ch in range(4):
                                        idx = SERIAL_PART_INDEX_FOR_LOGICAL_CH[ch]
                                        try:
                                            values.append(
                                                int(parts[idx]) if idx < len(parts) and parts[idx] else 0
                                            )
                                        except (ValueError, IndexError):
                                            values.append(0)
                                    self.queue_out.put((datetime.now(), values))
                        except (ValueError, UnicodeDecodeError):
                            pass
                else:
                    self.stop_event.wait(0.001)
            except (serial.SerialException, OSError, AttributeError):
                break
            except Exception:
                pass


# *****************************************************************************
class CalibrationDialog(tk.Toplevel):
    """Calibration data collection: raw and moving-averaged DAC for one channel."""

    STATE_WAIT_START = "waiting_to_start"
    STATE_WAIT_FREEZE = "waiting_for_freeze"
    STATE_WAIT_UNFREEZE = "waiting_to_unfreeze"

    def __init__(self, parent, main_app):
        super().__init__(parent)
        self.main_app = main_app
        self.title("Calibration")
        self.geometry("720x560")
        self.minsize(520, 420)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.state = self.STATE_WAIT_START
        self.updates_active = True
        self.fill_with_next_dac = True

        self.calibduration__sec = int(main_app.settings.get(
            "calibduration__sec", DEFAULT_CALIB_DURATION_SEC))
        self.movgavglength__sec = int(main_app.settings.get(
            "movgavglength__sec", DEFAULT_MOVAVG_LENGTH_SEC))
        self.calibcountdown__sec = self.calibduration__sec

        try:
            calib_channel = int(main_app.settings.get(
                "calib_channel", DEFAULT_CALIB_CHANNEL))
        except (TypeError, ValueError):
            calib_channel = DEFAULT_CALIB_CHANNEL
        calib_channel = max(CALIB_CHANNEL_MIN, min(CALIB_CHANNEL_MAX, calib_channel))
        self._channel_index = calib_channel - 1
        self.movgavg_array = []
        self.movgavgDAC = 0.0
        self.current_dac = None
        self.calib_plot_times = []
        self.calib_plot_raw = []
        self.calib_plot_avg = []

        self._tick_after_id = None
        self._countdown_after_id = None
        self._autoscale_after_id = None

        self._build_ui()
        self._enter_wait_start(initial_open=True)
        self._set_calib_y_axis_initial()
        self._schedule_tick()
        self._schedule_autoscale()

    def _set_calib_y_axis_initial(self):
        """Fixed Y range when opening the window or changing channel."""
        self.calib_ax.set_autoscaley_on(False)
        self.calib_ax.set_ylim(CALIB_Y_AXIS_INITIAL_MIN, CALIB_Y_AXIS_INITIAL_MAX)
        _refresh_axis_y_ticks(self.calib_ax, "left")
        self.calib_canvas.draw_idle()

    def _plotting_active(self):
        return self.state in (self.STATE_WAIT_START, self.STATE_WAIT_FREEZE)

    def _build_ui(self):
        f = ttk.Frame(self, padding=10)
        f.pack(fill=tk.BOTH, expand=True)
        row = 0
        ttk.Label(f, text="Channel:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.channel_var = tk.StringVar(value=f"Channel {self._channel_index + 1}")
        self.channel_combo = ttk.Combobox(
            f, textvariable=self.channel_var, state="readonly", width=12,
            values=["Channel 1", "Channel 2", "Channel 3", "Channel 4"],
        )
        self.channel_combo.grid(row=row, column=1, sticky=tk.W, pady=2)
        self.channel_combo.bind("<<ComboboxSelected>>", self._on_channel_change)
        row += 1
        ttk.Label(
            f, text=CALIB_SLIDER_LEFT_LABELS[0],
            width=CALIB_SLIDER_LABEL_WIDTH, anchor=tk.E,
        ).grid(row=row, column=0, sticky=tk.E, padx=(0, CALIB_SLIDER_GAP), pady=2)
        self.calibduration_var = tk.IntVar(value=self.calibduration__sec)
        self.calibduration_scale = ttk.Scale(
            f, from_=CALIB_DURATION_MIN_SEC, to=CALIB_DURATION_MAX_SEC,
            variable=self.calibduration_var, orient=tk.HORIZONTAL, length=220,
            command=self._on_calibduration_scale,
        )
        self.calibduration_scale.grid(
            row=row, column=1, sticky=tk.W, padx=(0, CALIB_SLIDER_GAP), pady=2)
        self.calibduration_label = ttk.Label(
            f, text=f"{self.calibduration__sec} sec")
        self.calibduration_label.grid(row=row, column=2, sticky=tk.W, pady=2)
        row += 1
        ttk.Label(
            f, text=CALIB_SLIDER_LEFT_LABELS[1],
            width=CALIB_SLIDER_LABEL_WIDTH, anchor=tk.E,
        ).grid(row=row, column=0, sticky=tk.E, padx=(0, CALIB_SLIDER_GAP), pady=2)
        self.movgavglength_var = tk.IntVar(value=self.movgavglength__sec)
        self.movgavglength_scale = ttk.Scale(
            f, from_=MOVAVG_LENGTH_MIN_SEC, to=MOVAVG_LENGTH_MAX_SEC,
            variable=self.movgavglength_var, orient=tk.HORIZONTAL, length=220,
            command=self._on_movgavglength_scale,
        )
        self.movgavglength_scale.grid(
            row=row, column=1, sticky=tk.W, padx=(0, CALIB_SLIDER_GAP), pady=2)
        self.movgavg_label = ttk.Label(
            f, text=f"Avg over {self.movgavglength__sec} sec")
        self.movgavg_label.grid(row=row, column=2, sticky=tk.W, pady=2)
        row += 1
        values_frame = ttk.Frame(f)
        values_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(8, 4))
        ch_num = self._channel_index + 1
        self.channel_title_label = ttk.Label(values_frame, text=f"Channel {ch_num}:")
        self.channel_title_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 4))
        self.channel_dac_var = tk.StringVar(value="—")
        self.channel_dac_label = tk.Label(
            values_frame, textvariable=self.channel_dac_var,
            font=CALIB_VALUE_FONT, fg=CALIB_DAC_COLOR,
            width=CALIB_DAC_VALUE_WIDTH, anchor=tk.E,
        )
        self.channel_dac_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 32))
        self.mavg_title_label = ttk.Label(values_frame, text="Moving Average:")
        self.mavg_title_label.grid(row=0, column=2, sticky=tk.W, padx=(0, 4))
        self.mavg_dac_var = tk.StringVar(value="—")
        self.mavg_dac_label = tk.Label(
            values_frame, textvariable=self.mavg_dac_var,
            font=CALIB_VALUE_FONT, fg=CALIB_MOVAVG_COLOR,
            width=CALIB_DAC_VALUE_WIDTH, anchor=tk.E,
        )
        self.mavg_dac_label.grid(row=0, column=3, sticky=tk.W)
        row += 1
        for col in range(3):
            f.columnconfigure(col, weight=1)
        plot_frame = ttk.Frame(f)
        plot_frame.grid(row=row, column=0, columnspan=3, sticky=tk.NSEW, pady=8)
        f.rowconfigure(row, weight=1)
        self.calib_fig = Figure(figsize=(5.5, 3.2), dpi=100)
        self.calib_ax = self.calib_fig.add_subplot(111)
        ch_label = f"Channel {ch_num}"
        self.calib_line_raw, = self.calib_ax.plot(
            [], [], color=CALIB_DAC_COLOR, linewidth=1.0, label=ch_label)
        self.calib_line_avg, = self.calib_ax.plot(
            [], [], color=CALIB_MOVAVG_COLOR, linewidth=1.5, label="Moving avg")
        self.calib_ax.set_xlabel("Time (s)")
        self.calib_ax.set_ylabel(ch_label)
        self.calib_ax.set_autoscaley_on(False)
        self.calib_ax.grid(True, alpha=0.3)
        self.calib_ax.legend(loc="upper right", fontsize=8)
        self.calib_canvas = FigureCanvasTkAgg(self.calib_fig, master=plot_frame)
        self.calib_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._layout_calib_figure()
        row += 1
        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=8)
        self.state_button = ttk.Button(btn_frame, text="Start", command=self._on_state_button)
        self.state_button.pack(side=tk.LEFT)
        self.state_hint_var = tk.StringVar()
        self.state_hint_label = ttk.Label(btn_frame, textvariable=self.state_hint_var)
        self.state_hint_label.pack(side=tk.LEFT, padx=(12, 0))
        self._update_state_hint()

    def _layout_calib_figure(self):
        """Center plot area in the figure and leave room for the Y-axis label."""
        self.calib_fig.subplots_adjust(
            left=0.14, right=0.88, bottom=0.16, top=0.88)
        try:
            self.calib_canvas.draw_idle()
        except (tk.TclError, AttributeError):
            pass

    def _movavg_buffer_size(self):
        sec = int(self.movgavglength__sec)
        if sec <= 0:
            return 1
        return max(1, int(sec * SAMPLES_PER_SEC))

    def _fill_movavg_array_with_value(self, value):
        size = self._movavg_buffer_size()
        self.movgavg_array = [float(value)] * size
        self.movgavgDAC = float(value)

    def _fill_movavg_array_with_last_avg(self):
        val = self.movgavgDAC if self.movgavg_array else 0.0
        self._fill_movavg_array_with_value(val)

    def _read_channel_dac(self):
        if self.main_app is None:
            return None
        raw = self.main_app.last_received_values[self._channel_index]
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def _process_dac_sample(self, dac):
        if self.fill_with_next_dac:
            self._fill_movavg_array_with_value(dac)
            self.fill_with_next_dac = False
        else:
            self.movgavg_array.append(float(dac))
            max_len = self._movavg_buffer_size()
            if len(self.movgavg_array) > max_len:
                self.movgavg_array = self.movgavg_array[-max_len:]
            self.movgavgDAC = sum(self.movgavg_array) / len(self.movgavg_array)

    def _trim_plot_buffers(self):
        if not self.calib_plot_times:
            return
        t_max = self.calib_plot_times[-1]
        t_cutoff = t_max - float(self.calibduration__sec)
        keep = [i for i, t in enumerate(self.calib_plot_times) if t >= t_cutoff]
        if keep:
            self.calib_plot_times = [self.calib_plot_times[i] for i in keep]
            self.calib_plot_raw = [self.calib_plot_raw[i] for i in keep]
            self.calib_plot_avg = [self.calib_plot_avg[i] for i in keep]

    def _update_calib_channel_labels(self):
        """Legend and Y-axis title use selected channel number."""
        ch_num = self._channel_index + 1
        ch_label = f"Channel {ch_num}"
        self.channel_title_label.config(text=f"{ch_label}:")
        self.calib_line_raw.set_label(ch_label)
        self.calib_ax.set_ylabel(ch_label)
        leg = self.calib_ax.get_legend()
        if leg is not None:
            leg.remove()
        self.calib_ax.legend(loc="upper right", fontsize=8)
        self._layout_calib_figure()

    def _update_state_hint(self):
        if self.state == self.STATE_WAIT_START:
            text = "Apply known mass and press Start"
        elif self.state == self.STATE_WAIT_FREEZE:
            text = (
                f"Wait {self.calibcountdown__sec}sec for values to stabilize")
        else:
            text = "Press Unfreeze to resume"
        self.state_hint_var.set(text)

    def _update_labels(self):
        ch_num = self._channel_index + 1
        self.channel_title_label.config(text=f"Channel {ch_num}:")
        if self.current_dac is not None:
            self.channel_dac_var.set(f"{int(round(self.current_dac))}")
            self.mavg_dac_var.set(f"{int(round(self.movgavgDAC))}")
        else:
            self.channel_dac_var.set("—")
            self.mavg_dac_var.set("—")
        self._update_state_hint()

    def _redraw_calib_plot(self):
        if not self.calib_plot_times:
            self.calib_line_raw.set_data([], [])
            self.calib_line_avg.set_data([], [])
            self.calib_ax.set_xlim(0, float(self.calibduration__sec))
            self._layout_calib_figure()
            self.calib_canvas.draw_idle()
            return
        t0 = self.calib_plot_times[0]
        time_sec = [t - t0 for t in self.calib_plot_times]
        self.calib_line_raw.set_data(time_sec, self.calib_plot_raw)
        self.calib_line_avg.set_data(time_sec, self.calib_plot_avg)
        self.calib_ax.set_xlim(0, float(self.calibduration__sec))
        self._layout_calib_figure()
        self.calib_canvas.draw_idle()

    def _autoscale_calib_plot(self):
        if not self.winfo_exists():
            return
        if self._plotting_active():
            if self.calib_plot_times:
                vals = list(self.calib_plot_raw) + list(self.calib_plot_avg)
                if vals:
                    _autoscale_ylim(
                        self.calib_ax, vals, min_span=CALIB_Y_AXIS_MIN_SPAN)
                    _refresh_axis_y_ticks(self.calib_ax, "left")
            else:
                self._set_calib_y_axis_initial()
            self._layout_calib_figure()
            self.calib_canvas.draw_idle()
        self._schedule_autoscale()

    def _schedule_autoscale(self):
        self._cancel_autoscale()
        if self.winfo_exists() and self._plotting_active():
            self._autoscale_after_id = self.after(1000, self._autoscale_calib_plot)

    def _cancel_autoscale(self):
        if self._autoscale_after_id is not None:
            try:
                self.after_cancel(self._autoscale_after_id)
            except tk.TclError:
                pass
            self._autoscale_after_id = None

    def _schedule_tick(self):
        self._cancel_tick()
        if self.winfo_exists():
            self._tick_after_id = self.after(50, self._calib_tick)

    def _cancel_tick(self):
        if self._tick_after_id is not None:
            try:
                self.after_cancel(self._tick_after_id)
            except tk.TclError:
                pass
            self._tick_after_id = None

    def _calib_tick(self):
        if not self.winfo_exists():
            return
        if self.updates_active:
            dac = self._read_channel_dac()
            if dac is not None:
                self.current_dac = dac
                self._process_dac_sample(dac)
                self.calib_plot_times.append(datetime.now().timestamp())
                self.calib_plot_raw.append(dac)
                self.calib_plot_avg.append(self.movgavgDAC)
                self._trim_plot_buffers()
                self._update_labels()
                self._redraw_calib_plot()
        self._schedule_tick()

    def _cancel_countdown(self):
        if self._countdown_after_id is not None:
            try:
                self.after_cancel(self._countdown_after_id)
            except tk.TclError:
                pass
            self._countdown_after_id = None

    def _schedule_countdown(self):
        self._cancel_countdown()
        if self.winfo_exists() and self.state == self.STATE_WAIT_FREEZE:
            self._countdown_after_id = self.after(1000, self._countdown_tick)

    def _countdown_tick(self):
        if not self.winfo_exists() or self.state != self.STATE_WAIT_FREEZE:
            return
        self.calibcountdown__sec -= 1
        self._update_state_hint()
        if self.calibcountdown__sec <= 0:
            self.calibcountdown__sec = 0
            self._enter_wait_unfreeze()
        else:
            self._schedule_countdown()

    def _enter_wait_start(self, initial_open=False, from_unfreeze=False):
        self.state = self.STATE_WAIT_START
        self.updates_active = True
        self.state_button.config(text="Start")
        self.calibcountdown__sec = self.calibduration__sec
        self._cancel_countdown()
        if initial_open:
            self.fill_with_next_dac = True
        elif from_unfreeze:
            self._fill_movavg_array_with_last_avg()
        self._update_labels()
        self._update_state_hint()
        if not self.calib_plot_times:
            self._set_calib_y_axis_initial()
        self._schedule_autoscale()

    def _clear_calib_plot_buffers(self):
        """Remove plotted history so the next trace starts at the left edge."""
        self.calib_plot_times = []
        self.calib_plot_raw = []
        self.calib_plot_avg = []

    def _set_calib_y_axis_around_dac(self, dac):
        """Y-axis span of current DAC ± 100."""
        self.calib_ax.set_autoscaley_on(False)
        try:
            center = float(dac)
        except (TypeError, ValueError):
            center = 0.0
        self.calib_ax.set_ylim(center - 100, center + 100)
        _refresh_axis_y_ticks(self.calib_ax, "left")
        self.calib_canvas.draw_idle()

    def _enter_wait_freeze(self):
        self.state = self.STATE_WAIT_FREEZE
        self.state_button.config(text="Freeze")
        self.calibcountdown__sec = self.calibduration__sec
        dac = self._read_channel_dac()
        if dac is None and self.current_dac is not None:
            dac = self.current_dac
        self._clear_calib_plot_buffers()
        self.fill_with_next_dac = True
        self._redraw_calib_plot()
        if dac is not None:
            self._set_calib_y_axis_around_dac(dac)
        self._update_labels()
        self._update_state_hint()
        self._schedule_countdown()
        self._schedule_autoscale()

    def _enter_wait_unfreeze(self):
        self.state = self.STATE_WAIT_UNFREEZE
        self.state_button.config(text="Unfreeze")
        self.updates_active = False
        self._cancel_countdown()
        self._cancel_autoscale()
        self._update_labels()
        self._update_state_hint()

    def _on_state_button(self):
        if self.state == self.STATE_WAIT_START:
            self._enter_wait_freeze()
        elif self.state == self.STATE_WAIT_FREEZE:
            self._enter_wait_unfreeze()
        elif self.state == self.STATE_WAIT_UNFREEZE:
            self._enter_wait_start(from_unfreeze=True)

    def _on_channel_change(self, event=None):
        try:
            label = self.channel_var.get()
            self._channel_index = int(label.split()[-1]) - 1
        except (ValueError, IndexError, tk.TclError):
            self._channel_index = 0
        self.calib_plot_times = []
        self.calib_plot_raw = []
        self.calib_plot_avg = []
        self.fill_with_next_dac = True
        self._enter_wait_start(initial_open=True)
        self._update_calib_channel_labels()
        self._set_calib_y_axis_initial()
        self._persist_calibration_settings()
        self._redraw_calib_plot()

    def _on_calibduration_scale(self, v):
        n = max(CALIB_DURATION_MIN_SEC, min(CALIB_DURATION_MAX_SEC, round(float(v))))
        self.calibduration_var.set(n)
        self.calibduration__sec = n
        self.calibduration_label.config(text=f"{n} sec")
        if self.state == self.STATE_WAIT_START:
            self.calibcountdown__sec = n
        self._update_state_hint()
        self._trim_plot_buffers()
        self._persist_calibration_settings()
        self._redraw_calib_plot()

    def _on_movgavglength_scale(self, v):
        n = max(MOVAVG_LENGTH_MIN_SEC, min(MOVAVG_LENGTH_MAX_SEC, round(float(v))))
        self.movgavglength_var.set(n)
        self.movgavglength__sec = n
        self.movgavg_label.config(text=f"Avg over {n} sec")
        self._fill_movavg_array_with_last_avg()
        self._persist_calibration_settings()

    def _persist_calibration_settings(self):
        self.main_app.settings["calibduration__sec"] = self.calibduration__sec
        self.main_app.settings["movgavglength__sec"] = self.movgavglength__sec
        self.main_app.settings["calib_channel"] = self._channel_index + 1
        try:
            save_app_settings(self.main_app.settings)
        except Exception:
            pass

    def _on_close(self):
        self._cancel_tick()
        self._cancel_countdown()
        self._cancel_autoscale()
        self.main_app.calib_dialog = None
        self.destroy()


# *****************************************************************************
class PreferencesDialog(tk.Toplevel):
    """
    ---------------------------------------------------------------
    PreferencesDialog (class)
    Toplevel window for COM port, baud, plot time, and data folder.
    Saves to LarynxSettings on any change.

    INPUTS
    parent ............... Parent tk window.
    current_settings ..... dict with com_port, baud_rate_bps, plot_time_sec, data_folder.
    on_save_callback ..... Optional callable(new_settings_dict) when settings are saved.

    OUTPUTS
    (GUI only)

   (c) Kai Kuck, 2026
    ---------------------------------------------------------------
    """
    def __init__(self, parent, current_settings, on_save_callback=None, main_app=None):
        super().__init__(parent)
        self.current_settings = dict(current_settings)
        self.on_save_callback = on_save_callback
        self.main_app = main_app
        self.title("Preferences")
        self._build_ui()
        self._populate()
        self._fit_preferences_window_size()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _fit_preferences_window_size(self):
        """Size window to content without large empty margins."""
        self.update_idletasks()
        frame_pad = 20  # ttk.Frame padding=10 left + right
        cal_w = getattr(self, "cal_table", None)
        if cal_w is not None:
            cal_w = cal_w.winfo_reqwidth()
        else:
            cal_w = 0
        req_w = max(PREF_MIN_WIDTH, cal_w + frame_pad, self.winfo_reqwidth())
        req_h = self.winfo_reqheight()
        self.minsize(req_w, req_h)
        self.geometry(f"{req_w}x{req_h}")

    def _refresh_folder_entry_view(self, *_):
        """Keep folder path visually right-aligned (show end of long paths)."""
        try:
            self.folder_entry.xview_moveto(1)
        except tk.TclError:
            pass

    def _build_ui(self):
        f = ttk.Frame(self, padding=10)
        f.pack(fill=tk.X, anchor=tk.NW)
        row = 0
        ttk.Label(f, text="COM port:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.com_var = tk.StringVar()
        self.com_combo = ttk.Combobox(f, textvariable=self.com_var, width=20, state="readonly")
        self.com_combo.grid(row=row, column=1, sticky=tk.W, pady=2)
        ports = list_com_ports()
        if not ports:
            ports = ["COM1", "COM2", "COM3", "COM4", "COM5"]
        self.com_combo["values"] = ports
        row += 1
        ttk.Label(f, text="Baud rate (bps):").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.baud_var = tk.IntVar()
        self.baud_combo = ttk.Combobox(
            f, values=[str(b) for b in BAUD_OPTIONS], width=20, state="readonly"
        )
        self.baud_combo.grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        ttk.Label(f, text="Plot time:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.plot_time_var = tk.IntVar()
        def on_plot_scale(v):
            n = max(PLOT_TIME_MIN_SEC, min(PLOT_TIME_MAX_SEC, round(float(v))))
            self.plot_time_var.set(n)
            self.plot_label.config(text=f"{n} sec")

        plot_row = ttk.Frame(f)
        plot_row.grid(row=row, column=1, sticky=tk.W, pady=2)
        self.plot_scale = ttk.Scale(
            plot_row, from_=PLOT_TIME_MIN_SEC, to=PLOT_TIME_MAX_SEC,
            variable=self.plot_time_var, orient=tk.HORIZONTAL, length=200,
            command=on_plot_scale
        )
        self.plot_scale.pack(side=tk.LEFT)
        self.plot_label = ttk.Label(plot_row, text="")
        self.plot_label.pack(side=tk.LEFT, padx=(PREF_VALUE_LABEL_GAP, 0))
        row += 1
        ttk.Label(f, text="Data folder:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.folder_var = tk.StringVar()
        folder_row = ttk.Frame(f)
        folder_row.grid(row=row, column=1, sticky=tk.W, pady=2)
        self.folder_entry = tk.Entry(
            folder_row, textvariable=self.folder_var, width=50, justify="right")
        self.folder_entry.pack(side=tk.LEFT)
        self.folder_entry.bind("<FocusIn>", self._refresh_folder_entry_view)
        self.folder_entry.bind("<KeyRelease>", self._refresh_folder_entry_view)
        ttk.Button(folder_row, text="Browse...", command=self._browse_folder).pack(
            side=tk.LEFT, padx=(PREF_VALUE_LABEL_GAP, 0))
        row += 1
        ttk.Label(f, text="Data filename prefix:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.prefix_var = tk.StringVar()
        prefix_row = ttk.Frame(f)
        prefix_row.grid(row=row, column=1, sticky=tk.W, pady=2)
        self.prefix_entry = ttk.Entry(prefix_row, textvariable=self.prefix_var, width=32)
        self.prefix_entry.pack(side=tk.LEFT)
        ttk.Label(prefix_row, text="(≤30 chars, no \\ / : * ? \" < > |)").pack(
            side=tk.LEFT, padx=(PREF_VALUE_LABEL_GAP, 0))
        self.prefix_entry.bind("<KeyRelease>", self._on_prefix_key)
        row += 1
        ttk.Label(f, text="Y-axis:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.y_axis_mode_var = tk.StringVar(value="auto")
        y_frame = ttk.Frame(f)
        y_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=2)
        ttk.Radiobutton(y_frame, text="Auto-scaled", variable=self.y_axis_mode_var, value="auto",
                        command=self._on_y_axis_mode_change).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Radiobutton(y_frame, text="Fixed", variable=self.y_axis_mode_var, value="fixed",
                        command=self._on_y_axis_mode_change).pack(side=tk.LEFT)
        row += 1
        ttk.Label(f, text="Y min:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.y_axis_min_var = tk.IntVar(value=DEFAULT_Y_AXIS_MIN)
        y_min_row = ttk.Frame(f)
        y_min_row.grid(row=row, column=1, sticky=tk.W, pady=2)
        self.y_axis_min_scale = ttk.Scale(
            y_min_row, from_=Y_AXIS_RANGE_MIN, to=Y_AXIS_RANGE_MAX,
            variable=self.y_axis_min_var, orient=tk.HORIZONTAL, length=200,
            command=self._on_y_axis_min_scale
        )
        self.y_axis_min_scale.pack(side=tk.LEFT)
        self.y_axis_min_label = ttk.Label(y_min_row, text="")
        self.y_axis_min_label.pack(side=tk.LEFT, padx=(PREF_VALUE_LABEL_GAP, 0))
        row += 1
        ttk.Label(f, text="Y max:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.y_axis_max_var = tk.IntVar(value=DEFAULT_Y_AXIS_MAX)
        y_max_row = ttk.Frame(f)
        y_max_row.grid(row=row, column=1, sticky=tk.W, pady=2)
        self.y_axis_max_scale = ttk.Scale(
            y_max_row, from_=Y_AXIS_RANGE_MIN, to=Y_AXIS_RANGE_MAX,
            variable=self.y_axis_max_var, orient=tk.HORIZONTAL, length=200,
            command=self._on_y_axis_max_scale
        )
        self.y_axis_max_scale.pack(side=tk.LEFT)
        self.y_axis_max_label = ttk.Label(y_max_row, text="")
        self.y_axis_max_label.pack(side=tk.LEFT, padx=(PREF_VALUE_LABEL_GAP, 0))
        row += 1
        ttk.Label(
            f, text="Min for Left Y axis upper value (mass, force):",
        ).grid(row=row, column=0, sticky=tk.W, pady=2)
        self.primary_y_max_floor_var = tk.DoubleVar(value=DEFAULT_PRIMARY_Y_MAX_FLOOR)
        primary_y_row = ttk.Frame(f)
        primary_y_row.grid(row=row, column=1, sticky=tk.W, pady=2)
        self.primary_y_max_floor_scale = ttk.Scale(
            primary_y_row, from_=PRIMARY_Y_MAX_FLOOR_MIN, to=PRIMARY_Y_MAX_FLOOR_MAX,
            variable=self.primary_y_max_floor_var, orient=tk.HORIZONTAL, length=200,
            command=self._on_primary_y_max_floor_scale
        )
        self.primary_y_max_floor_scale.pack(side=tk.LEFT)
        self.primary_y_max_floor_label = ttk.Label(primary_y_row, text="")
        self.primary_y_max_floor_label.pack(side=tk.LEFT, padx=(PREF_VALUE_LABEL_GAP, 0))
        row += 1
        ttk.Label(f, text="Plot channels:").grid(row=row, column=0, sticky=tk.W, pady=2)
        ch_frame = ttk.Frame(f)
        ch_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=2)
        self.plot_ch1_var = tk.BooleanVar(value=DEFAULT_PLOT_CHANNEL_1)
        self.plot_ch2_var = tk.BooleanVar(value=DEFAULT_PLOT_CHANNEL_2)
        self.plot_ch3_var = tk.BooleanVar(value=DEFAULT_PLOT_CHANNEL_3)
        self.plot_ch4_var = tk.BooleanVar(value=DEFAULT_PLOT_CHANNEL_4)
        ttk.Checkbutton(ch_frame, text="Channel 1", variable=self.plot_ch1_var, command=self._save_now).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(ch_frame, text="Channel 2", variable=self.plot_ch2_var, command=self._save_now).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(ch_frame, text="Channel 3", variable=self.plot_ch3_var, command=self._save_now).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(ch_frame, text="Channel 4", variable=self.plot_ch4_var, command=self._save_now).pack(side=tk.LEFT)
        row += 1
        ttk.Label(f, text="DAC → mass model:").grid(row=row, column=0, sticky=tk.NW, pady=(10, 2))
        model_block = ttk.Frame(f)
        model_block.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=(10, 2))
        ttk.Label(model_block, text=MODEL_FORMULA_TEXT, wraplength=360).pack(anchor=tk.W)
        row += 1
        cal_table = ttk.Frame(f)
        cal_table.grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))
        self.cal_table = cal_table
        ttk.Label(cal_table, text="", width=6).grid(row=0, column=0, padx=(0, 4))
        ttk.Label(cal_table, text="a", anchor=tk.CENTER).grid(
            row=0, column=1, padx=4, sticky=tk.EW)
        ttk.Label(cal_table, text="b", anchor=tk.CENTER).grid(
            row=0, column=2, padx=4, sticky=tk.EW)
        ttk.Label(cal_table, text="Display as", anchor=tk.CENTER).grid(
            row=0, column=3, padx=(8, 0), sticky=tk.EW)
        row += 1
        self.ch_a_vars = []
        self.ch_b_vars = []
        self.ch_mode_combos = []
        self.ch_zero_buttons = []
        self.ch_target_vars = []
        self.ch_target_entries = []
        self.ch_target_g_labels = []
        self.ch_number_buttons = []
        self.ch_slope_labels = []
        mode_labels = [CHANNEL_DISPLAY_MODE_LABELS[m] for m in CHANNEL_DISPLAY_MODES]
        for i in range(4):
            grid_row = i + 1
            row_pady = (2, 4) if i == 3 else 2
            ttk.Label(cal_table, text=f"Ch{i + 1}", width=6).grid(
                row=grid_row, column=0, padx=(0, 4), pady=row_pady)
            a_var = tk.StringVar(value=str(DEFAULT_CHANNEL_MODEL_A))
            b_var = tk.StringVar(value=str(DEFAULT_CHANNEL_MODEL_B))
            self.ch_a_vars.append(a_var)
            self.ch_b_vars.append(b_var)
            ttk.Entry(cal_table, textvariable=a_var, width=10).grid(
                row=grid_row, column=1, padx=4, pady=row_pady)
            ttk.Entry(cal_table, textvariable=b_var, width=10).grid(
                row=grid_row, column=2, padx=4, pady=row_pady)
            mode_combo = ttk.Combobox(
                cal_table, values=mode_labels, width=14, state="readonly"
            )
            mode_combo.grid(row=grid_row, column=3, padx=(8, 0), pady=row_pady)
            mode_combo.bind("<<ComboboxSelected>>", lambda e, idx=i: self._on_channel_mode_change(idx))
            self.ch_mode_combos.append(mode_combo)
            zero_btn = ttk.Button(
                cal_table, text="Zero", width=6, command=lambda idx=i: self._zero_channel(idx))
            zero_btn.grid(row=grid_row, column=4, padx=(6, 0), pady=row_pady)
            self.ch_zero_buttons.append(zero_btn)
            target_var = tk.StringVar(value=str(DEFAULT_CHANNEL_TARGET_MASS))
            self.ch_target_vars.append(target_var)
            target_entry = ttk.Entry(cal_table, textvariable=target_var, width=8)
            target_entry.grid(row=grid_row, column=5, padx=(6, 2), pady=row_pady)
            self.ch_target_entries.append(target_entry)
            g_lbl = ttk.Label(cal_table, text="g")
            g_lbl.grid(row=grid_row, column=6, padx=(0, 4), pady=row_pady)
            self.ch_target_g_labels.append(g_lbl)
            number_btn = ttk.Button(
                cal_table, text=SCALE_APPLY_BUTTON_TEXT,
                command=lambda idx=i: self._apply_scale_channel(idx)
            )
            number_btn.grid(row=grid_row, column=7, padx=(0, 6), pady=row_pady)
            self.ch_number_buttons.append(number_btn)
            slope_lbl = ttk.Label(
                cal_table, text="slope: 1.0000", width=PREF_SLOPE_LABEL_WIDTH)
            slope_lbl.grid(row=grid_row, column=8, padx=(0, 0), pady=row_pady, sticky=tk.W)
            self.ch_slope_labels.append(slope_lbl)
            target_var.trace_add("write", self._save_now)
            a_var.trace_add("write", self._save_now)
            b_var.trace_add("write", self._save_now)
        row += 1
        self._bind_save_on_change()

    def _channel_mode_from_ui(self, ch_index):
        label = self.ch_mode_combos[ch_index].get().strip()
        mode = DISPLAY_MODE_LABEL_TO_KEY.get(label, DEFAULT_CHANNEL_DISPLAY_MODE)
        if mode not in CHANNEL_DISPLAY_MODES:
            mode = DEFAULT_CHANNEL_DISPLAY_MODE
        return mode

    def _channel_zero_applied(self, ch_index):
        """Session-only flag: True after Zero was pressed this run."""
        if self.main_app is None:
            return False
        return bool(self.main_app.channel_zero_applied[ch_index])

    def _update_slope_factor_label(self, ch_index):
        try:
            if not self._channel_zero_applied(ch_index):
                slope = DEFAULT_CHANNEL_SLOPE_FACTOR
            else:
                slope = float(self.current_settings.get(
                    f"channel_{ch_index + 1}_slope_factor", DEFAULT_CHANNEL_SLOPE_FACTOR
                ))
            self.ch_slope_labels[ch_index].config(text=f"slope: {slope:.4f}")
        except (ValueError, tk.TclError, AttributeError, TypeError):
            pass

    def _update_channel_calibration_ui(self, ch_index):
        mode = self._channel_mode_from_ui(ch_index)
        zero_applied = self._channel_zero_applied(ch_index)
        show_zero = mode in ("grams", "newtons")
        show_scale = show_zero and zero_applied
        zero_btn = self.ch_zero_buttons[ch_index]
        if show_zero:
            zero_btn.grid()
        else:
            zero_btn.grid_remove()
        for widget in (
            self.ch_target_entries[ch_index],
            self.ch_target_g_labels[ch_index],
            self.ch_number_buttons[ch_index],
            self.ch_slope_labels[ch_index],
        ):
            if show_scale:
                widget.grid()
            else:
                widget.grid_remove()
        if show_scale:
            self._update_slope_factor_label(ch_index)

    def _on_channel_mode_change(self, ch_index):
        self._update_channel_calibration_ui(ch_index)
        self._fit_preferences_window_size()
        self._save_now()

    def _zero_channel(self, ch_index):
        """Set tare to current mass_raw for this channel (live reading)."""
        if self.main_app is None:
            return
        raw = self.main_app.last_received_values[ch_index]
        if raw is None:
            messagebox.showwarning(
                "Zero",
                f"No live data on channel {ch_index + 1}. Connect and wait for a reading.",
                parent=self,
            )
            return
        try:
            a_val = float(self.ch_a_vars[ch_index].get().strip())
            b_val = float(self.ch_b_vars[ch_index].get().strip())
        except (ValueError, tk.TclError):
            messagebox.showwarning("Zero", f"Invalid model constants for channel {ch_index + 1}.", parent=self)
            return
        mass_raw = dac_to_mass_grams(raw, a_val, b_val)
        if mass_raw is None:
            messagebox.showwarning(
                "Zero",
                f"Cannot compute mass from current reading on channel {ch_index + 1}.",
                parent=self,
            )
            return
        ch_key = ch_index + 1
        self.current_settings[f"channel_{ch_key}_tare_value"] = mass_raw
        self.current_settings[f"channel_{ch_key}_slope_factor"] = DEFAULT_CHANNEL_SLOPE_FACTOR
        if self.main_app is not None:
            self.main_app.channel_zero_applied[ch_index] = True
            self.main_app.settings[f"channel_{ch_key}_slope_factor"] = DEFAULT_CHANNEL_SLOPE_FACTOR
        self._update_channel_calibration_ui(ch_index)
        self._fit_preferences_window_size()
        self._save_now()
        if self.main_app is not None and self.main_app.recording:
            self.main_app.record_csv_calibration_event(ch_index, "#Zero")

    def _apply_scale_channel(self, ch_index):
        """Set slopefactor = target_mass / mass_zeroed at button press."""
        if self.main_app is None:
            return
        try:
            target_mass = float(self.ch_target_vars[ch_index].get().strip())
        except (ValueError, tk.TclError):
            messagebox.showwarning(
                "Scale",
                f"Enter a valid target mass for channel {ch_index + 1}.",
                parent=self,
            )
            return
        raw = self.main_app.last_received_values[ch_index]
        if raw is None:
            messagebox.showwarning(
                "Scale",
                f"No live data on channel {ch_index + 1}. Connect and wait for a reading.",
                parent=self,
            )
            return
        try:
            a_val = float(self.ch_a_vars[ch_index].get().strip())
            b_val = float(self.ch_b_vars[ch_index].get().strip())
        except (ValueError, tk.TclError):
            messagebox.showwarning("Scale", f"Invalid model constants for channel {ch_index + 1}.", parent=self)
            return
        mass_zeroed_pressed = mass_zeroed_from_dac(
            raw, self.current_settings, ch_index, a=a_val, b=b_val
        )
        if mass_zeroed_pressed is None:
            messagebox.showwarning(
                "Scale",
                f"Cannot compute zeroed mass on channel {ch_index + 1}.",
                parent=self,
            )
            return
        if mass_zeroed_pressed == 0:
            messagebox.showwarning(
                "Scale",
                f"Zeroed mass is 0 on channel {ch_index + 1}; cannot compute scale factor.",
                parent=self,
            )
            return
        slopefactor = target_mass / mass_zeroed_pressed
        ch_key = ch_index + 1
        self.current_settings[f"channel_{ch_key}_slope_factor"] = slopefactor
        self.current_settings[f"channel_{ch_key}_target_mass"] = target_mass
        if self.main_app is not None:
            self.main_app.settings[f"channel_{ch_key}_slope_factor"] = slopefactor
        self._update_slope_factor_label(ch_index)
        self._save_now()
        if self.main_app is not None and self.main_app.recording:
            self.main_app.record_csv_calibration_event(ch_index, "#Slope")

    def _on_prefix_key(self, event=None):
        """Restrict prefix to Windows-safe filename chars and max length."""
        try:
            raw = self.prefix_var.get()
            sanitized = sanitize_windows_filename_prefix(raw)
            if raw != sanitized:
                self.prefix_var.set(sanitized)
                self.prefix_entry.icursor(tk.END)
        except tk.TclError:
            pass

    def _on_y_axis_mode_change(self):
        is_fixed = self.y_axis_mode_var.get() == "fixed"
        state = "normal" if is_fixed else "disabled"
        self.y_axis_min_scale.config(state=state)
        self.y_axis_max_scale.config(state=state)
        self._save_now()

    def _on_y_axis_min_scale(self, v):
        n = max(Y_AXIS_RANGE_MIN, min(Y_AXIS_RANGE_MAX, round(float(v))))
        y_max = self.y_axis_max_var.get()
        if n >= y_max:
            n = y_max - 1
        if n < Y_AXIS_RANGE_MIN:
            n = Y_AXIS_RANGE_MIN
        self.y_axis_min_var.set(n)
        self.y_axis_min_label.config(text=str(n))
        self._save_now()

    def _on_y_axis_max_scale(self, v):
        n = max(Y_AXIS_RANGE_MIN, min(Y_AXIS_RANGE_MAX, round(float(v))))
        y_min = self.y_axis_min_var.get()
        if n <= y_min:
            n = y_min + 1
        if n > Y_AXIS_RANGE_MAX:
            n = Y_AXIS_RANGE_MAX
        self.y_axis_max_var.set(n)
        self.y_axis_max_label.config(text=str(n))
        self._save_now()

    def _on_primary_y_max_floor_scale(self, v):
        n = round(max(
            PRIMARY_Y_MAX_FLOOR_MIN,
            min(PRIMARY_Y_MAX_FLOOR_MAX, float(v)),
        ), 1)
        self.primary_y_max_floor_var.set(n)
        self.primary_y_max_floor_label.config(text=f"{n:.1f} g, N")
        self._save_now()

    def _bind_save_on_change(self):
        self.com_var.trace_add("write", self._save_now)
        self.baud_combo.bind("<<ComboboxSelected>>", lambda e: self._save_now())
        self.plot_time_var.trace_add("write", self._save_now)
        self.folder_var.trace_add("write", self._save_now)
        self.folder_var.trace_add("write", self._refresh_folder_entry_view)
        self.y_axis_mode_var.trace_add("write", self._save_now)
        self.y_axis_min_var.trace_add("write", self._save_now)
        self.y_axis_max_var.trace_add("write", self._save_now)
        self.primary_y_max_floor_var.trace_add("write", self._save_now)
        self.plot_ch1_var.trace_add("write", self._save_now)
        self.plot_ch2_var.trace_add("write", self._save_now)
        self.plot_ch3_var.trace_add("write", self._save_now)
        self.plot_ch4_var.trace_add("write", self._save_now)

    def _save_now(self, *_):
        try:
            com = self.com_var.get().strip() or DEFAULT_COM_PORT
            baud_s = self.baud_combo.get().strip()
            baud = int(baud_s) if baud_s and baud_s.isdigit() else DEFAULT_BAUD
            pt = max(PLOT_TIME_MIN_SEC, min(PLOT_TIME_MAX_SEC, self.plot_time_var.get()))
            folder = self.folder_var.get().strip() or DEFAULT_DATA_FOLDER
            y_mode = self.y_axis_mode_var.get()
            if y_mode not in ("auto", "fixed"):
                y_mode = DEFAULT_Y_AXIS_MODE
            y_min = self.y_axis_min_var.get()
            y_max = self.y_axis_max_var.get()
            if y_min >= y_max:
                y_min, y_max = DEFAULT_Y_AXIS_MIN, DEFAULT_Y_AXIS_MAX
            primary_y_max_floor = round(max(
                PRIMARY_Y_MAX_FLOOR_MIN,
                min(PRIMARY_Y_MAX_FLOOR_MAX, self.primary_y_max_floor_var.get()),
            ), 1)
            plot_ch1 = self.plot_ch1_var.get()
            plot_ch2 = self.plot_ch2_var.get()
            plot_ch3 = self.plot_ch3_var.get()
            plot_ch4 = self.plot_ch4_var.get()
            prefix = sanitize_windows_filename_prefix(self.prefix_var.get()) or DEFAULT_DATA_FILE_PREFIX
            channel_settings = self._read_channel_settings_from_ui()
            save_settings(com, baud, pt, folder, y_axis_mode=y_mode, y_axis_min=y_min, y_axis_max=y_max,
                         primary_y_max_floor=primary_y_max_floor,
                         plot_channel_1=plot_ch1, plot_channel_2=plot_ch2, plot_channel_3=plot_ch3, plot_channel_4=plot_ch4,
                         channel_settings=channel_settings)
            self.current_settings = {
                "com_port": com,
                "baud_rate_bps": baud,
                "plot_time_sec": pt,
                "data_folder": folder,
                "y_axis_mode": y_mode,
                "y_axis_min": y_min,
                "y_axis_max": y_max,
                "primary_y_max_floor": primary_y_max_floor,
                "plot_channel_1": plot_ch1,
                "plot_channel_2": plot_ch2,
                "plot_channel_3": plot_ch3,
                "plot_channel_4": plot_ch4,
                "data_file_prefix": prefix,
            }
            self.current_settings.update(channel_settings)
            if self.on_save_callback:
                self.on_save_callback(self.current_settings)
        except (ValueError, tk.TclError):
            pass

    def _read_channel_settings_from_ui(self):
        """Read per-channel model a, b, and display mode from preferences widgets."""
        channel_settings = {}
        for n in range(1, 5):
            i = n - 1
            try:
                a_val = float(self.ch_a_vars[i].get().strip())
            except (ValueError, AttributeError, tk.TclError):
                a_val = DEFAULT_CHANNEL_MODEL_A
            try:
                b_val = float(self.ch_b_vars[i].get().strip())
            except (ValueError, AttributeError, tk.TclError):
                b_val = DEFAULT_CHANNEL_MODEL_B
            label = self.ch_mode_combos[i].get().strip()
            mode = DISPLAY_MODE_LABEL_TO_KEY.get(label, DEFAULT_CHANNEL_DISPLAY_MODE)
            if mode not in CHANNEL_DISPLAY_MODES:
                mode = DEFAULT_CHANNEL_DISPLAY_MODE
            try:
                tare_val = float(self.current_settings.get(
                    f"channel_{n}_tare_value", DEFAULT_CHANNEL_TARE_VALUE))
            except (TypeError, ValueError):
                tare_val = DEFAULT_CHANNEL_TARE_VALUE
            try:
                slope_val = float(self.current_settings.get(
                    f"channel_{n}_slope_factor", DEFAULT_CHANNEL_SLOPE_FACTOR))
            except (TypeError, ValueError):
                slope_val = DEFAULT_CHANNEL_SLOPE_FACTOR
            try:
                target_mass = float(self.ch_target_vars[i].get().strip())
            except (ValueError, AttributeError, tk.TclError):
                target_mass = float(self.current_settings.get(
                    f"channel_{n}_target_mass", DEFAULT_CHANNEL_TARGET_MASS))
            channel_settings[f"channel_{n}_model_a"] = a_val
            channel_settings[f"channel_{n}_model_b"] = b_val
            channel_settings[f"channel_{n}_display_mode"] = mode
            channel_settings[f"channel_{n}_tare_value"] = tare_val
            channel_settings[f"channel_{n}_slope_factor"] = slope_val
            channel_settings[f"channel_{n}_target_mass"] = target_mass
        return channel_settings

    def _populate(self):
        s = self.current_settings
        port = s.get("com_port", DEFAULT_COM_PORT)
        ports = list(self.com_combo["values"])
        if port in ports:
            self.com_var.set(port)
        else:
            self.com_var.set(ports[0] if ports else DEFAULT_COM_PORT)
        self.baud_combo.set(str(s.get("baud_rate_bps", DEFAULT_BAUD)))
        self.plot_time_var.set(s.get("plot_time_sec", DEFAULT_PLOT_TIME_SEC))
        self.folder_var.set(s.get("data_folder", DEFAULT_DATA_FOLDER))
        self.after_idle(self._refresh_folder_entry_view)
        self.prefix_var.set(s.get("data_file_prefix", DEFAULT_DATA_FILE_PREFIX))
        y_mode = s.get("y_axis_mode", DEFAULT_Y_AXIS_MODE)
        if y_mode not in ("auto", "fixed"):
            y_mode = DEFAULT_Y_AXIS_MODE
        self.y_axis_mode_var.set(y_mode)
        y_min = s.get("y_axis_min", DEFAULT_Y_AXIS_MIN)
        y_max = s.get("y_axis_max", DEFAULT_Y_AXIS_MAX)
        if y_min >= y_max:
            y_min, y_max = DEFAULT_Y_AXIS_MIN, DEFAULT_Y_AXIS_MAX
        self.y_axis_min_var.set(y_min)
        self.y_axis_max_var.set(y_max)
        self.y_axis_min_label.config(text=str(y_min))
        self.y_axis_max_label.config(text=str(y_max))
        primary_y_max_floor = s.get("primary_y_max_floor", DEFAULT_PRIMARY_Y_MAX_FLOOR)
        try:
            primary_y_max_floor = float(primary_y_max_floor)
        except (TypeError, ValueError):
            primary_y_max_floor = DEFAULT_PRIMARY_Y_MAX_FLOOR
        primary_y_max_floor = round(max(
            PRIMARY_Y_MAX_FLOOR_MIN,
            min(PRIMARY_Y_MAX_FLOOR_MAX, primary_y_max_floor),
        ), 1)
        self.primary_y_max_floor_var.set(primary_y_max_floor)
        self.primary_y_max_floor_label.config(text=f"{primary_y_max_floor:.1f} g, N")
        self.plot_ch1_var.set(s.get("plot_channel_1", DEFAULT_PLOT_CHANNEL_1))
        self.plot_ch2_var.set(s.get("plot_channel_2", DEFAULT_PLOT_CHANNEL_2))
        self.plot_ch3_var.set(s.get("plot_channel_3", DEFAULT_PLOT_CHANNEL_3))
        self.plot_ch4_var.set(s.get("plot_channel_4", DEFAULT_PLOT_CHANNEL_4))
        for n in range(1, 5):
            i = n - 1
            self.ch_a_vars[i].set(str(s.get(f"channel_{n}_model_a", DEFAULT_CHANNEL_MODEL_A)))
            self.ch_b_vars[i].set(str(s.get(f"channel_{n}_model_b", DEFAULT_CHANNEL_MODEL_B)))
            mode = s.get(f"channel_{n}_display_mode", DEFAULT_CHANNEL_DISPLAY_MODE)
            if mode not in CHANNEL_DISPLAY_MODES:
                mode = DEFAULT_CHANNEL_DISPLAY_MODE
            self.ch_mode_combos[i].set(CHANNEL_DISPLAY_MODE_LABELS[mode])
            try:
                target_mass = float(s.get(f"channel_{n}_target_mass", DEFAULT_CHANNEL_TARGET_MASS))
            except (TypeError, ValueError):
                target_mass = DEFAULT_CHANNEL_TARGET_MASS
            self.ch_target_vars[i].set(str(target_mass))
            self._update_channel_calibration_ui(i)
        self._on_y_axis_mode_change()
        self._update_plot_label()

    def _update_plot_label(self):
        try:
            v = self.plot_time_var.get()
            self.plot_label.config(text=f"{v} sec")
        except tk.TclError:
            pass

    def _browse_folder(self):
        path = filedialog.askdirectory(initialdir=self.folder_var.get() or os.path.expanduser("~"))
        if path:
            self.folder_var.set(path)
            self._refresh_folder_entry_view()
            self._save_now()

    def _on_close(self):
        self._save_now()
        if self.main_app is not None:
            self.main_app.prefs_dialog = None
        self.destroy()


# *****************************************************************************
class MainApp:
    """
    ---------------------------------------------------------------
    MainApp (class)
    Main window: serial connect/disconnect, scrolling plot, record Play/Stop, menus, status bar.

    INPUTS
    (none) — instantiate and call run().

    OUTPUTS
    (GUI event loop)

   (c) Kai Kuck, 2026
    ---------------------------------------------------------------
    """

    def __init__(self):
        ensure_settings_file_exists()
        self.settings = load_settings()
        # Zero/scale workflow is per session; slope starts at 1.0 each run
        self.channel_zero_applied = [False, False, False, False]
        for n in range(1, 5):
            self.settings[f"channel_{n}_slope_factor"] = DEFAULT_CHANNEL_SLOPE_FACTOR
        try:
            primary_y_max_floor = float(self.settings.get(
                "primary_y_max_floor", DEFAULT_PRIMARY_Y_MAX_FLOOR
            ))
        except (TypeError, ValueError):
            primary_y_max_floor = DEFAULT_PRIMARY_Y_MAX_FLOOR
        self.settings["primary_y_max_floor"] = round(max(
            PRIMARY_Y_MAX_FLOOR_MIN,
            min(PRIMARY_Y_MAX_FLOOR_MAX, primary_y_max_floor),
        ), 1)
        self.root = tk.Tk()
        self.root.title("Larynx Force Sensor - Data Acquisition")
        self.root.geometry("960x600")
        self.root.minsize(MAIN_MIN_WIDTH, 400)
        self.ser = None
        self.reader_thread = None
        self.reader_stop = threading.Event()
        self.data_queue = queue.Queue()
        self.plot_times = []
        self.plot_values = [[], [], [], []]  # one list per channel
        self.max_points = 0
        self.connected = False
        self.recording = False
        self.record_file_handle = None
        self.record_filename = None
        self.record_start_time = None
        self.record_chunk_start = None
        self.record_first_row_pending = False
        self.calib_dialog = None
        self.prefs_dialog = None
        self.dac_axis_initial_range_pending = True
        self.anim = None
        self.last_received_values = [None, None, None, None]  # latest value per channel
        self._build_menus()
        self._build_toolbar()
        self._build_plot()
        self._build_status_bar()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._update_plot_length()
        self._schedule_value_label_update()
        self.root.after_idle(self._fit_main_window_min_width)

    def _fit_main_window_min_width(self):
        """Ensure window is wide enough for Connect/Record and all Latest channel groups."""
        try:
            self.root.update_idletasks()
            req_w = max(MAIN_MIN_WIDTH, self.root.winfo_reqwidth())
            min_h = self.root.minsize()[1]
            self.root.minsize(req_w, min_h)
            if self.root.winfo_width() < req_w:
                self.root.geometry(f"{req_w}x{self.root.winfo_height()}")
        except tk.TclError:
            pass

    def _build_menus(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu, underline=0)
        self.root.bind("<Alt-f>", lambda e: self.root.focus_set())
        file_menu.add_command(label="Save As", underline=0, accelerator="Alt+F, A", command=self._file_save_as)
        file_menu.add_command(label="Preferences", underline=0, accelerator="Alt+F, P", command=self._file_preferences)
        file_menu.add_command(
            label="Calibration", underline=0, accelerator="Alt+F, C",
            command=self._file_calibration,
        )
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu, underline=0)
        self.root.bind("<Alt-h>", lambda e: self.root.focus_set())
        help_menu.add_command(label="About", underline=0, accelerator="Alt+H, A", command=self._help_about)

    def _file_save_as(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=self.settings.get("data_folder", DEFAULT_DATA_FOLDER),
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write("timestamp;ch1;ch2;ch3;ch4\n")
                    n = len(self.plot_times)
                    for i in range(n):
                        t = self.plot_times[i]
                        ts = datetime.fromtimestamp(t).strftime("%d-%m-%Y;%H:%M:%S.") + f"{int((t % 1) * 1000):03d}"
                        v1 = self.plot_values[0][i] if i < len(self.plot_values[0]) else ""
                        v2 = self.plot_values[1][i] if i < len(self.plot_values[1]) else ""
                        v3 = self.plot_values[2][i] if i < len(self.plot_values[2]) else ""
                        v4 = self.plot_values[3][i] if i < len(self.plot_values[3]) else ""
                        f.write(f"{ts};{v1};{v2};{v3};{v4}\n")
                messagebox.showinfo("Save As", f"Saved to {path}")
            except Exception as ex:
                messagebox.showerror("Save As", str(ex))

    def _file_preferences(self):
        if self.prefs_dialog is not None:
            try:
                if self.prefs_dialog.winfo_exists():
                    self.prefs_dialog.deiconify()
                    self.prefs_dialog.lift()
                    self.prefs_dialog.focus_force()
                    return
            except tk.TclError:
                pass

        def on_save(new_s):
            old_s = dict(self.settings)
            self.settings = new_s
            self._check_dac_display_mode_newly_selected(old_s, new_s)
            self._update_plot_length()
            self._update_status()
            self._redraw_plot()
            if self.recording:
                model_changed = set()
                for ch in range(4):
                    n = ch + 1
                    if (old_s.get(f"channel_{n}_model_a") != new_s.get(f"channel_{n}_model_a")
                            or old_s.get(f"channel_{n}_model_b") != new_s.get(f"channel_{n}_model_b")):
                        model_changed.add(ch)
                if model_changed:
                    self._write_record_row(datetime.now(), self.last_received_values,
                                           calib_channels=model_changed)

        self.prefs_dialog = PreferencesDialog(self.root, self.settings, on_save_callback=on_save, main_app=self)
        self.prefs_dialog.transient(self.root)

    def _file_calibration(self):
        self._open_calibration_window()

    def _open_calibration_window(self):
        if self.calib_dialog is not None:
            try:
                if self.calib_dialog.winfo_exists():
                    self.calib_dialog.deiconify()
                    self.calib_dialog.lift()
                    self.calib_dialog.focus_force()
                    return
            except tk.TclError:
                pass
        self.calib_dialog = CalibrationDialog(self.root, self)
        self.calib_dialog.transient(self.root)

    def _help_about(self):
        messagebox.showinfo(
            "About",
            "Larynx Force Sensor - Data Acquisition\n\n"
            "Connects to ESP-32 via USB serial.\n"
            "Plots A/D values and records to CSV.\n\n\n"
            "Use File/Preferences to change settings.\n\n\n"
            "kai.kuck@hsc.utah.edu"
        )

    RECORD_STOP_BTN_SIZE = 40
    RECORD_STOP_SHAPE_SCALE = 0.525
    RECORD_STOP_CENTER_OFFSET_X = -2
    RECORD_STOP_CENTER_OFFSET_Y = -1

    def _draw_record_stop_icon(self):
        """Draw red circle (Record) or black square (Stop) centered on the canvas.
        Canvas coords: (0,0) = top-left; x1,y1 = top-left of shape, x2,y2 = bottom-right."""
        self.record_stop_canvas.delete("all")
        w = self.record_stop_canvas.winfo_width()
        h = self.record_stop_canvas.winfo_height()
        if w <= 1 or h <= 1:
            w = h = self.RECORD_STOP_BTN_SIZE
        shape_size = int(min(w, h) * self.RECORD_STOP_SHAPE_SCALE)
        cx = w // 2 + self.RECORD_STOP_CENTER_OFFSET_X
        cy = h // 2 + self.RECORD_STOP_CENTER_OFFSET_Y
        half = shape_size // 2
        x1 = cx - half
        y1 = cy - half
        x2 = x1 + shape_size
        y2 = y1 + shape_size
        if self.recording:
            self.record_stop_canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="black")
        else:
            self.record_stop_canvas.create_oval(x1, y1, x2, y2, fill="red", outline="red")

    def _on_record_button_enter(self, event=None):
        """Schedule tooltip to show after a short delay."""
        self._hide_record_tooltip()
        if self._record_tooltip_after_id is not None:
            self.root.after_cancel(self._record_tooltip_after_id)
        self._record_tooltip_after_id = self.root.after(400, self._show_record_tooltip)

    def _on_record_button_leave(self, event=None):
        """Cancel scheduled tooltip and hide it."""
        if self._record_tooltip_after_id is not None:
            self.root.after_cancel(self._record_tooltip_after_id)
            self._record_tooltip_after_id = None
        self._hide_record_tooltip()

    def _show_record_tooltip(self):
        self._record_tooltip_after_id = None
        self._hide_record_tooltip()
        try:
            text = "Stop Recording" if self.recording else "Start Recording"
            self._record_tooltip_window = tk.Toplevel(self.root)
            self._record_tooltip_window.overrideredirect(True)
            self._record_tooltip_window.wm_attributes("-topmost", True)
            lbl = tk.Label(self._record_tooltip_window, text=text, bg="#ffffc0", fg="black", relief=tk.SOLID, bd=1, font=("Segoe UI", 9), padx=6, pady=2)
            lbl.pack()
            self._record_tooltip_window.update_idletasks()
            x = self.record_stop_frame.winfo_rootx() + self.record_stop_frame.winfo_width() // 2 - self._record_tooltip_window.winfo_reqwidth() // 2
            y = self.record_stop_frame.winfo_rooty() + self.record_stop_frame.winfo_height() + 2
            self._record_tooltip_window.wm_geometry(f"+{x}+{y}")
        except (tk.TclError, AttributeError):
            pass

    def _hide_record_tooltip(self):
        try:
            if self._record_tooltip_window is not None and self._record_tooltip_window.winfo_exists():
                self._record_tooltip_window.destroy()
        except tk.TclError:
            pass
        self._record_tooltip_window = None

    def _match_record_stop_to_connect_height(self, event=None):
        """Resize Record/Stop button to be square with same height as Connect button."""
        try:
            if event is not None:
                h = event.height
            else:
                self.root.update_idletasks()
                h = self.connect_btn.winfo_height()
            if h and int(h) > 1:
                h = int(h)
                self.record_stop_frame.config(width=h, height=h)
                self.record_stop_canvas.config(width=h, height=h)
                self._draw_record_stop_icon()
        except (tk.TclError, AttributeError):
            pass

    def _build_toolbar(self):
        toolbar = ttk.Frame(self.root, padding=4)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        self.connect_btn = tk.Button(
            toolbar, text="Connect", command=self._toggle_connect,
            bg="#e0e0e0", fg="black", relief=tk.RAISED, bd=2,
            font=("Segoe UI", 10), width=10
        )
        self.connect_btn.pack(side=tk.LEFT, padx=2)
        self.connect_btn.bind("<Configure>", self._match_record_stop_to_connect_height)
        self.record_stop_frame = tk.Frame(toolbar, width=self.RECORD_STOP_BTN_SIZE, height=self.RECORD_STOP_BTN_SIZE, bg="#e0e0e0", relief=tk.RAISED, bd=2)
        self.record_stop_frame.pack(side=tk.LEFT, padx=2)
        self.record_stop_frame.pack_propagate(False)
        self.record_stop_canvas = tk.Canvas(self.record_stop_frame, width=self.RECORD_STOP_BTN_SIZE, height=self.RECORD_STOP_BTN_SIZE, bg="#e0e0e0", highlightthickness=0)
        self.record_stop_canvas.pack()
        self.record_stop_canvas.bind("<Button-1>", lambda e: self._toggle_record())
        self.record_stop_canvas.bind("<Configure>", lambda e: self._draw_record_stop_icon())
        self.record_stop_canvas.bind("<Enter>", self._on_record_button_enter)
        self.record_stop_canvas.bind("<Leave>", self._on_record_button_leave)
        self._record_tooltip_after_id = None
        self._record_tooltip_window = None
        self._draw_record_stop_icon()
        self.root.after(300, lambda: self._match_record_stop_to_connect_height())
        self.latest_frame = ttk.Frame(toolbar)
        self.latest_frame.pack(side=tk.LEFT, padx=(15, 0))
        ttk.Label(self.latest_frame, text="Latest:").pack(side=tk.LEFT, padx=(0, 8))
        self.value_label_vars = [tk.StringVar(value="—") for _ in range(4)]
        self.value_unit_vars = [tk.StringVar(value="") for _ in range(4)]
        self.value_labels = []
        self.value_unit_labels = []
        self.channel_title_labels = []
        plot_ch = [
            self.settings.get("plot_channel_1", DEFAULT_PLOT_CHANNEL_1),
            self.settings.get("plot_channel_2", DEFAULT_PLOT_CHANNEL_2),
            self.settings.get("plot_channel_3", DEFAULT_PLOT_CHANNEL_3),
            self.settings.get("plot_channel_4", DEFAULT_PLOT_CHANNEL_4),
        ]
        for i in range(4):
            fg = CHANNEL_COLORS[i] if plot_ch[i] else "gray"
            group = ttk.Frame(self.latest_frame)
            group.pack(side=tk.LEFT, padx=(0, MAIN_LATEST_GROUP_PAD))
            title_lbl = tk.Label(
                group, text=f"Ch{i + 1}:",
                font=MAIN_VALUE_FONT, fg=fg,
            )
            title_lbl.pack(side=tk.LEFT)
            self.channel_title_labels.append(title_lbl)
            lbl = tk.Label(
                group, textvariable=self.value_label_vars[i],
                font=MAIN_VALUE_FONT, fg=fg,
                width=MAIN_LATEST_VALUE_WIDTH, anchor=tk.W,
            )
            lbl.pack(side=tk.LEFT, padx=(MAIN_LATEST_INNER_PAD, 0))
            self.value_labels.append(lbl)
            unit_lbl = tk.Label(
                group, textvariable=self.value_unit_vars[i],
                font=MAIN_VALUE_FONT, fg=fg,
                width=MAIN_LATEST_UNIT_WIDTH, anchor=tk.W,
            )
            unit_lbl.pack(side=tk.LEFT)
            self.value_unit_labels.append(unit_lbl)

    def _channel_display_mode(self, ch):
        mode = self.settings.get(
            f"channel_{ch + 1}_display_mode", DEFAULT_CHANNEL_DISPLAY_MODE
        )
        return mode if mode in CHANNEL_DISPLAY_MODES else DEFAULT_CHANNEL_DISPLAY_MODE

    def _channel_uses_dac_axis(self, ch):
        return self._channel_display_mode(ch) == "dac"

    def _axis_for_channel(self, ch):
        return self.ax_dac if self._channel_uses_dac_axis(ch) else self.ax

    def _plot_channel_flags(self):
        return [
            self.settings.get("plot_channel_1", DEFAULT_PLOT_CHANNEL_1),
            self.settings.get("plot_channel_2", DEFAULT_PLOT_CHANNEL_2),
            self.settings.get("plot_channel_3", DEFAULT_PLOT_CHANNEL_3),
            self.settings.get("plot_channel_4", DEFAULT_PLOT_CHANNEL_4),
        ]

    def _dac_axis_has_plotted_channel(self, plot_ch=None):
        if plot_ch is None:
            plot_ch = self._plot_channel_flags()
        return any(plot_ch[ch] and self._channel_uses_dac_axis(ch) for ch in range(4))

    def _apply_dac_axis_y_limits(self, dac_vals, y_mode):
        """Right Y-axis: initial 0..27000 once, then auto or fixed per preferences."""
        if not self._dac_axis_has_plotted_channel():
            self.ax_dac.set_yticks([])
            return
        if y_mode == "fixed":
            y_min = self.settings.get("y_axis_min", DEFAULT_Y_AXIS_MIN)
            y_max = self.settings.get("y_axis_max", DEFAULT_Y_AXIS_MAX)
            if y_min >= y_max:
                y_min, y_max = DEFAULT_Y_AXIS_MIN, DEFAULT_Y_AXIS_MAX
            self.ax_dac.set_ylim(y_min, y_max)
        elif self.dac_axis_initial_range_pending:
            self.ax_dac.set_ylim(CALIB_Y_AXIS_INITIAL_MIN, CALIB_Y_AXIS_INITIAL_MAX)
            self.dac_axis_initial_range_pending = False
        elif dac_vals:
            _autoscale_ylim(
                self.ax_dac, dac_vals, min_span=CALIB_Y_AXIS_MIN_SPAN)
        else:
            self.ax_dac.set_ylim(CALIB_Y_AXIS_INITIAL_MIN, CALIB_Y_AXIS_INITIAL_MAX)
        _refresh_dac_axis_y_ticks(self.ax_dac)

    def _check_dac_display_mode_newly_selected(self, old_settings, new_settings):
        """Re-apply initial DAC axis range when a plotted channel switches to raw DAC."""
        plot_ch = [
            new_settings.get("plot_channel_1", DEFAULT_PLOT_CHANNEL_1),
            new_settings.get("plot_channel_2", DEFAULT_PLOT_CHANNEL_2),
            new_settings.get("plot_channel_3", DEFAULT_PLOT_CHANNEL_3),
            new_settings.get("plot_channel_4", DEFAULT_PLOT_CHANNEL_4),
        ]
        for ch in range(4):
            if not plot_ch[ch]:
                continue
            n = ch + 1
            old_mode = old_settings.get(
                f"channel_{n}_display_mode", DEFAULT_CHANNEL_DISPLAY_MODE)
            new_mode = new_settings.get(
                f"channel_{n}_display_mode", DEFAULT_CHANNEL_DISPLAY_MODE)
            if old_mode != "dac" and new_mode == "dac":
                self.dac_axis_initial_range_pending = True
                return

    def _ensure_line_on_axis(self, ch):
        """Move channel line to left (mass/force) or right (DAC) axis if mode changed."""
        target_ax = self._axis_for_channel(ch)
        line = self.lines[ch]
        if line.axes is target_ax:
            return
        color = CHANNEL_COLORS[ch]
        line.remove()
        new_line, = target_ax.plot([], [], color=color, linewidth=1.0)
        self.lines[ch] = new_line

    def _build_plot(self):
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax_dac = self.ax.twinx()
        self.lines = []
        for c in CHANNEL_COLORS:
            line, = self.ax.plot([], [], color=c, linewidth=1.0)
            self.lines.append(line)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Mass (g),   Force (N)")
        self.ax_dac.set_ylabel("A/D value")
        self.ax_dac.set_autoscaley_on(False)
        self.ax_dac.set_ylim(CALIB_Y_AXIS_INITIAL_MIN, CALIB_Y_AXIS_INITIAL_MAX)
        self.ax.grid(True, alpha=0.3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _build_status_bar(self):
        self.status_var = tk.StringVar()
        status = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status.pack(side=tk.BOTTOM, fill=tk.X)
        self._update_status()

    def _update_plot_length(self):
        plot_time_sec = int(self.settings.get("plot_time_sec", DEFAULT_PLOT_TIME_SEC))
        self.max_points = max(1, plot_time_sec * SAMPLES_PER_SEC)
        # Trim buffers so plot immediately reflects new time range
        if len(self.plot_times) > self.max_points:
            self.plot_times = self.plot_times[-self.max_points:]
            for ch in range(4):
                self.plot_values[ch] = self.plot_values[ch][-self.max_points:]

    def _schedule_value_label_update(self):
        """Schedule the next value-label update in 250 ms."""
        self.root.after(250, self._update_value_label)

    def _update_value_label(self):
        """Update the four 'Latest' value labels and their colors; then reschedule in 250 ms."""
        try:
            if not self.root.winfo_exists():
                return
            plot_ch = [
                self.settings.get("plot_channel_1", DEFAULT_PLOT_CHANNEL_1),
                self.settings.get("plot_channel_2", DEFAULT_PLOT_CHANNEL_2),
                self.settings.get("plot_channel_3", DEFAULT_PLOT_CHANNEL_3),
                self.settings.get("plot_channel_4", DEFAULT_PLOT_CHANNEL_4),
            ]
            for i in range(4):
                raw = self.last_received_values[i]
                if raw is None:
                    num_text, unit_text = "—", ""
                else:
                    converted = convert_dac_for_display(raw, self.settings, i)
                    num_text, unit_text = format_display_value_parts(
                        converted, self.settings, i)
                self.value_label_vars[i].set(num_text)
                self.value_unit_vars[i].set(unit_text)
                fg = CHANNEL_COLORS[i] if plot_ch[i] else "gray"
                self.channel_title_labels[i].config(fg=fg)
                self.value_labels[i].config(fg=fg)
                self.value_unit_labels[i].config(fg=fg)
        except (tk.TclError, AttributeError):
            return
        self._schedule_value_label_update()

    def _update_status(self):
        if self.connected and self.ser:
            self.status_var.set(
                f"Connected — {self.settings.get('com_port', '?')} @ "
                f"{self.settings.get('baud_rate_bps', '?')} bps"
            )
            self.connect_btn.config(text="Disconnect")
        else:
            self.status_var.set("Not connected")
            self.connect_btn.config(text="Connect")

    def _connect_serial(self):
        if serial is None:
            self.status_var.set("PySerial not installed. pip install pyserial")
            return
        port = self.settings.get("com_port", DEFAULT_COM_PORT)
        baud = self.settings.get("baud_rate_bps", DEFAULT_BAUD)
        try:
            self.ser = serial.Serial(port=port, baudrate=baud, timeout=0.01)
            self.reader_stop.clear()
            self.reader_thread = SerialReaderThread(self.ser, self.data_queue, self.reader_stop)
            self.reader_thread.start()
            self.connected = True
            self._update_status()
        except Exception as ex:
            self.status_var.set(f"Connection failed: {ex}")
            self.connected = False
            self._update_status()

    def _disconnect_serial(self):
        self.connected = False
        self.reader_stop.set()
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=2.0)
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
        self._update_status()

    def _toggle_connect(self):
        if self.connected:
            self._disconnect_serial()
        else:
            self._connect_serial()

    def _toggle_record(self):
        if self.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        folder = self.settings.get("data_folder", DEFAULT_DATA_FOLDER)
        os.makedirs(folder, exist_ok=True)
        prefix = (self.settings.get("data_file_prefix") or "").strip() or DEFAULT_DATA_FILE_PREFIX
        prefix = sanitize_windows_filename_prefix(prefix) or DEFAULT_DATA_FILE_PREFIX
        base = f"{prefix}_{data_filename_datetime()}.csv"
        self.record_filename = os.path.join(folder, base)
        try:
            self.record_file_handle = open(self.record_filename, "w", encoding="utf-8")
            self.record_file_handle.write(prefix + "\n")
            self.record_file_handle.write(CSV_RECORD_HEADER + "\n")
            self.record_start_time = datetime.now()
            self.record_chunk_start = self.record_start_time
            self.record_first_row_pending = True
            self.recording = True
            self._draw_record_stop_icon()
        except Exception as ex:
            messagebox.showerror("Recording", f"Cannot create file: {ex}")

    def _write_record_row(self, dt, dac_values, events=None, calib_channels=None):
        """Append one CSV data row while recording."""
        if not self.recording or not self.record_file_handle:
            return
        if events is None:
            events = ["", "", "", ""]
        if calib_channels is None:
            calib_channels = set()
        if self.record_first_row_pending:
            calib_channels = set(range(4))
            self.record_first_row_pending = False
        seconds_since_start = (dt - self.record_start_time).total_seconds()
        row = format_record_row(
            dt, seconds_since_start, dac_values, self.settings,
            events=events, calib_channels=calib_channels,
        )
        self.record_file_handle.write(row)

    def record_csv_calibration_event(self, channel_index, event_tag):
        """Write an event row (#Zero or #Slope) with updated calibration for one channel."""
        if not self.recording:
            return
        events = ["", "", "", ""]
        events[channel_index] = event_tag
        self._write_record_row(
            datetime.now(),
            self.last_received_values,
            events=events,
            calib_channels={channel_index},
        )

    def _stop_recording(self):
        if not self.recording:
            return
        if self.record_file_handle:
            try:
                self.record_file_handle.close()
            except Exception:
                pass
            self.record_file_handle = None
        self.recording = False
        self._draw_record_stop_icon()

    def _process_queue_and_record(self):
        n_max = max(1, self.max_points)  # ensure we never divide by zero or trim to 0
        while True:
            try:
                dt, values = self.data_queue.get_nowait()
            except queue.Empty:
                break
            if not isinstance(values, (list, tuple)) or len(values) < 4:
                values = [values if i == 0 else 0 for i in range(4)] if isinstance(values, (int, float)) else [0, 0, 0, 0]
            t = dt.timestamp()
            self.plot_times.append(t)
            for ch in range(4):
                self.plot_values[ch].append(values[ch] if ch < len(values) else 0)
            if len(self.plot_times) > n_max:
                self.plot_times = self.plot_times[-n_max:]
                for ch in range(4):
                    self.plot_values[ch] = self.plot_values[ch][-n_max:]
            self.last_received_values = list(values) if len(values) >= 4 else values + [0] * (4 - len(values))
            if self.recording and self.record_file_handle:
                self._write_record_row(dt, self.last_received_values)
                elapsed = (dt - self.record_chunk_start).total_seconds()
                if elapsed >= RECORD_CHUNK_DURATION_SEC:
                    self.record_file_handle.flush()
                    self.record_file_handle.close()
                    self.record_file_handle = open(self.record_filename, "a", encoding="utf-8")
                    self.record_chunk_start = dt

    def _redraw_plot(self):
        self._process_queue_and_record()
        plot_ch = self._plot_channel_flags()
        y_mode = self.settings.get("y_axis_mode", DEFAULT_Y_AXIS_MODE)
        dac_vals = []
        for ch in range(4):
            if plot_ch[ch] and self._channel_uses_dac_axis(ch) and self.plot_values[ch]:
                dac_vals.extend(self.plot_values[ch])
        if not self.plot_times:
            self._apply_dac_axis_y_limits(dac_vals, y_mode)
            return
        plot_time_sec = self.settings.get("plot_time_sec", DEFAULT_PLOT_TIME_SEC)
        # Time-based window: keep only the last plot_time_sec seconds so trace scrolls
        # (works regardless of sample rate; point-count trim alone fails if rate < 16 Hz)
        t_max = self.plot_times[-1]
        t_cutoff = t_max - float(plot_time_sec)
        keep_idx = [i for i in range(len(self.plot_times)) if self.plot_times[i] >= t_cutoff]
        if keep_idx:
            self.plot_times = [self.plot_times[i] for i in keep_idx]
            for ch in range(4):
                self.plot_values[ch] = [self.plot_values[ch][i] for i in keep_idx]
        if not self.plot_times:
            return
        t0 = self.plot_times[0]
        time_sec = [t - t0 for t in self.plot_times]
        for ch in range(4):
            self._ensure_line_on_axis(ch)
            if plot_ch[ch] and self.plot_values[ch]:
                plot_y = []
                for dac in self.plot_values[ch]:
                    v = convert_dac_for_display(dac, self.settings, ch)
                    plot_y.append(v if v is not None else float("nan"))
                self.lines[ch].set_data(time_sec, plot_y)
                self.lines[ch].set_visible(True)
            else:
                self.lines[ch].set_data([], [])
                self.lines[ch].set_visible(False)
        left_label, right_label = plot_axis_labels(self.settings, plot_ch)
        self.ax.set_ylabel(left_label)
        self.ax_dac.set_ylabel(right_label)
        self.ax.relim()
        self.ax_dac.relim()
        self.ax.set_xlim(0, plot_time_sec)
        left_vals = []
        dac_vals = []
        for ch in range(4):
            if not plot_ch[ch]:
                continue
            if self._channel_uses_dac_axis(ch):
                dac_vals.extend(self.plot_values[ch])
            else:
                for dac in self.plot_values[ch]:
                    v = convert_dac_for_display(dac, self.settings, ch)
                    if v is not None:
                        left_vals.append(v)
        if left_vals:
            primary_y_max_floor = self.settings.get(
                "primary_y_max_floor", DEFAULT_PRIMARY_Y_MAX_FLOOR
            )
            try:
                primary_y_max_floor = float(primary_y_max_floor)
            except (TypeError, ValueError):
                primary_y_max_floor = DEFAULT_PRIMARY_Y_MAX_FLOOR
            _autoscale_ylim(
                self.ax, left_vals, ymax_floor=primary_y_max_floor
            )
            _refresh_axis_y_ticks(self.ax, "left")
        else:
            self.ax.set_yticks([])
        self._apply_dac_axis_y_limits(dac_vals, y_mode)
        self.ax.autoscale_view(scalex=False, scaley=False)
        self.ax_dac.autoscale_view(scalex=False, scaley=False)
        self.canvas.draw_idle()

    def _on_close(self):
        if self.recording:
            self._stop_recording()
        if self.calib_dialog is not None:
            try:
                if self.calib_dialog.winfo_exists():
                    self.calib_dialog._on_close()
            except tk.TclError:
                pass
            self.calib_dialog = None
        if self.prefs_dialog is not None:
            try:
                if self.prefs_dialog.winfo_exists():
                    self.prefs_dialog._on_close()
            except tk.TclError:
                pass
            self.prefs_dialog = None
        self._disconnect_serial()
        if self.anim:
            self.anim.event_source.stop()
        self.root.destroy()

    def run(self):
        self.anim = animation.FuncAnimation(
            self.fig, lambda i: self._redraw_plot(), interval=50, cache_frame_data=False
        )
        self.root.mainloop()


# *****************************************************************************
def main():
    """
    ---------------------------------------------------------------
    main()
    Entry point: load settings, run main application.

    INPUTS
    (none)

    OUTPUTS
    returns: None.

   (c) Kai Kuck, 2026
    ---------------------------------------------------------------
    """
    app = MainApp()
    app.run()


if __name__ == "__main__":
    main()
