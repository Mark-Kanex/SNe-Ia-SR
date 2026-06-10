import math
import numpy as np
import pandas as pd


OPERATOR_COMPLEXITIES = {"+": 1, "-": 1, "*": 2, "/": 5, "^": 5, "sin": 5, "exp": 5}


def read_csv_any(path: str) -> pd.DataFrame:
    data_frame = pd.read_csv(path)
    data_frame = data_frame.rename(columns={c: c.strip() for c in data_frame.columns})
    rename = {}
    for c in list(data_frame.columns):
        lc = c.strip().lower()
        if lc == "tjd" and "time" not in data_frame.columns:
            rename[c] = "time"
        if lc in ("cts_per_s", "counts_per_sec", "counts_per_s") and "flux" not in data_frame.columns:
            rename[c] = "flux"
        if lc in ("e_cts_per_s", "flux_err", "e_flux", "fluxerror", "err") and "flux_err" not in data_frame.columns:
            rename[c] = "flux_err"
    if rename:
        data_frame = data_frame.rename(columns=rename)
    if "flux_err" not in data_frame.columns:
        data_frame["flux_err"] = np.nan
    return data_frame[["time", "flux", "flux_err"]]


def find_explosion_epoch(time_values: np.ndarray, flux_values: np.ndarray, flux_errors: np.ndarray) -> float:
    explosion_time = float(time_values[0])
    for index in range(len(time_values)):
        if np.all(flux_values[index:] > 5 * flux_errors[index:]) and np.all(flux_values[index:] > 0.0):
            explosion_time = float(time_values[index])
            break
    return explosion_time


def shift_lower_bound_by_point(time_since_explosion: np.ndarray, lower_bound: float) -> float:
    if time_since_explosion.size == 0:
        return lower_bound
    mask = np.isfinite(time_since_explosion)
    if not np.any(mask):
        return lower_bound
    values = np.sort(time_since_explosion[mask])
    candidates = values[values < lower_bound]
    if candidates.size == 0:
        return lower_bound
    return float(candidates[-1])


def get_plot_window(
    base_name: str,
    time_since_explosion: np.ndarray,
    flux_values: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    core_name = base_name.replace("_binned", "").lower()
    if core_name == "2018fub":
        return 6.0, 20.0, 1.0, 1.0
    if core_name == "2018adf":
        return 1.0, 20.0, 1.0, 1.0
    if core_name == "2018alh":
        if time_since_explosion.size == 0 or not np.isfinite(time_since_explosion).any():
            plot_end = 15.0
        elif flux_values is None or flux_values.size == 0 or not np.isfinite(flux_values).any():
            plot_end = math.floor(float(np.nanmax(time_since_explosion)))
        else:
            idx = int(np.nanargmax(flux_values))
            plot_end = math.floor(float(time_since_explosion[idx]))
        return 15.0, plot_end, 1.0, 1.0
    if core_name == "2018auo":
        if time_since_explosion.size == 0 or not np.isfinite(time_since_explosion).any():
            plot_end = 32.0
        elif flux_values is None or flux_values.size == 0 or not np.isfinite(flux_values).any():
            plot_end = math.floor(float(np.nanmax(time_since_explosion)))
        else:
            idx = int(np.nanargmax(flux_values))
            plot_end = math.floor(float(time_since_explosion[idx]))
        return 32.0, plot_end, 1.0, 1.0
    if core_name == "2018agk":
        if time_since_explosion.size == 0 or not np.isfinite(time_since_explosion).any():
            plot_end = 10.0
        elif flux_values is None or flux_values.size == 0 or not np.isfinite(flux_values).any():
            plot_end = math.floor(float(np.nanmax(time_since_explosion)))
        else:
            idx = int(np.nanargmax(flux_values))
            plot_end = math.floor(float(time_since_explosion[idx]))
        return 10.0, plot_end, 1.0, 1.0
    if core_name == "2018nt":
        if time_since_explosion.size == 0 or not np.isfinite(time_since_explosion).any():
            plot_end = 1.0
        elif flux_values is None or flux_values.size == 0 or not np.isfinite(flux_values).any():
            plot_end = math.floor(float(np.nanmax(time_since_explosion)))
        else:
            idx = int(np.nanargmax(flux_values))
            plot_end = math.floor(float(time_since_explosion[idx]))
        return 1.0, plot_end, 1.0, 1.0
    if core_name == "2018oh":
        return 1.0, 17.0, 1.0, 1.0
    if core_name == "2020tld":
        plot_end = float(np.nanmax(time_since_explosion)) - 1.0
        return 0.0, plot_end, 1.0, 1.0
    if core_name == "2023inb":
        plot_start = shift_lower_bound_by_point(time_since_explosion, 0.0)
        return plot_start, 14.0, 1.0, 1.0
    if core_name == "2023bee":
        plot_start = shift_lower_bound_by_point(time_since_explosion, 4.0)
        return plot_start, 12.5, 1.0, 1.0
    return 0.0, float(np.nanmax(time_since_explosion)), 1.0, 1.0
