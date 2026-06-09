
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares

plt.style.use("seaborn-v0_8-white")

POINT_COLOR = "#2F2A26"
FIT_COLOR   = "#3B82F6"

DATA_DIR   = Path(__file__).parent.parent / "TESS_Data"
GRAPHS_DIR = Path(__file__).parent
OUT_CSV    = DATA_DIR / "fit_curve_window_results.csv"


def find_explosion_epoch(times, fluxes, flux_errs):
    t_exp = float(times[0])
    for i in range(len(times)):
        if np.all(fluxes[i:] > 5 * flux_errs[i:]) and np.all(fluxes[i:] > 0.0):
            t_exp = float(times[i])
            break
    return t_exp


def get_plot_window(sn_id, time_since_exp, fluxes=None):
    name = sn_id.lower()
    def peak_time():
        if fluxes is not None and fluxes.size > 0 and np.isfinite(fluxes).any():
            return math.floor(float(time_since_exp[np.nanargmax(fluxes)]))
        return math.floor(float(np.nanmax(time_since_exp)))

    if name == "2018fub":  return 6.0,  20.0,       1.0, 1.0
    if name == "2018adf":  return 1.0,  20.0,       1.0, 1.0
    if name == "2018oh":   return 1.0,  17.0,       1.0, 1.0
    if name == "2018alh":  return 15.0, peak_time(), 1.0, 1.0
    if name == "2018auo":  return 32.0, peak_time(), 1.0, 1.0
    if name == "2018agk":  return 10.0, peak_time(), 1.0, 1.0
    if name == "2018nt":   return 1.0,  peak_time(), 1.0, 1.0
    if name == "2020tld":
        return 0.0, float(np.nanmax(time_since_exp)) - 1.0, 1.0, 1.0
    if name == "2023inb":
        candidates = time_since_exp[time_since_exp < 0.0]
        plot_start = float(candidates[-1]) if candidates.size > 0 else 0.0
        return plot_start, 14.0, 1.0, 1.0
    if name == "2023bee":
        candidates = time_since_exp[time_since_exp < 4.0]
        plot_start = float(candidates[-1]) if candidates.size > 0 else 4.0
        return plot_start, 12.5, 1.0, 1.0
    return 0.0, float(np.nanmax(time_since_exp)), 1.0, 1.0


def _model(x, a, b, c, d, e, f):
    return a * x**2 + b * x + c + d * np.sin(e * x + f)


def fit_parametric(x, y, yerr):
    mask = np.isfinite(x) & np.isfinite(y)
    sigma = None
    if yerr is not None and np.any(yerr):
        s = np.where((yerr <= 0) | (~np.isfinite(yerr)), np.nan, yerr)
        if np.isfinite(s).any():
            sigma = s
            mask = mask & np.isfinite(sigma)

    xf, yf = x[mask], y[mask]
    sf = sigma[mask] if sigma is not None else None
    if xf.size < 6:
        return None

    a0, b0, c0 = np.polyfit(xf, yf, 2)
    d0   = float(np.nanstd(yf - (a0*xf**2 + b0*xf + c0))) or 0.0
    span = float(np.nanmax(xf) - np.nanmin(xf)) if xf.size else 0.0
    e0   = 2.0 * np.pi / span if span > 0 else 1.0

    p0_options = [
        [a0, b0, c0, d0, e0, 0.0],
        [a0, b0, c0, d0 * 0.5, e0 * 2, 0.0],
        [a0, b0, c0, 0.0, e0, 0.0],
    ]

    best_popt = None
    best_sse = float('inf')
    for p0 in p0_options:
        try:
            popt, _ = curve_fit(_model, xf, yf, p0=p0,
                                sigma=sf, absolute_sigma=sf is not None, maxfev=20000)
            sse = float(np.sum((_model(xf, *popt) - yf) ** 2))
            if sse < best_sse:
                best_sse = sse
                best_popt = popt
        except Exception:
            pass
        try:
            def res(p):
                r = _model(xf, *p) - yf
                return r / sf if sf is not None else r
            result = least_squares(res, p0, loss="soft_l1",
                                   f_scale=max(float(np.nanmedian(np.abs(yf))), 1e-6),
                                   max_nfev=20000)
            sse = float(np.sum((_model(xf, *result.x) - yf) ** 2))
            if sse < best_sse:
                best_sse = sse
                best_popt = result.x
        except Exception:
            pass

    return best_popt


def _ylim(values, pad=0.05):
    v = values[np.isfinite(values)]
    if v.size == 0:
        return None, None
    lo, hi = float(v.min()), float(v.max())
    m = pad * (hi - lo) if hi > lo else 1.0
    return lo - m, hi + m


def save_grid(curves, out_path, *, show_fit=True, nrows=2, ncols=5):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 10})
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5*ncols, 4.5*nrows), squeeze=False)

    for ax, c in zip(axes.ravel(), curves):
        x_disp = c["x_plot"] - c["plot_start"]
        ax.errorbar(x_disp, c["y_plot"], yerr=c["yerr_plot"],
                    fmt="o", ms=3, lw=0.8, color=POINT_COLOR, label="Data")
        if show_fit and c["params"] is not None:
            xs = np.linspace(c["plot_start"], c["t_max"], 300)
            ax.plot(xs - c["plot_start"], _model(xs, *c["params"]),
                    color=FIT_COLOR, linewidth=1.4, label="Regression")
        ax.text(0.05, 0.95, c["name"], transform=ax.transAxes,
                fontsize=24, fontweight="bold", va="top", ha="left")
        ax.set_xlabel("Days since explosion")
        ax.set_ylabel("Flux")
        ax.set_xlim(0.0, c["display_t_max"])
        lo, hi = _ylim(c["y_plot"])
        if lo is not None:
            ax.set_ylim(lo, hi)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(False)
        if show_fit:
            ax.legend(frameon=False, fontsize=8)

    for ax in axes.ravel()[len(curves):]:
        ax.axis("off")

    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


rows   = []
curves = []

for csv_path in sorted(DATA_DIR.glob("*_binned.csv")):
    sn_id = csv_path.stem.replace("_binned", "")
    df    = pd.read_csv(csv_path).dropna(subset=["time", "flux"])

    times     = df["time"].to_numpy(dtype=float)
    fluxes    = df["flux"].to_numpy(dtype=float)
    flux_errs = df["flux_err"].fillna(0.0).to_numpy(dtype=float)

    t_exp         = find_explosion_epoch(times, fluxes, flux_errs)
    time_rel      = times - t_exp
    plot_start, plot_end, left_margin, right_margin = get_plot_window(sn_id, time_rel, fluxes)
    time_since    = time_rel - plot_start
    t_max         = plot_end - plot_start

    plot_mask = (time_since >= 0.0) & (time_since <= t_max)
    fit_mask  = (time_since >= -left_margin) & (time_since <= t_max + right_margin)

    # 2020tld has a flux spike at t≈59107.876 TJD.
    # including it forces the regression into rapid oscillation to accommodate the outlier.
    # This point is therefore excluded for regression purposes.

    if sn_id == "2020tld":
        fit_mask = fit_mask & (np.abs(times - 59107.8756786985) > 0.1)

    print(f"{sn_id}: n_plot={plot_mask.sum()}, t_max={t_max:.2f}")

    params = fit_parametric(time_since[fit_mask], fluxes[fit_mask], flux_errs[fit_mask])
    if params is None:
        print(f"  fit failed")
        continue

    a, b, c, d, e, f = params
    rows.append({"SN_ID": sn_id, "a": a, "b": b, "c": c, "d": d, "e": e, "f": f})

    curves.append({
        "name":        sn_id,
        "x_plot":      time_since[plot_mask],
        "y_plot":      fluxes[plot_mask],
        "yerr_plot":   flux_errs[plot_mask],
        "params":      params,
        "plot_start":  0.0,
        "t_max":       t_max,
        "display_t_max": t_max,
    })

pd.DataFrame(rows, columns=["SN_ID", "a", "b", "c", "d", "e", "f"]).to_csv(OUT_CSV, index=False)
print(f"\nWrote {len(rows)} fits to {OUT_CSV}")

if curves:
    fit_out  = GRAPHS_DIR / "all_parametric_2x5.png"
    plain_out = DATA_DIR  / "all_plain_2x5.png"
    save_grid(curves, str(fit_out), show_fit=True)
    save_grid(curves, str(plain_out), show_fit=False)
    print(f"Wrote parametric grid: {fit_out}")
    print(f"Wrote plain grid:      {plain_out}")
