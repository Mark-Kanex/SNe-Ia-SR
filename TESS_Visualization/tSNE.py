from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from adjustText import adjust_text
import corner
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity, NearestNeighbors

MULTI_PANEL = True
SNR_FACTOR = 0
LIMIT_MULTI_PANEL_COLORBAR = True
FIX_F_RANGE = True
REPRESENTATIVE_METHOD = "kde"  # "medoid" or "kde"
HIGH_DENSITY_FRACTION = 0.2

BASE_DIR = Path(__file__).resolve().parent
VALIDATION_CURVE_DIR = BASE_DIR / "catalog_split"
TRAINING_CURVE_DIR = BASE_DIR.parent / "TESS_Data"
PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

ICML_SINGLE_COLUMN_WIDTH = 3.25
ICML_DOUBLE_COLUMN_WIDTH = 6.75
BASE_FONT_SIZE = 8
TICK_FONT_SIZE = 7
LEGEND_FONT_SIZE = 7
SMALL_TEXT_SIZE = 5.5

plt.rcParams.update(
    {
        "font.size": BASE_FONT_SIZE,
        "axes.labelsize": BASE_FONT_SIZE,
        "axes.titlesize": BASE_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "figure.titlesize": BASE_FONT_SIZE,
        "savefig.dpi": 300,
    }
)


def wrap_phase_to_pi(values):
    values = pd.to_numeric(values, errors="coerce")
    return ((values + np.pi) % (2.0 * np.pi)) - np.pi


def get_wrapped_phase_series(df, column="f"):
    original = pd.to_numeric(df[column], errors="coerce")
    wrapped = wrap_phase_to_pi(original)
    # Values like +pi, +3pi, ... should stay on the positive endpoint.
    return wrapped.mask(np.isclose(wrapped, -np.pi) & (original > 0), np.pi)


def canonicalize_sine_parameters(df, d_col="d_norm", f_col="f"):
    d_values = pd.to_numeric(df[d_col], errors="coerce")
    f_values = get_wrapped_phase_series(df, column=f_col)

    negative_mask = d_values < 0
    shifted_f = wrap_phase_to_pi(f_values + np.pi)
    shifted_f = shifted_f.mask(np.isclose(shifted_f, -np.pi) & ((f_values + np.pi) > 0), np.pi)

    df[d_col] = d_values.abs()
    df[f_col] = f_values.where(~negative_mask, shifted_f)
    return df


def get_colorbar_limits(series, mask=None):
    if mask is not None and mask.any():
        values = pd.to_numeric(series.loc[mask], errors="coerce")
    else:
        values = pd.to_numeric(series, errors="coerce")

    values = values[np.isfinite(values)]
    if values.empty:
        values = pd.to_numeric(series, errors="coerce")
        values = values[np.isfinite(values)]

    if values.empty:
        return -1.0, 1.0

    vmin = values.min()
    vmax = values.max()
    if np.isclose(vmin, vmax):
        vmin -= 1e-9
        vmax += 1e-9
    return float(vmin), float(vmax)


def passes_amplitude_threshold(sn_id, source):
    if source == "Validation":
        path = VALIDATION_CURVE_DIR / f"{sn_id}_binned.csv"
        flux_col = "CRate"
        err_col = "e_CRate"
    else:
        path = TRAINING_CURVE_DIR / f"{sn_id}_binned.csv"
        flux_col = "flux"
        err_col = "flux_err"

    if not path.exists():
        return False

    df = pd.read_csv(path)
    flux = pd.to_numeric(df[flux_col], errors="coerce").dropna()
    err = pd.to_numeric(df[err_col], errors="coerce").dropna()
    if flux.empty or err.empty:
        return False

    amplitude = flux.max() - flux.min()
    median_err = err.median()
    if pd.isna(amplitude) or pd.isna(median_err) or median_err <= 0:
        return False
    return amplitude > SNR_FACTOR * median_err


validation_df = pd.read_csv("catalog_split/fit_curve_window_results_normalized.csv")
validation_df = validation_df[validation_df["n_points"] >= 30].copy()
validation_df["source"] = "Validation"
validation_df["chi2_per_point"] = validation_df["chi2"] / validation_df["n_points"]
validation_df["high_chi2"] = validation_df["chi2_per_point"] >= validation_df["chi2_per_point"].quantile(0.9)
validation_df["high_r2"] = validation_df["r2"] >= validation_df["r2"].quantile(0.9)

tess_df = pd.read_csv("../TESS_Data/fit_curve_window_results.csv") #lc_rescaled_fromMark.csv")
tess_df = tess_df[tess_df['SN_ID'] != '2020tld'].copy() # Exclude SN2020tld which has an extremely high chi2/n value that skews the color scale
#tess_df = tess_df[tess_df["n_points"] >= 30].copy()
tess_df["source"] = "Training"
tess_df["high_chi2"] = False
tess_df["high_r2"] = False

if FIX_F_RANGE:
    validation_df["f"] = get_wrapped_phase_series(validation_df)
    tess_df["f"] = get_wrapped_phase_series(tess_df)

validation_df = canonicalize_sine_parameters(validation_df)
tess_df = canonicalize_sine_parameters(tess_df)

cols = ["a_norm", "b_norm", "c_norm", "d_norm", "e", "f"]
display_labels = {
    "a_norm": r"$\tilde{a}$",
    "b_norm": r"$\tilde{b}$",
    "c_norm": r"$\tilde{c}$",
    "d_norm": r"$\tilde{d}$",
    "e": r"$e$",
    "f": r"$f$",
}


def format_sn_label(sn_id):
    return sn_id if str(sn_id).startswith("SN ") else f"SN {sn_id}"


plot_df = pd.concat(
    [
        validation_df[["SN_ID", "source", "high_chi2", "high_r2"] + cols],
        tess_df[["SN_ID", "source", "high_chi2", "high_r2"] + cols],
    ],
    ignore_index=True,
).dropna().copy()
plot_df["low_snr"] = ~plot_df.apply(
    lambda row: passes_amplitude_threshold(row["SN_ID"], row["source"]),
    axis=1,
)


def get_point_color(row, default_color):
    if row["low_snr"]:
        return "red"
    if row["source"] == "Validation" and row["high_chi2"]:
        return "orange"
    if row["source"] == "Validation" and row["high_r2"]:
        return "purple"
    return default_color


def normalized_model(x, params):
    return (
        normalized_quadratic(x, params)
        + normalized_sine(x, params)
    )


def normalized_quadratic(x, params):
    return params["a_norm"] * x**2 + params["b_norm"] * x + params["c_norm"]


def normalized_sine(x, params):
    return params["d_norm"] * np.sin(params["e"] * x + params["f"])


def fit_shifted_t_squared(times, fluxes, t0_init=0.0):
    times = np.asarray(times, dtype=float)
    fluxes = np.asarray(fluxes, dtype=float)

    def solve_amplitude_and_rss(t0):
        basis = (times - t0) ** 2
        denom = np.dot(basis, basis)
        if denom <= 0:
            return 0.0, np.inf
        amplitude = np.dot(fluxes, basis) / denom
        residuals = fluxes - amplitude * basis
        rss = np.dot(residuals, residuals)
        return amplitude, rss

    center = float(t0_init)
    span = 3.0
    best_t0 = center
    best_a, best_rss = solve_amplitude_and_rss(best_t0)

    for _ in range(4):
        grid = np.linspace(center - span, center + span, 801)
        candidates = [(*solve_amplitude_and_rss(t0), t0) for t0 in grid]
        best_a, best_rss, best_t0 = min(candidates, key=lambda item: item[1])
        center = best_t0
        span /= 5.0

    return best_a, best_t0


def standardize_array(values):
    center = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    return (values - center) / scale, center, scale


def estimate_kde_bandwidth(standardized_values):
    n_samples, n_dim = standardized_values.shape
    if n_samples <= 1:
        return 1.0
    return max(n_samples ** (-1.0 / (n_dim + 4)), 0.1)


def get_representative_params(model_df, method, cols):
    values = model_df[cols].to_numpy(dtype=float)
    standardized_values, _, _ = standardize_array(values)

    if len(model_df) == 1:
        return model_df.iloc[0][cols].copy(), "single sample"

    if method == "medoid":
        n_neighbors = min(10, len(model_df) - 1)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1)
        nbrs.fit(standardized_values)
        distances, _ = nbrs.kneighbors(standardized_values)
        local_density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-12)
        density_threshold = np.quantile(local_density, max(0.0, 1.0 - HIGH_DENSITY_FRACTION))
        dense_mask = local_density >= density_threshold
        dense_values = standardized_values[dense_mask]
        dense_df = model_df.loc[dense_mask].reset_index(drop=True)
        if len(dense_df) == 0:
            dense_values = standardized_values
            dense_df = model_df.reset_index(drop=True)
        pairwise = np.linalg.norm(dense_values[:, None, :] - dense_values[None, :, :], axis=2)
        medoid_idx = np.argmin(pairwise.sum(axis=1))
        return dense_df.iloc[medoid_idx][cols].copy(), "high-density medoid"

    if method == "kde":
        bandwidth = estimate_kde_bandwidth(standardized_values)
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde.fit(standardized_values)
        log_density = kde.score_samples(standardized_values)
        mode_idx = int(np.argmax(log_density))
        return model_df.iloc[mode_idx][cols].copy(), "KDE peak sample"

    raise ValueError(f"Unsupported REPRESENTATIVE_METHOD: {method}")


def format_representative_label(label):
    return label.replace("KDE", "KDE").replace("kde", "KDE")

embedding = TSNE(
    n_components=2,
    perplexity=15,
    random_state=42,
    init="pca",
    learning_rate="auto",
).fit_transform(plot_df[cols].to_numpy())

plot_df["TSNE1"] = embedding[:, 0]
plot_df["TSNE2"] = embedding[:, 1]

colorbar_mask = (
    ((plot_df["TSNE1"] < 0) & (plot_df["TSNE2"] > -5))
#    | ((plot_df["TSNE1"] >= -7) & (plot_df["TSNE1"] < 10) & (plot_df["TSNE2"] > 5))
    | ((plot_df["TSNE1"] < 10) & (plot_df["TSNE2"] <= -5)) 
)

fitted_validation = int((plot_df["source"] == "Validation").sum())
fitted_training = int((plot_df["source"] == "Training").sum())
mask_validation = int(((plot_df["source"] == "Validation") & colorbar_mask).sum())
mask_training = int(((plot_df["source"] == "Training") & colorbar_mask).sum())

print(f"Fitted objects: total={len(plot_df)}, validation={fitted_validation}, training={fitted_training}")
print(
    "Objects in colorbar_mask: "
    f"total={int(colorbar_mask.sum())}, validation={mask_validation}, training={mask_training}"
)

marker_map = {
    "Training": "*",
    "Validation": "o",
}

legend_handles = [
    Line2D([0], [0], marker="*", color="w", markerfacecolor="gray", markersize=12, label="Training"),
    Line2D([0], [0], marker="o", color="gray", markerfacecolor="none", markersize=8, label="Validation"),
    Line2D([0], [0], marker="o", color="red", markerfacecolor="none", markersize=8, label="Low S/N"),
    Line2D([0], [0], marker="o", color="orange", markerfacecolor="none", markersize=8, label="Top 10% chi2/n"),
    Line2D([0], [0], marker="o", color="purple", markerfacecolor="none", markersize=8, label="Top 10% R^2"),
]

legend_handles_single = legend_handles
legend_handles_multi = [
    Line2D([0], [0], marker="*", color="w", markerfacecolor="gray", markersize=12, label="Training"),
    Line2D([0], [0], marker="o", color="gray", markerfacecolor="none", markersize=8, label="Validation"),
]

if MULTI_PANEL:
    fig, axes = plt.subplots(2, 3, figsize=(ICML_DOUBLE_COLUMN_WIDTH, 4.6))
    axes = axes.ravel()
    cmap = "viridis"

    for idx, (ax, feature) in enumerate(zip(axes, cols)):
        mask = colorbar_mask if LIMIT_MULTI_PANEL_COLORBAR else None
        vmin, vmax = get_colorbar_limits(plot_df[feature], mask=mask)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = colormaps[cmap]
        texts = []
        all_x = plot_df["TSNE1"].to_numpy()
        all_y = plot_df["TSNE2"].to_numpy()

        for source in ["Training", "Validation"]:
            source_df = plot_df[plot_df["source"] == source]
            feature_values = pd.to_numeric(source_df[feature], errors="coerce").to_numpy()
            if source == "Training":
                ax.scatter(
                    source_df["TSNE1"],
                    source_df["TSNE2"],
                    c=feature_values,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    s=90,
                    alpha=0.85,
                    marker=marker_map[source],
                    edgecolors="none",
                )
                for _, row in source_df.iterrows():
                    texts.append(
                        ax.text(
                            row["TSNE1"] + 1.0,
                            row["TSNE2"] + 0.6,
                            format_sn_label(row["SN_ID"]),
                            fontsize=SMALL_TEXT_SIZE,
                            alpha=1,
                            color=cmap_obj(norm(float(row[feature]))),
                        )
                    )
            else:
                edge_colors = cmap_obj(norm(feature_values))
                ax.scatter(
                    source_df["TSNE1"],
                    source_df["TSNE2"],
                    s=45,
                    marker=marker_map[source],
                    facecolors="none",
                    edgecolors=edge_colors,
                    linewidths=1.2,
                    alpha=0.5,
                )

        adjust_text(
            texts,
            x=all_x,
            y=all_y,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.4),
            expand_points=(1.2, 1.4),
            expand_text=(1.1, 1.2),
        )

        row_idx, col_idx = divmod(idx, 3)
        if row_idx == 1:
            ax.set_xlabel("t-SNE 1")
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)

        if col_idx == 0:
            ax.set_ylabel("t-SNE 2")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        mappable.set_array([])
        cax = inset_axes(ax, width="36%", height="6%", loc="upper right", borderpad=2.)
        cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=TICK_FONT_SIZE - 1, pad=1, length=2)
        cbar.set_label(display_labels[feature], fontsize=TICK_FONT_SIZE, labelpad=1)
        cbar.ax.xaxis.set_label_position("top")

    fig.legend(handles=legend_handles_multi, loc="upper center", ncol=2, frameon=False, fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout(rect=[0, 0, 1, 0.97], pad=0.5, w_pad=0.3, h_pad=0.4)
else:
    fig = plt.figure(figsize=(ICML_DOUBLE_COLUMN_WIDTH, 4.1))
    texts = []
    for source in ["Training", "Validation"]:
        source_df = plot_df[plot_df["source"] == source]
        if source == "Training":
            low_snr_df = source_df[source_df["low_snr"]]
            normal_df = source_df[~source_df["low_snr"]]
            plt.scatter(
                normal_df["TSNE1"],
                normal_df["TSNE2"],
                s=90,
                alpha=0.85,
                color="black",
                marker=marker_map[source],
                edgecolors="none",
                label=source,
            )
            plt.scatter(
                low_snr_df["TSNE1"],
                low_snr_df["TSNE2"],
                s=90,
                alpha=0.9,
                color="red",
                marker=marker_map[source],
                edgecolors="none",
            )
            for _, row in normal_df.iterrows():
                texts.append(
                    plt.text(
                        row["TSNE1"] + 0.6,
                        row["TSNE2"] + 0.6,
                        format_sn_label(row["SN_ID"]),
                        fontsize=SMALL_TEXT_SIZE,
                        alpha=0.85,
                        color="black",
                    )
                )
            for _, row in low_snr_df.iterrows():
                texts.append(
                    plt.text(
                        row["TSNE1"] + 0.6,
                        row["TSNE2"] + 0.6,
                        format_sn_label(row["SN_ID"]),
                        fontsize=SMALL_TEXT_SIZE,
                        alpha=0.9,
                        color="red",
                    )
                )
        else:
            first_validation = True
            for _, row in source_df.iterrows():
                edge_color = get_point_color(row, "C0")
                plt.scatter(
                    [row["TSNE1"]],
                    [row["TSNE2"]],
                    s=45,
                    marker=marker_map[source],
                    facecolors="none",
                    edgecolors=edge_color,
                    linewidths=1.4 if edge_color != "C0" else 1.2,
                    alpha=0.5,
                    label=source if first_validation else None,
                )
                first_validation = False
                texts.append(
                    plt.text(
                        row["TSNE1"] + 0.6,
                        row["TSNE2"] + 0.6,
                        format_sn_label(row["SN_ID"]),
                        fontsize=SMALL_TEXT_SIZE,
                        alpha=0.9 if edge_color != "C0" else 0.85,
                        color=edge_color,
                    )
                )

    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
    )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(handles=legend_handles_single, frameon=False, fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()

tsne_plot_name = "tsne_multi_panel.png" if MULTI_PANEL else "tsne_single_panel.png"
fig.savefig(PLOTS_DIR / tsne_plot_name, dpi=200, bbox_inches="tight")

hist_df = plot_df.loc[colorbar_mask, cols].copy()
if not hist_df.empty:
    fig_hist, axes_hist = plt.subplots(2, 3, figsize=(ICML_DOUBLE_COLUMN_WIDTH, 4.6))
    axes_hist = axes_hist.ravel()

    for ax, feature in zip(axes_hist, cols):
        values = pd.to_numeric(hist_df[feature], errors="coerce")
        values = values[np.isfinite(values)]

        if values.empty:
            ax.set_visible(False)
            continue

        ax.hist(values, bins=20, color="0.35", alpha=0.85, edgecolor="white")
        percentile_specs = [
            (0.1, "--", "tab:blue"),
            (0.2, ":", "tab:green"),
            (0.8, ":", "tab:orange"),
            (0.9, "--", "tab:red"),
        ]
        for quantile, linestyle, color in percentile_specs:
            ax.axvline(
                values.quantile(quantile),
                color=color,
                linestyle=linestyle,
                linewidth=1.4,
                alpha=0.95,
            )
        ax.axvline(
            values.median(),
            color="black",
            linestyle="-",
            linewidth=2.2,
            alpha=0.95,
        )
        ax.set_xlabel(display_labels[feature])
        ax.set_ylabel("Count")

        if feature == "f":
            ax.set_xlim(-np.pi, np.pi)

    fig_hist.tight_layout()
    fig_hist.savefig(PLOTS_DIR / "histogram.png", dpi=200, bbox_inches="tight")

    model_df = hist_df.apply(pd.to_numeric, errors="coerce").dropna().copy()
    representative_params, representative_label = get_representative_params(
        model_df,
        REPRESENTATIVE_METHOD,
        cols,
    )
    print(f"Representative method: {REPRESENTATIVE_METHOD}")
    for col in cols:
        print(f"{col} = {float(representative_params[col]):.8g}")
    x_model = np.linspace(0.0, 15.0, 400)

    fig_model, axes_model = plt.subplots(2, 3, figsize=(ICML_DOUBLE_COLUMN_WIDTH, 3.9))
    axes_model = axes_model.ravel()
    cmap_model = colormaps["viridis"]

    for idx, (ax, feature) in enumerate(zip(axes_model, cols)):
        values = pd.to_numeric(model_df[feature], errors="coerce")
        values = values[np.isfinite(values)]

        if values.empty:
            ax.set_visible(False)
            continue

        vmin = values.min()
        vmax = values.max()
        if np.isclose(vmin, vmax):
            vmin -= 1e-9
            vmax += 1e-9

        norm = Normalize(vmin=vmin - 1e-12, vmax=vmax + 1e-12)

        for _, params in model_df.iterrows():
            y_model = normalized_model(x_model, params)
            ax.plot(
                x_model,
                y_model,
                color=cmap_model(norm(float(params[feature]))),
                lw=1.0,
                alpha=0.28,
            )

        representative_curve = normalized_model(x_model, representative_params)
        ax.plot(
            x_model,
            representative_curve,
            color="black",
            lw=3.2,
            alpha=0.98,
            zorder=10,
        )

        row_idx, col_idx = divmod(idx, 3)
        if row_idx == 1:
            ax.set_xlabel("t (days since explosion)")
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)
        if col_idx == 0:
            ax.set_ylabel("Normalized model")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlim(x_model.min(), x_model.max())
        ax.set_ylim(-0.1, 1.1)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_model)
        mappable.set_array([])
        cax = inset_axes(ax, width="36%", height="6%", loc="lower right", borderpad=1.6)
        cbar = fig_model.colorbar(mappable, cax=cax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=TICK_FONT_SIZE - 1, pad=1, length=2)
        cbar.set_label(display_labels[feature], fontsize=TICK_FONT_SIZE, labelpad=1)
        cbar.ax.xaxis.set_label_position("top")

    fig_model.tight_layout(pad=0.5, w_pad=0.3, h_pad=0.4)
    fig_model.savefig(PLOTS_DIR / "ensemble_lc.png", dpi=200, bbox_inches="tight")

    fig_model_single, ax_model_single = plt.subplots(figsize=(ICML_SINGLE_COLUMN_WIDTH, 3))
    for _, params in model_df.iterrows():
        y_model = normalized_model(x_model, params)
        ax_model_single.plot(
            x_model,
            y_model,
            color="black",
            lw=1.0,
            alpha=0.05,
        )

    representative_curve_single = normalized_model(x_model, representative_params)
    ax_model_single.plot(
        x_model,
        representative_curve_single,
        color="black",
        lw=3.2,
        alpha=0.98,
        zorder=10,
        label='KDE Peak Sample',#representative_label.title(),
    )
    t2_fit_mask = (x_model >= 0.0) & (x_model <= 7.0)
    t2_plot_mask = (x_model >= 0.0) & (x_model <= 12.0)
    if np.count_nonzero(t2_fit_mask) > 0:
        t_fit = x_model[t2_fit_mask]
        y_fit = representative_curve_single[t2_fit_mask]
        amp_t2, t0_t2 = fit_shifted_t_squared(t_fit, y_fit, t0_init=0.0)
        t_plot = x_model[t2_plot_mask]
        y_t2 = amp_t2 * (t_plot - t0_t2) ** 2
        ax_model_single.plot(
            t_plot,
            y_t2,
            color="red",
            lw=2.0,
            linestyle=":",
            zorder=12,
            label=r"Empirical $t^2$ law",
        )
    ax_model_single.set_xlim(x_model.min(), x_model.max())
    ax_model_single.set_ylim(-0.1, 1.1)
    ax_model_single.set_xlabel("t (days since explosion)")
    ax_model_single.set_ylabel("Normalized model")
    ax_model_single.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
    fig_model_single.tight_layout()
    fig_model_single.savefig(
        PLOTS_DIR / "ensemble_lc_single_panel.png",
        dpi=200,
        bbox_inches="tight",
    )

    training_info_df = pd.read_csv("../TESS_Data/sne_info.csv")
    compare_targets = {
        "2018agk": "tab:blue",
        "2018oh": "tab:orange",
    }
    fig_compare, ax_compare = plt.subplots(figsize=(ICML_SINGLE_COLUMN_WIDTH, 3))
    x_compare = np.linspace(-3.0, 15.0, 450)

    representative_total = normalized_model(x_compare, representative_params)
    representative_quad = normalized_quadratic(x_compare, representative_params)
    representative_sine = normalized_sine(x_compare, representative_params)
    ax_compare.plot(x_compare, representative_total, color="black", lw=2.8)
    ax_compare.plot(x_compare, representative_quad, color="black", lw=2.0, linestyle="--")
    ax_compare.plot(x_compare, representative_sine, color="black", lw=2.0, linestyle=":")

    for sn_id, color in compare_targets.items():
        fit_row = tess_df.loc[tess_df["SN_ID"] == sn_id]
        info_row = training_info_df.loc[training_info_df["object"] == sn_id]
        curve_path = TRAINING_CURVE_DIR / f"{sn_id}_binned.csv"
        if fit_row.empty or info_row.empty or not curve_path.exists():
            continue

        fit_row = fit_row.iloc[0]
        info_row = info_row.iloc[0]
        curve_df = pd.read_csv(curve_path)

        t_rel = pd.to_numeric(curve_df["time"], errors="coerce") - float(info_row["t_exp_tjd"])
        y_norm = (
            pd.to_numeric(curve_df["flux"], errors="coerce") - float(fit_row["baseline"])
        ) / float(fit_row["peak_minus_baseline"])

        valid = np.isfinite(t_rel) & np.isfinite(y_norm)
        t_rel = t_rel[valid]
        y_norm = y_norm[valid]
        if sn_id == "2018oh":
            t_rel = t_rel #/ 1.15
        in_window = (t_rel >= -3.0) & (t_rel <= 15.0)
        ax_compare.scatter(
            t_rel[in_window],
            y_norm[in_window],
            s=18,
            alpha=0.45,
            color=color,
            zorder=6 if sn_id == "2018oh" else 4,
        )

        params = fit_row[cols].apply(pd.to_numeric, errors="coerce")
        x_eval = x_compare * 1 if sn_id == "2018oh" else x_compare
        total = normalized_model(x_eval, params)
        quad = normalized_quadratic(x_eval, params)
        sine = normalized_sine(x_eval, params)
        zorder = 7 if sn_id == "2018oh" else 5
        ax_compare.plot(x_compare, total, color=color, lw=2.6, zorder=zorder)
        ax_compare.plot(x_compare, quad, color=color, lw=1.8, linestyle="--", zorder=zorder)
        ax_compare.plot(x_compare, sine, color=color, lw=1.8, linestyle=":", zorder=zorder)

    representative_legend_label = format_representative_label(representative_label)
    color_handles = [
        Line2D([0], [0], color="black", lw=2.8, label=representative_legend_label),
        Line2D([0], [0], color="tab:orange", lw=2.6, label="SN 2018oh"),
        Line2D([0], [0], color="tab:blue", lw=2.6, label="SN 2018agk"),
    ]
    style_handles = [
        Line2D([0], [0], color="0.25", lw=2.6, linestyle="-", label="SR model"),
        Line2D([0], [0], color="0.25", lw=2.0, linestyle="--", label="Polynomial"),
        Line2D([0], [0], color="0.25", lw=2.0, linestyle=":", label="Sinusoidal"),
        Line2D([0], [0], marker="o", color="0.25", linestyle="None", markersize=6, alpha=0.45, label="Data"),
    ]

    ax_compare.set_xlim(0.0, 15.0)
    ax_compare.set_ylim(-0.2, 1.1)
    ax_compare.set_xlabel("t (days since explosion)")
    ax_compare.set_ylabel("Normalized flux")
    combined_handles = color_handles + style_handles
    ax_compare.legend(
        handles=combined_handles,
        loc="upper left",
        ncol=1,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )
    fig_compare.tight_layout()
    fig_compare.savefig(PLOTS_DIR / "median_model_vs_2018oh_2018agk.png", dpi=200, bbox_inches="tight")

corner_df_all = plot_df[cols].apply(pd.to_numeric, errors="coerce").dropna().copy()
if not corner_df_all.empty:
    fig_corner_all = corner.corner(
        corner_df_all.to_numpy(),
        labels=[display_labels[col] for col in cols],
        bins=25,
        show_titles=True,
        title_fmt=".3g",
        plot_density=True,
        plot_datapoints=True,
        fill_contours=True,
    )
    fig_corner_all.set_size_inches(ICML_DOUBLE_COLUMN_WIDTH, ICML_DOUBLE_COLUMN_WIDTH)
    medians_all = corner_df_all.median()
    axes_all = np.array(fig_corner_all.axes).reshape((len(cols), len(cols)))
    for i, col in enumerate(cols):
        axes_all[i, i].axvline(medians_all[col], color="black", linestyle="--", linewidth=1.4, alpha=0.9)
        for j in range(i):
            axes_all[i, j].axvline(medians_all[cols[j]], color="black", linestyle="--", linewidth=1.0, alpha=0.7)
            axes_all[i, j].axhline(medians_all[col], color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    fig_corner_all.savefig(PLOTS_DIR / "tsne_corner_all_targets.png", dpi=200, bbox_inches="tight")

corner_df_mask = plot_df.loc[colorbar_mask, cols].apply(pd.to_numeric, errors="coerce").dropna().copy()
if not corner_df_mask.empty:
    fig_corner_mask = corner.corner(
        corner_df_mask.to_numpy(),
        labels=[display_labels[col] for col in cols],
        bins=25,
        show_titles=True,
        title_fmt=".3g",
        plot_density=True,
        plot_datapoints=True,
        fill_contours=True,
    )
    fig_corner_mask.set_size_inches(ICML_DOUBLE_COLUMN_WIDTH, ICML_DOUBLE_COLUMN_WIDTH)
    medians_mask = corner_df_mask.median()
    axes_mask = np.array(fig_corner_mask.axes).reshape((len(cols), len(cols)))
    for i, col in enumerate(cols):
        axes_mask[i, i].axvline(medians_mask[col], color="black", linestyle="--", linewidth=1.4, alpha=0.9)
        for j in range(i):
            axes_mask[i, j].axvline(medians_mask[cols[j]], color="black", linestyle="--", linewidth=1.0, alpha=0.7)
            axes_mask[i, j].axhline(medians_mask[col], color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    fig_corner_mask.savefig(PLOTS_DIR / "correlation_corner_plot.png", dpi=200, bbox_inches="tight")

plt.show()
