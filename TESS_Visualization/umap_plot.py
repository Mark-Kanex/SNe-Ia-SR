from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from adjustText import adjust_text
import umap.umap_ as umap

MULTI_PANEL = True
SNR_FACTOR = 0
LIMIT_MULTI_PANEL_COLORBAR = True
FIX_F_RANGE = True

BASE_DIR = Path(__file__).resolve().parent
VALIDATION_CURVE_DIR = BASE_DIR / "catalog_split"
TRAINING_CURVE_DIR = BASE_DIR.parent / "TESS_Data"


def wrap_phase_to_pi(values):
    values = pd.to_numeric(values, errors="coerce")
    return ((values + np.pi) % (2.0 * np.pi)) - np.pi


def get_wrapped_phase_series(df, column="f"):
    original = pd.to_numeric(df[column], errors="coerce")
    wrapped = wrap_phase_to_pi(original)
    return wrapped.mask(np.isclose(wrapped, -np.pi) & (original > 0), np.pi)


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

tess_df = pd.read_csv("../TESS_Data/fit_curve_window_results.csv")
tess_df = tess_df[tess_df["SN_ID"] != "2020tld"].copy()
tess_df["source"] = "Training"
tess_df["high_chi2"] = False
tess_df["high_r2"] = False

if FIX_F_RANGE:
    validation_df["f"] = get_wrapped_phase_series(validation_df)
    tess_df["f"] = get_wrapped_phase_series(tess_df)

cols = ["a_norm", "b_norm", "c_norm", "d_norm", "e", "f"]
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


embedding = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42,
).fit_transform(plot_df[cols].to_numpy())

plot_df["UMAP1"] = embedding[:, 0]
plot_df["UMAP2"] = embedding[:, 1]

colorbar_mask = (
    ((plot_df["UMAP1"] > 6) & (plot_df["UMAP2"] > 5))
    | ((plot_df["UMAP1"] <= 10.5) & (plot_df["UMAP1"] >= 8) & (plot_df["UMAP2"] <= 5))
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
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    cmap = "viridis"

    for ax, feature in zip(axes, cols):
        mask = colorbar_mask if LIMIT_MULTI_PANEL_COLORBAR else None
        vmin, vmax = get_colorbar_limits(plot_df[feature], mask=mask)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = colormaps[cmap]
        texts = []
        all_x = plot_df["UMAP1"].to_numpy()
        all_y = plot_df["UMAP2"].to_numpy()

        for source in ["Training", "Validation"]:
            source_df = plot_df[plot_df["source"] == source]
            feature_values = pd.to_numeric(source_df[feature], errors="coerce").to_numpy()
            if source == "Training":
                ax.scatter(
                    source_df["UMAP1"],
                    source_df["UMAP2"],
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
                            row["UMAP1"] + 1.0,
                            row["UMAP2"] + 0.6,
                            row["SN_ID"],
                            fontsize=7,
                            alpha=0.85,
                            color=cmap_obj(norm(float(row[feature]))),
                        )
                    )
            else:
                edge_colors = cmap_obj(norm(feature_values))
                ax.scatter(
                    source_df["UMAP1"],
                    source_df["UMAP2"],
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

        ax.set_title(feature)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax)
        cbar.set_label(feature)

    fig.legend(handles=legend_handles_multi, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("UMAP of normalized a,b,c,d and e,f", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
else:
    plt.figure(figsize=(8, 6))
    texts = []
    for source in ["Training", "Validation"]:
        source_df = plot_df[plot_df["source"] == source]
        if source == "Training":
            low_snr_df = source_df[source_df["low_snr"]]
            normal_df = source_df[~source_df["low_snr"]]
            plt.scatter(
                normal_df["UMAP1"],
                normal_df["UMAP2"],
                s=90,
                alpha=0.85,
                color="black",
                marker=marker_map[source],
                edgecolors="none",
                label=source,
            )
            plt.scatter(
                low_snr_df["UMAP1"],
                low_snr_df["UMAP2"],
                s=90,
                alpha=0.9,
                color="red",
                marker=marker_map[source],
                edgecolors="none",
            )
            for _, row in normal_df.iterrows():
                texts.append(
                    plt.text(
                        row["UMAP1"] + 0.6,
                        row["UMAP2"] + 0.6,
                        row["SN_ID"],
                        fontsize=7,
                        alpha=0.85,
                        color="black",
                    )
                )
            for _, row in low_snr_df.iterrows():
                texts.append(
                    plt.text(
                        row["UMAP1"] + 0.6,
                        row["UMAP2"] + 0.6,
                        row["SN_ID"],
                        fontsize=7,
                        alpha=0.9,
                        color="red",
                    )
                )
        else:
            first_validation = True
            for _, row in source_df.iterrows():
                edge_color = get_point_color(row, "C0")
                plt.scatter(
                    [row["UMAP1"]],
                    [row["UMAP2"]],
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
                        row["UMAP1"] + 0.6,
                        row["UMAP2"] + 0.6,
                        row["SN_ID"],
                        fontsize=7,
                        alpha=0.9 if edge_color != "C0" else 0.85,
                        color=edge_color,
                    )
                )

    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
    )

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title("UMAP of normalized a,b,c,d and e,f")
    plt.legend(handles=legend_handles_single, frameon=False)
    plt.tight_layout()

plt.show()
