import os
import json
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
os.environ.setdefault("JULIA_NUM_THREADS", "4")
from pysr import PySRRegressor
from sympy import sympify, Symbol, lambdify, preorder_traversal, sin, exp
DEF_MAX_COMPONENTS = 2
DEF_DATA_DIR = "TESS_Data"
DEF_OUT_DIR = "TESS_Symmetry_Informed_Residual_Symbolic_Regression"
DEF_NITER = 50
DEF_MAXSIZE = 25
DEF_MAXDEPTH = 12
DEF_NCYCLES_PER_ITERATION = 50
DEF_OPERATOR_COMPLEXITIES = {
    "+": 1,
    "-": 1,
    "*": 2,
    "/": 3,
    "^": 4,
    "sin": 5,
    "exp": 5,
}
DEF_EQUIV_DUPLICATES = 100
DEF_EQUIV_A_MIN = 0.5
DEF_EQUIV_A_MAX = 5.0
DEF_EQUIV_SEED = None
DEF_EQUIV_WINDOW_SCALE_MIN = 0.90
DEF_EQUIV_WINDOW_SCALE_MAX = 1.10

def find_explosion_epoch(time_values: np.ndarray, flux_values: np.ndarray, flux_errors: np.ndarray) -> float:
    explosion_time = float(time_values[0])
    for index in range(len(time_values)):
        if np.all(flux_values[index:] > 5 * flux_errors[index:]) and np.all(flux_values[index:] > 0.0):
            explosion_time = float(time_values[index])
            break
    return explosion_time


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


def bootstrap_adjacent_triplets(times: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
                                max_span: float = 0.5, sigma: float = 0.01,
                                seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if times.size < 3:
        return times, flux, flux_err
    t_out = times.copy().astype(float)
    f_out = flux.copy().astype(float)
    e_out = flux_err.copy()
    rng = np.random.default_rng(seed)
    i = 1
    while i < times.size - 1:
        idx0, idx1, idx2 = i - 1, i, i + 1
        t0, t1, t2 = times[idx0], times[idx1], times[idx2]
        if (np.isfinite([t0, t1, t2]).all()
                and np.isfinite([flux[idx0], flux[idx1], flux[idx2]]).all()
                and (t2 - t0) <= max_span):
            V = np.array([[t0*t0, t0, 1.0],
                          [t1*t1, t1, 1.0],
                          [t2*t2, t2, 1.0]], dtype=float)
            y = np.array([flux[idx0], flux[idx1], flux[idx2]], dtype=float)
            try:
                a, b, c = np.linalg.solve(V, y)
            except np.linalg.LinAlgError:
                i += 1
                continue
            low, high = min(t0, t2), max(t0, t2)
            new_ts = np.array([t0, t1, t2]) + rng.normal(0.0, sigma, size=3)
            new_ts = np.clip(new_ts, low, high)
            new_ts.sort()
            t_out[idx0:idx2+1] = new_ts
            f_out[idx0:idx2+1] = a*new_ts**2 + b*new_ts + c
            i += 3
        else:
            i += 1
    return t_out, f_out, e_out


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


def get_plot_window(base_name: str, time_since_explosion: np.ndarray, flux_values: np.ndarray | None = None) -> tuple[float, float, float, float]:
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

def generate_symmetry_informed_dataset(time_since_t_exp: np.ndarray, flux_values: np.ndarray,
                                 flux_errors: np.ndarray, plot_mask: np.ndarray,
                                 default_window_max: float,
                                 duplicates: int, t0_min: float, t0_max: float,
                                 a_min: float, a_max: float, seed: int | None) -> pd.DataFrame:
    base_times_rel = time_since_t_exp[plot_mask]
    base_flux = flux_values[plot_mask]
    base_err = flux_errors[plot_mask]
    if base_times_rel.size == 0:
        return pd.DataFrame(columns=["time", "flux", "flux_err", "t_0", "amplitude", "window_scale", "duplicate_id"])
    plot_rel_min = 0.0
    plot_rel_max = float(default_window_max)
    if not np.isfinite(plot_rel_max) or plot_rel_max <= plot_rel_min:
        plot_rel_max = float(np.nanmax(base_times_rel)) if base_times_rel.size else 1.0

    rng = np.random.default_rng(seed)
    out_time = []
    out_flux = []
    out_t0 = []
    out_a = []
    out_scale = []
    out_err = []
    out_dup = []
    out_window_min = []
    out_window_max = []
    t0_sigma = (t0_max - t0_min) / 4.0 if t0_max > t0_min else 0.0
    scale_sigma = 0.05
    for dup_idx in range(int(duplicates)):
        t0 = float(np.clip(rng.normal(0.0, t0_sigma), t0_min, t0_max)) if t0_sigma > 0 else 0.0
        a = float(rng.uniform(a_min, a_max))
        window_scale = float(np.clip(rng.normal(1.0, scale_sigma), DEF_EQUIV_WINDOW_SCALE_MIN, DEF_EQUIV_WINDOW_SCALE_MAX))
        target_min = plot_rel_min
        target_max = plot_rel_max * window_scale
        source_mask = (
            np.isfinite(base_times_rel)
            & np.isfinite(base_flux)
            & np.isfinite(base_err)
        )
        if not np.any(source_mask):
            continue
        shifted_rel = base_times_rel[source_mask] - t0
        valid_rel = np.isfinite(shifted_rel) & (shifted_rel >= target_min) & (shifted_rel <= target_max)
        if not np.any(valid_rel):
            continue
        shifted_rel_valid = shifted_rel[valid_rel]
        scaled_flux = a * base_flux[source_mask][valid_rel]
        scaled_err = np.abs(a) * base_err[source_mask][valid_rel]
        out_time.append(shifted_rel_valid)
        out_flux.append(scaled_flux)
        out_err.append(scaled_err)
        out_t0.append(np.full(shifted_rel_valid.size, t0, dtype=float))
        out_a.append(np.full(shifted_rel_valid.size, a, dtype=float))
        out_scale.append(np.full(shifted_rel_valid.size, window_scale, dtype=float))
        out_dup.append(np.full(shifted_rel_valid.size, dup_idx, dtype=int))
        out_window_min.append(np.full(shifted_rel_valid.size, plot_rel_min, dtype=float))
        out_window_max.append(np.full(shifted_rel_valid.size, plot_rel_max, dtype=float))

    if not out_time:
        return pd.DataFrame(columns=["time", "flux", "flux_err", "t_0", "amplitude", "window_scale", "window_min", "window_max", "duplicate_id"])

    time_concat = np.concatenate(out_time)
    flux_concat = np.concatenate(out_flux)
    t0_concat = np.concatenate(out_t0)
    a_concat = np.concatenate(out_a)
    scale_concat = np.concatenate(out_scale)
    err_concat = np.concatenate(out_err)
    dup_concat = np.concatenate(out_dup)
    window_min_concat = np.concatenate(out_window_min)
    window_max_concat = np.concatenate(out_window_max)
    return pd.DataFrame({
        "time": time_concat,
        "flux": flux_concat,
        "flux_err": err_concat,
        "t_0": t0_concat,
        "amplitude": a_concat,
        "window_scale": scale_concat,
        "window_min": window_min_concat,
        "window_max": window_max_concat,
        "duplicate_id": dup_concat,
    })


def fit_symbolic_regression(feature_matrix: np.ndarray, target: np.ndarray,
                            niterations: int, maxsize: int, maxdepth: int, ncycles_per_iteration: int,
                            variable_names: list[str] | None = None) -> tuple[PySRRegressor, np.ndarray, str]:
    model = PySRRegressor(
        niterations=niterations,
        populations=24,
        ncycles_per_iteration=ncycles_per_iteration,
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sin", "exp"],
        nested_constraints={"sin": {"sin": 0, "exp": 0}, "exp": {"sin": 0, "exp": 0}},
        constraints={"^": (-1, 1)},
        elementwise_loss="L2DistLoss()",
        model_selection="best",
        maxsize=maxsize,
        parsimony=0.0001,
        batching=True,
        batch_size=500,
        weight_optimize=0.01,
        turbo=False,
        complexity_of_operators=DEF_OPERATOR_COMPLEXITIES,
        maxdepth=maxdepth,
        random_state=0,
        deterministic=False,
        parallelism="multithreading",
        extra_sympy_mappings={},
        variable_names=variable_names,
    )
    try:
        model.fit(feature_matrix, target, variable_names=variable_names)
    except TypeError:
        model.fit(feature_matrix, target)
    predicted = model.predict(feature_matrix)
    equation = None
    try:
        best_result = model.get_best()
        if hasattr(best_result, 'get'):
            equation = best_result.get('sympy_equation', best_result.get('equation', None))
        if equation is None and hasattr(best_result, 'sympy_equation'):
            equation = best_result.sympy_equation
        if equation is None and hasattr(model, 'sympy'):
            equation = model.sympy()
        equation = str(equation) if equation is not None else '<equation unavailable>'
    except Exception:
        try:
            equation = str(model.sympy())
        except Exception:
            equation = '<equation unavailable>'
    return model, predicted, equation


def count_operator_stats(equations: list[str]) -> tuple[int, int]:
    terms = [str(eq) for eq in equations if eq]
    if not terms:
        return 0, 0
    try:
        expr = sympify("+".join(terms))
    except Exception:
        return 0, 0
    from sympy import Integer as _Int
    op_count = 0
    complexity = 0
    for node in preorder_traversal(expr):
        if node.is_Add:
            inc = max(len(node.args) - 1, 1)
            op_count += inc
            complexity += DEF_OPERATOR_COMPLEXITIES["+"] * inc
        elif node.is_Mul:
            div_args = [a for a in node.args if a.is_Pow and a.exp == _Int(-1)]
            mul_count = max(len(node.args) - 1 - len(div_args), 0)
            op_count += mul_count + len(div_args)
            complexity += (DEF_OPERATOR_COMPLEXITIES["*"] * mul_count
                           + DEF_OPERATOR_COMPLEXITIES["/"] * len(div_args))
        elif node.is_Pow:
            if node.exp != _Int(-1):
                if node.exp.is_Integer and int(node.exp) >= 2:
                    n = int(node.exp)
                    op_count += (n - 1)
                    complexity += (n - 1) * DEF_OPERATOR_COMPLEXITIES["*"]
                elif not node.exp.is_Integer:
                    op_count += 1
                    complexity += DEF_OPERATOR_COMPLEXITIES["^"]
        elif node.func is sin:
            op_count += 1
            complexity += DEF_OPERATOR_COMPLEXITIES["sin"]
        elif node.func is exp:
            op_count += 1
            complexity += DEF_OPERATOR_COMPLEXITIES["exp"]
        elif node.is_Atom:
            complexity += 1
    return op_count, complexity


def bic_from_mse(mse: float, n_points: int, k_complexity: int) -> float:
    if n_points <= 0:
        return float("inf")
    mse_f = float(mse)
    if not math.isfinite(mse_f) or mse_f < 0:
        return float("inf")
    mse_safe = max(mse_f, 1e-12)
    k_val = max(int(k_complexity), 1)
    n = float(n_points)
    return math.log(mse_safe) + float(k_val) * math.log(n) / n



def symmetry_informed_symbolic_regression_for_file(path: str, out_dir: str, niterations: int, maxsize: int,
                                             maxdepth: int, ncycles_per_iteration: int, duplicates: int,
                                             a_min: float, a_max: float, seed: int | None,
                                             make_plots: bool = True) -> dict:
    base_name = os.path.splitext(os.path.basename(path))[0]
    core_name = base_name.replace("_binned", "").lower()
    data_frame = read_csv_any(path).dropna(subset=["time", "flux"])
    if data_frame.empty:
        return {"file": base_name, "status": "empty"}

    plots_dir = os.path.join(out_dir, "plots")
    csvs_dir = os.path.join(out_dir, "csv")
    jsons_dir = os.path.join(out_dir, "json")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(csvs_dir, exist_ok=True)
    os.makedirs(jsons_dir, exist_ok=True)
    time_values = data_frame["time"].to_numpy()
    flux_values = data_frame["flux"].to_numpy()
    flux_errors = data_frame["flux_err"].to_numpy()
    time_values, flux_values, flux_errors = bootstrap_adjacent_triplets(
        time_values, flux_values, flux_errors)
    explosion_time = find_explosion_epoch(time_values, flux_values, flux_errors)

    time_since_explosion = time_values - explosion_time
    plot_start, plot_end, left_margin, right_margin = get_plot_window(base_name, time_since_explosion, flux_values)
    t_exp = explosion_time + plot_start
    plot_start_rel = 0.0
    plot_end_rel = plot_end - plot_start
    time_since_t_exp = time_since_explosion - plot_start
    fit_mask = (time_since_t_exp >= plot_start_rel - left_margin) & (time_since_t_exp <= plot_end_rel + right_margin)

    # 2020tld has a flux spike at t≈59107.876 TJD.
    # including it forces the regression into rapid oscillation to accommodate the outlier.
    # This point is therefore excluded for regression purposes.

    if core_name == "2020tld":
        fit_mask = fit_mask & (np.abs(time_values - 59107.8756786985) > 0.1)

    window_span = float(plot_end_rel - plot_start_rel)
    t0_window = 0.10 * window_span if window_span > 0 else 0.0
    t0_min_local = -t0_window
    t0_max_local = t0_window

    symmetry_frame = generate_symmetry_informed_dataset(
        time_since_t_exp,
        flux_values,
        flux_errors,
        fit_mask,
        plot_end_rel,
        duplicates,
        t0_min_local,
        t0_max_local,
        a_min,
        a_max,
        seed,
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_output_path = os.path.join(csvs_dir, f"{base_name}_symmetry_informed.csv")
    symmetry_frame.to_csv(csv_output_path, index=False)

    if symmetry_frame.empty:
        return {"file": base_name, "status": "no_symmetry_points", "csv": csv_output_path}

    features = np.column_stack([
        symmetry_frame["time"].to_numpy(),
        symmetry_frame["t_0"].to_numpy(),
        symmetry_frame["amplitude"].to_numpy(),
    ])
    target = symmetry_frame["flux"].to_numpy()

    components = []
    component_preds = []
    equations = []
    residual = target.copy()
    bic_by_order = []
    prev_bic = float("inf")
    last_rejected_bic = None
    last_rejected_index = None
    component_index = 0
    while True:
        if component_index >= DEF_MAX_COMPONENTS:
            break
        model, predicted, equation = fit_symbolic_regression(
            features,
            residual,
            niterations,
            maxsize,
            maxdepth,
            ncycles_per_iteration,
            variable_names=["t", "t_0", "A"],
        )
        new_residual = residual - predicted
        mse_new = float(np.nanmean(np.square(new_residual))) if new_residual.size else float("inf")
        _, k_complexity = count_operator_stats(equations + [equation])
        bic_new = bic_from_mse(mse_new, int(target.size), int(k_complexity))
        if np.isfinite(prev_bic):
            print(f"  comp{component_index + 1} BIC improvement: {prev_bic - bic_new:.6f}")
        else:
            print(f"  comp{component_index + 1} BIC improvement: inf")
        print(f"  comp{component_index + 1} BIC (per-point): {bic_new:.6f}")
        if (
            component_index >= 1
            and np.isfinite(prev_bic)
            and bic_new >= prev_bic
        ):
            last_rejected_bic = bic_new
            last_rejected_index = component_index + 1
            break
        components.append(model)
        component_preds.append(predicted)
        equations.append(equation)
        bic_by_order.append(bic_new)
        residual = new_residual
        prev_bic = bic_new
        component_index += 1

    cumulative_preds = []
    running = np.zeros_like(target, dtype=float)
    for pred in component_preds:
        running = running + pred
        cumulative_preds.append(running.copy())

    order_r2 = []
    cumulative_bic_by_order = []
    for i, y_pred in enumerate(cumulative_preds):
        mask = np.isfinite(target) & np.isfinite(y_pred)
        if np.count_nonzero(mask) >= 2:
            yt = target[mask]
            yp = y_pred[mask]
            sse = float(np.sum((yt - yp) ** 2))
            sst = float(np.sum((yt - np.mean(yt)) ** 2))
            r2o = 1.0 - (sse / sst) if sst > 0 else float("nan")
            mse_cum = float(np.nanmean(np.square(yt - yp)))
            _, k_cum = count_operator_stats(equations[: i + 1])
            bic_cum = bic_from_mse(mse_cum, int(np.count_nonzero(mask)), k_cum)
        else:
            r2o = float("nan")
            bic_cum = float("inf")
        order_r2.append(r2o)
        cumulative_bic_by_order.append(bic_cum)

    components_out = {"time": symmetry_frame["time"].to_numpy(), "flux": target}
    for i, pred in enumerate(component_preds, start=1):
        components_out[f"comp{i}"] = pred
    components_out["model"] = cumulative_preds[-1] if cumulative_preds else np.zeros_like(target, dtype=float)
    components_out["residual"] = residual
    components_df = pd.DataFrame(components_out)
    components_csv_path = os.path.join(csvs_dir, f"{base_name}_symmetry_components.csv")
    components_df.to_csv(components_csv_path, index=False)

    summary = {
        "file": base_name,
        "t_exp": float(t_exp),
        "n_components": int(len(equations)),
        "equations": equations,
        "bic_by_order": bic_by_order,
        "cumulative_bic_by_order": cumulative_bic_by_order,
        "bic_last_rejected": last_rejected_bic,
        "ic_last_rejected_index": last_rejected_index,
        "csv": csv_output_path,
        "components_csv": components_csv_path,
        "r2_by_order": order_r2,
        "duplicates": int(duplicates),
        "t0_range": [float(t0_min_local), float(t0_max_local)],
        "a_range": [float(a_min), float(a_max)],
    }
    summary_json_path = os.path.join(jsons_dir, f"{base_name}_symmetry_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    summary["summary_json"] = summary_json_path

    if make_plots:
        fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex=False, sharey=False, constrained_layout=False)
        axes_list = axes.flatten()
        for ax in axes_list:
            ax.grid(True, alpha=0.4)
            ax.tick_params(axis="both", labelbottom=True, labelleft=True)
        grouped = symmetry_frame.groupby("duplicate_id", sort=True)
        for ax, (dup_id, df_dup) in zip(axes_list, grouped):
            t0_val = float(df_dup["t_0"].iloc[0])
            a_val = float(df_dup["amplitude"].iloc[0])
            window_scale_val = float(df_dup["window_scale"].iloc[0])
            window_min_val = float(df_dup["window_min"].iloc[0])
            window_max_val = float(df_dup["window_max"].iloc[0])
            ax.errorbar(
                df_dup["time"],
                df_dup["flux"],
                yerr=df_dup["flux_err"],
                fmt="o",
                markersize=3,
                alpha=0.7,
                label="points",
            )
            if components:
                t_vals = df_dup["time"].to_numpy()
                features_dup = np.column_stack([
                    t_vals,
                    np.full_like(t_vals, t0_val, dtype=float),
                    np.full_like(t_vals, a_val, dtype=float),
                ])
                order = np.argsort(t_vals)
                cumulative = np.zeros_like(t_vals, dtype=float)
                for idx, model in enumerate(components, start=1):
                    cumulative = cumulative + model.predict(features_dup)
                    ax.plot(t_vals[order], cumulative[order], linewidth=1.3, label=f"model {idx}")
            ax.set_title(f"dup {int(dup_id)}: t_0={t0_val:.2f}, A={a_val:.2f}, scale={window_scale_val:.3f}")
            ax.set_xlim(window_min_val, window_max_val)
            ax.legend(fontsize=8)
        for ax in axes_list[len(grouped):]:
            ax.axis("off")
        fig.suptitle(f"{base_name} — Symmetry-informed shifted/scaled duplicates", fontsize=12)
        fig.supxlabel("Time since T_exp (days)")
        fig.supylabel("Flux")
        fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.92, bottom=0.08)
        plot_output_path = os.path.join(plots_dir, f"{base_name}_symmetry_duplicates.png")
        plt.savefig(plot_output_path, dpi=300, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)
        summary["plot"] = plot_output_path

    return summary


def plot_all_rsr_components(plot_paths: list[str], out_path: str) -> str | None:
    if not plot_paths:
        return None
    images = []
    titles = []
    for path in plot_paths:
        try:
            image = plt.imread(path)
        except Exception:
            continue
        images.append(image)
        name = os.path.splitext(os.path.basename(path))[0]
        titles.append(name.replace("_rsr_components", ""))
    if not images:
        return None
    count = len(images)
    ncols = 2 if count <= 4 else 3
    nrows = (count + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.0 * nrows), squeeze=False)
    axes_list = axes.flatten()
    for ax in axes_list[count:]:
        ax.axis("off")
    for ax, image, title in zip(axes_list, images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.05, hspace=0.08)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Symmetry-Informed Symbolic Regression (PySR) over binned TESS light curves")
    parser.add_argument("--data-dir", default=DEF_DATA_DIR)
    parser.add_argument("--out-dir", default=DEF_OUT_DIR)
    parser.add_argument("--niterations", type=int, default=DEF_NITER)
    parser.add_argument("--maxsize", type=int, default=DEF_MAXSIZE)
    parser.add_argument("--maxdepth", type=int, default=DEF_MAXDEPTH)
    parser.add_argument("--ncycles-per-iteration", type=int, default=DEF_NCYCLES_PER_ITERATION)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--equiv-duplicates", type=int, default=DEF_EQUIV_DUPLICATES)
    parser.add_argument("--equiv-a-min", type=float, default=DEF_EQUIV_A_MIN)
    parser.add_argument("--equiv-a-max", type=float, default=DEF_EQUIV_A_MAX)
    parser.add_argument("--equiv-seed", type=int, default=DEF_EQUIV_SEED)
    args = parser.parse_args()
    data_dir = args.data_dir
    out_dir = args.out_dir
    niterations = args.niterations
    maxsize = args.maxsize
    maxdepth = args.maxdepth
    ncycles_per_iteration = args.ncycles_per_iteration
    make_plots = not args.no_plots

    equiv_duplicates = int(args.equiv_duplicates)
    equiv_a_min = float(args.equiv_a_min)
    equiv_a_max = float(args.equiv_a_max)
    equiv_seed = int(args.equiv_seed) if args.equiv_seed is not None else None

    plots_dir = os.path.join(out_dir, "plots")
    if os.path.isdir(plots_dir):
        for name in os.listdir(plots_dir):
            path = os.path.join(plots_dir, name)
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except FileNotFoundError:
                pass

    files = [f for f in os.listdir(data_dir) if f.endswith("_binned.csv")]
    if not files:
        print("No *_binned.csv files found.")
        return
    iterator = tqdm(files, desc="RSR")
    results = []
    for file_name in iterator:
        try:
            path = os.path.join(data_dir, file_name)
            summary = symmetry_informed_symbolic_regression_for_file(
                path,
                out_dir,
                niterations,
                maxsize,
                maxdepth,
                ncycles_per_iteration,
                equiv_duplicates,
                equiv_a_min,
                equiv_a_max,
                equiv_seed,
                make_plots,
            )
            results.append(summary)
            print(f"\n{file_name}:")
            for i, equation in enumerate(summary.get("equations", []), start=1):
                print(f"  comp{i}: {equation}")
            if "plot" in summary:
                print(f"  plot: {summary['plot']}")
            print(f"  csv:  {summary['csv']}")
        except Exception as e:
            print(f"Error on {file_name}: {e}")
    if results:
        print("\n=== RSR Summary ===")
        for result in results:
            if not result or 'equations' not in result:
                continue
            print(f"\n{result.get('file', '<unknown>')}: t_exp={result.get('t_exp')}  components={result.get('n_components')}")
            for i, eq in enumerate(result.get('equations', []), start=1):
                print(f"  comp{i}: {eq}")
            print(f"  csv:  {result.get('csv')}")
            if 'plot' in result:
                print(f"  plot: {result.get('plot')}")
        all_summary = {
            "total_files": len(results),
            "n_components": {
                r["file"]: r.get("n_components", 0)
                for r in results if r and r.get("file")
            },
            "equations": {
                r["file"]: r.get("equations", [])
                for r in results if r and r.get("file")
            },
            "r2_by_order": {
                r["file"]: r.get("r2_by_order", [])
                for r in results if r and r.get("file")
            },
        }
        manifest = {
            "args": {
                "data_dir": data_dir,
                "out_dir": out_dir,
                "niterations": niterations,
                "maxsize": maxsize,
                "maxdepth": maxdepth,
                "ncycles_per_iteration": ncycles_per_iteration,
                "equiv_duplicates": equiv_duplicates,
                "equiv_a_min": equiv_a_min,
                "equiv_a_max": equiv_a_max,
                "equiv_seed": equiv_seed,
            },
            "all_summary": all_summary,
            "summaries": results,
        }
        jsons_dir = os.path.join(out_dir, "json")
        os.makedirs(jsons_dir, exist_ok=True)
        manifest_path = os.path.join(jsons_dir, "rsr_run_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nRun manifest: {manifest_path}")
        if make_plots:
            plots_dir = os.path.join(out_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            plot_paths = [r.get("plot") for r in results if r and r.get("plot")]
            combined_path = plot_all_rsr_components(plot_paths, os.path.join(plots_dir, "all_rsr_components.png"))
            if combined_path:
                print(f"\nCombined plot: {combined_path}")


if __name__ == "__main__":
    main()