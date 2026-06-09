import os
import sys
import json
import argparse
import subprocess
from collections import defaultdict, Counter
from typing import Dict, List

import math
import shutil
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares
from sympy import sympify, Symbol, Integer, expand, sin, cos, exp, lambdify
from sympy.core import numbers
from sympy.printing.str import sstr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_RUNS = 100
DEFAULT_RSR_NITER = 50
DEFAULT_RSR_MAXSIZE = 25
DEFAULT_RSR_MAXDEPTH = 12
DEFAULT_RSR_NCYCLES_PER_ITERATION = 50
DEFAULT_CLEAN_PYCACHE_EVERY = 10
VALIDATION_ROOT = SCRIPT_DIR
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, '.bootstrap_runs')
LIGHTCURVE_ROOT = os.path.join(SCRIPT_DIR, 'lightcurves')

DEFAULT_DATA_DIRECTORY = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'TESS_Data'))
RSR_SCRIPT_PATH = os.path.abspath(os.path.join(
    SCRIPT_DIR,
    '..',
    'TESS_Symmetry_Informed_Residual_Symbolic_Regression',
    'Symmetry_Informed_Residual_Symbolic_Regression.py',
))

OPERATOR_COMPLEXITY = {"+": 1, "-": 1, "*": 2, "/": 3, "sin": 5, "exp": 5}

WEIGHT_R2 = 1.0
WEIGHT_OCCURRENCE = 1.0
WEIGHT_COMPLEXITY = 0.10

DATA_VARIABLE_NAMES = {"time", "t", "x0"}


def is_data_variable(symbol: Symbol) -> bool:
    return symbol.name in DATA_VARIABLE_NAMES or symbol.name.startswith("x")


def get_data_symbols(expr) -> set[Symbol]:
    try:
        free = getattr(expr, "free_symbols", set())
    except Exception:
        return set()
    return {s for s in free if is_data_variable(s)}


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




def generalized_form_complexity(expr) -> float:
    if expr is None:
        return 0.0
    if expr.is_Add:
        return OPERATOR_COMPLEXITY.get("+", 1) * (len(expr.args) - 1) + sum(
            generalized_form_complexity(arg) for arg in expr.args
        )
    if expr.is_Mul:
        div_args = [a for a in expr.args if a.is_Pow and a.exp == Integer(-1)]
        mul_count = max(len(expr.args) - 1 - len(div_args), 0)
        return (OPERATOR_COMPLEXITY.get("*", 2) * mul_count
                + OPERATOR_COMPLEXITY.get("/", 3) * len(div_args)
                + sum(generalized_form_complexity(arg) for arg in expr.args))
    if expr.is_Pow:
        if expr.exp == Integer(-1):
            return generalized_form_complexity(expr.base)
        # No ^ operator: integer powers like x**2 are expressed as repeated multiplication
        if expr.exp.is_Integer and int(expr.exp) >= 2:
            n = int(expr.exp)
            return (n - 1) * OPERATOR_COMPLEXITY.get("*", 2) + n * generalized_form_complexity(expr.base)
        return (
            OPERATOR_COMPLEXITY.get("^", 4)
            + generalized_form_complexity(expr.base)
            + generalized_form_complexity(expr.exp)
        )
    if expr.func.__name__ == "sin":
        return OPERATOR_COMPLEXITY.get("sin", 5) + sum(
            generalized_form_complexity(arg) for arg in expr.args
        )
    if expr.func.__name__ == "exp":
        return OPERATOR_COMPLEXITY.get("exp", 5) + sum(
            generalized_form_complexity(arg) for arg in expr.args
        )
    args = getattr(expr, "args", ())
    if not args:
        return 1.0  # leaf node: variable or constant
    return sum(generalized_form_complexity(arg) for arg in args)


def load_lightcurve_window(base_name: str, data_dir: str) -> dict | None:
    path = os.path.join(data_dir, f"{base_name}.csv")
    if not os.path.exists(path):
        alt = os.path.join(data_dir, f"{base_name.replace('_binned', '')}_binned.csv")
        path = alt if os.path.exists(alt) else path
    if not os.path.exists(path):
        return None
    data_frame = read_csv_any(path).dropna(subset=["time", "flux"])
    if data_frame.empty:
        return None
    time_values = data_frame["time"].to_numpy(dtype=float)
    flux_values = data_frame["flux"].to_numpy(dtype=float)
    flux_errors = data_frame["flux_err"].to_numpy(dtype=float)
    flux_errors = np.where(np.isfinite(flux_errors), flux_errors, 1.0)
    if time_values.size < 2:
        return None
    explosion_time = find_explosion_epoch(time_values, flux_values, flux_errors)
    time_since_explosion = time_values - explosion_time
    plot_start, plot_end, left_margin, right_margin = get_plot_window(base_name, time_since_explosion, flux_values)
    if plot_start is not None:
        plot_start_rel = 0.0
        plot_end_rel = plot_end - plot_start
        fit_start = plot_start_rel - left_margin
        fit_end = plot_end_rel + right_margin
        x0 = time_since_explosion - plot_start
        plot_mask = (x0 >= plot_start_rel) & (x0 <= plot_end_rel)
        fit_mask = (x0 >= fit_start) & (x0 <= fit_end)
    else:
        x0_min = float(np.nanmin(time_since_explosion))
        x0_max = float(np.nanmax(time_since_explosion))
        plot_start_rel = 0.0
        plot_end_rel = x0_max - x0_min
        x0 = time_since_explosion - x0_min
        plot_mask = (x0 >= plot_start_rel) & (x0 <= plot_end_rel)
        fit_mask = plot_mask.copy()
    # 2020tld has a flux spike at t≈59107.876 TJD;
    # including it forces the regression into rapid oscillation to accommodate the outlier.
    # This point is therefore excluded for regression purposes.
    if base_name.replace("_binned", "").lower() == "2020tld":
        spike_mask = np.abs(time_values - 59107.8756786985) > 0.1
        fit_mask = fit_mask & spike_mask
        plot_mask = plot_mask & spike_mask
    return {
        "x0": x0,
        "flux": flux_values,
        "fit_mask": fit_mask,
        "plot_mask": plot_mask,
    }


def run_rsr_once(
    data_dir: str,
    output_root: str,
    run_index: int,
    *,
    niterations: int = DEFAULT_RSR_NITER,
    maxsize: int = DEFAULT_RSR_MAXSIZE,
    maxdepth: int = DEFAULT_RSR_MAXDEPTH,
    ncycles_per_iteration: int = DEFAULT_RSR_NCYCLES_PER_ITERATION,
) -> str:
    run_output_dir = os.path.join(output_root, f'bootstrap_run_{run_index:03d}')
    os.makedirs(run_output_dir, exist_ok=True)
    cmd = [sys.executable, RSR_SCRIPT_PATH, '--data-dir', data_dir, '--out-dir', run_output_dir]
    cmd += ['--niterations', str(int(niterations))]
    cmd += ['--maxsize', str(int(maxsize))]
    cmd += ['--maxdepth', str(int(maxdepth))]
    cmd += ['--ncycles-per-iteration', str(int(ncycles_per_iteration))]
    cmd += ['--no-plots']
    subprocess.run(cmd, check=True)
    manifest_path = os.path.join(run_output_dir, 'json', 'rsr_run_manifest.json')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f'Manifest not found: {manifest_path}')
    return manifest_path


def generalize_expression(expression_str: str, variable_name: str | None = None) -> str:
    expression_str = expression_str.replace('^', '**')
    try:
        expr = sympify(expression_str)
    except Exception:
        return expression_str
    if variable_name is None:
        try:
            expr = expr.xreplace({Symbol("time"): Symbol("x0")})
        except Exception:
            pass
        try:
            expr = expr.xreplace({Symbol("t"): Symbol("x0")})
        except Exception:
            pass
        try:
            expr = expr.subs({Symbol("x1"): 0, Symbol("x2"): 1})
        except Exception:
            pass
    if variable_name is not None:
        data_variables = {Symbol(variable_name)}
    else:
        if Symbol("x0") in getattr(expr, "free_symbols", set()):
            data_variables = {Symbol("x0")}
        else:
            data_variables = get_data_symbols(expr)
    try:
        constants = sorted(expr.atoms(numbers.Number), key=lambda z: str(z))
        constants_filtered = [
            c for c in constants
            if not isinstance(c, Integer)
        ]
        substitution_map = {c: Symbol(f'p{i}') for i, c in enumerate(constants_filtered)}
        expr = expr.xreplace(substitution_map) if substitution_map else expr
    except Exception:
        pass
    expr = expand(expr)
    try:
        if variable_name is not None:
            data_variables = {Symbol(variable_name)}
        else:
            if Symbol("x0") in getattr(expr, "free_symbols", set()):
                data_variables = {Symbol("x0")}
            else:
                data_variables = get_data_symbols(expr)
        parameter_symbols = [s for s in expr.free_symbols if s not in data_variables]
        parameter_set = set(parameter_symbols)
        if expr.is_Add:
            terms = expr.as_ordered_terms()
            parameter_only_terms = []
            variable_terms = []
            for term in terms:
                try:
                    free = term.free_symbols
                except Exception:
                    free = set()
                if not free:
                    parameter_only_terms.append(term)
                elif free.issubset(parameter_set):
                    parameter_only_terms.append(term)
                else:
                    variable_terms.append(term)
            if parameter_only_terms:
                sym = Symbol('a')
                expr = sym + sum(variable_terms)
        elif expr.is_Number or (getattr(expr, 'free_symbols', set()).issubset(parameter_set) and expr != 0):
            expr = Symbol('a')
    except Exception:
        pass
    try:
        parameter_symbols = [s for s in expr.free_symbols if s not in data_variables]
        parameter_set = set(parameter_symbols)
        cache: Dict = {}
        def collapse_parameter_only_subexpressions(e):
            try:
                free = e.free_symbols
            except Exception:
                free = set()
            if free.issubset(parameter_set) and not e.is_Number:
                if e in cache:
                    return cache[e]
                sym = Symbol(f'q{len(cache)}')
                cache[e] = sym
                return sym
            if e.args:
                return e.func(*[collapse_parameter_only_subexpressions(a) for a in e.args])
            return e
        expr = collapse_parameter_only_subexpressions(expr)
    except Exception:
        pass
    try:
        if Symbol("x0") in getattr(expr, "free_symbols", set()):
            _dv = {Symbol("x0")}
        else:
            _dv = get_data_symbols(expr)
        _x0_sym = sorted(_dv, key=lambda s: s.name)[0] if _dv else Symbol('x0')
        _pow_map: Dict = {}
        _pow_counter = [2]
        def _collect_x0_pows(e):
            if e.is_Pow and e.base == _x0_sym and getattr(e.exp, 'free_symbols', set()):
                if e not in _pow_map:
                    _pow_map[e] = _x0_sym ** _pow_counter[0]
                    _pow_counter[0] += 1
            for a in getattr(e, 'args', []):
                _collect_x0_pows(a)
        _collect_x0_pows(expr)
        if _pow_map:
            expr = expr.xreplace(_pow_map)
    except Exception:
        pass
    try:
        parameter_symbols = [s for s in expr.free_symbols if s not in data_variables]
        parameter_set = set(parameter_symbols)
        trig_symbols: List[Symbol] = []
        def normalize_trig_arguments(e):
            if e.func is sin and e.args:
                arg = e.args[0]
                for v in sorted(data_variables, key=lambda s: s.name):
                    if v in getattr(arg, 'free_symbols', set()):
                        free = getattr(arg, 'free_symbols', set()) - {v}
                        if free.issubset(parameter_set):
                            from sympy import Poly
                            try:
                                P = Poly(arg, v)
                                if P.degree() == 1:
                                    coeffs = P.all_coeffs()
                                    a0 = coeffs[0]
                                    b0 = coeffs[1] if len(coeffs) > 1 else 0
                                    if b0 == 0:
                                        k_mul = Symbol(f'ktrig_mul{len(trig_symbols)}')
                                        trig_symbols.append(k_mul)
                                        return sin(k_mul * v)
                                    if a0 == 1:
                                        k_add = Symbol(f'ktrig_add{len(trig_symbols)}')
                                        trig_symbols.append(k_add)
                                        return sin(v + k_add)
                                    k_mul = Symbol(f'ktrig_mul{len(trig_symbols)}')
                                    trig_symbols.append(k_mul)
                                    k_add = Symbol(f'ktrig_add{len(trig_symbols)}')
                                    trig_symbols.append(k_add)
                                    return sin(k_mul * v + k_add)
                            except Exception:
                                pass
                        if arg.is_Mul and (getattr(arg, 'free_symbols', set()) - {v}).issubset(parameter_set):
                            k_mul = Symbol(f'ktrig_mul{len(trig_symbols)}')
                            trig_symbols.append(k_mul)
                            return sin(k_mul * v)
                        if arg.is_Add and v in getattr(arg, 'free_symbols', set()) and (getattr(arg, 'free_symbols', set()) - {v}).issubset(parameter_set):
                            k_add = Symbol(f'ktrig_add{len(trig_symbols)}')
                            trig_symbols.append(k_add)
                            return sin(v + k_add)
            if e.args:
                return e.func(*[normalize_trig_arguments(a) for a in e.args])
            return e
        expr = normalize_trig_arguments(expr)
    except Exception:
        pass
    try:
        parameter_symbols = [s for s in expr.free_symbols if s not in data_variables]
        parameter_set = set(parameter_symbols)
        terms = expr.as_ordered_terms() if expr.is_Add else [expr]
        def split_term(t):
            if t.is_Mul:
                coeff = 1
                var_part = 1
                for a in t.args:
                    if a.is_Number or a.free_symbols.issubset(parameter_set):
                        coeff *= a
                    else:
                        var_part *= a
                return coeff, var_part
            if t.is_Pow or t.func is sin or t.func is cos or t.func is exp:
                return 1, t
            if t.free_symbols.issubset(parameter_set) or t.is_Number:
                return t, 1
            return 1, t
        shapes = []
        for t in terms:
            _, vpart = split_term(t)
            if all(not vpart.equals(s) for s in shapes):
                shapes.append(vpart)
        def term_category(s, v):
            free = getattr(s, 'free_symbols', set())
            if not free:
                return 0
            if s == v:
                return 1
            if s.is_Pow and s.base == v:
                ex = s.exp
                if ex == 1:
                    return 1
                return 2
            if s.func is sin and v in getattr(s.args[0], 'free_symbols', set()):
                return 3
            if s.is_Mul and any(arg.func is sin for arg in s.args):
                return 4
            if v in free:
                return 5
            return 9
        main_var = sorted(data_variables, key=lambda s: s.name)[0]
        shapes_sorted = sorted(shapes, key=lambda s: (term_category(s, main_var), sstr(s)))
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        rebuilt = 0
        for i, shp in enumerate(shapes_sorted):
            name = alphabet[i] if i < len(alphabet) else f'a{i}'
            rebuilt += Symbol(name) * shp
        expr = expand(rebuilt)
    except Exception:
        pass
    def sort_args(e):
        if e.is_Add or e.is_Mul:
            args = tuple(sort_args(a) for a in e.args)
            args = tuple(sorted(args, key=sstr))
            return e.func(*args)
        if e.args:
            return e.func(*[sort_args(a) for a in e.args])
        return e
    try:
        expr = sort_args(expr)
    except Exception:
        pass
    try:
        import re as _re
        parameter_symbols_all = [s for s in expr.free_symbols if s not in data_variables]
        expr_str_repr = sstr(expr)
        def _param_pos(sym, s):
            m = _re.search(r'(?<![A-Za-z0-9_])' + _re.escape(str(sym)) + r'(?![A-Za-z0-9_])', s)
            return m.start() if m else len(s)
        parameter_symbols_all = sorted(parameter_symbols_all, key=lambda s: _param_pos(str(s), expr_str_repr))
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        remap = {}
        for i, old in enumerate(parameter_symbols_all):
            name = alphabet[i] if i < len(alphabet) else f'a{i}'
            remap[old] = Symbol(name)
        if remap:
            expr = expr.xreplace(remap)
        return sstr(expr)
    except Exception:
        try:
            return sstr(expr)
        except Exception:
            return str(expr)


def collect_general_forms(manifest_paths: List[str]):
    per_lightcurve: Dict[str, Counter] = defaultdict(Counter)
    form_stats = {}
    for manifest_path in manifest_paths:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        for summary in manifest.get('summaries', []):
            base = summary.get('file')
            equations = list(summary.get('equations', []))
            if len(equations) > 2:
                equations = equations[:2]
            if not base or not equations:
                continue
            summed = None
            if isinstance(equations, list) and len(equations) > 0:
                try:
                    exprs = [str(e).replace('^', '**') for e in equations]
                    summed = '+'.join(exprs)
                except Exception:
                    summed = '+'.join([str(e) for e in equations])
            if not summed:
                continue
            generalized = generalize_expression(summed)
            try:
                expr = sympify(generalized)
            except Exception:
                expr = None
            complexity = generalized_form_complexity(expr)
            per_lightcurve[base][generalized] += 1
            key = (base, generalized)
            if key not in form_stats:
                form_stats[key] = {"sum_complexity": 0.0, "sum_count": 0}
            form_stats[key]["sum_complexity"] += float(complexity) if complexity is not None else 0.0
            form_stats[key]["sum_count"] += 1
    return per_lightcurve, form_stats


def save_bootstrap_plots(manifest_path: str, lightcurve_root: str, run_index: int) -> None:
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    for summary in manifest.get('summaries', []):
        base = summary.get('file')
        informed_path = summary.get('csv')
        components_path = summary.get('components_csv')
        if not base or not informed_path or not components_path:
            continue
        if not os.path.exists(informed_path) or not os.path.exists(components_path):
            continue
        try:
            inf_df = pd.read_csv(informed_path)
            comp_df = pd.read_csv(components_path)
        except Exception:
            continue
        if len(inf_df) != len(comp_df):
            continue
        df = inf_df.copy()
        for col in ("comp1", "comp2", "model"):
            if col in comp_df.columns:
                df[col] = comp_df[col].values
        model_cols = [c for c in ("comp1", "comp2") if c in df.columns]
        dup_items = list(df.groupby("duplicate_id", sort=True))[:10]
        fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharex=False, sharey=False, constrained_layout=False)
        axes_list = axes.flatten()
        for ax in axes_list:
            ax.grid(True, alpha=0.4)
            ax.tick_params(axis="both", labelbottom=True, labelleft=True)
        for ax, (dup_id, df_dup) in zip(axes_list, dup_items):
            t0_val = float(df_dup["t_0"].iloc[0])
            a_val = float(df_dup["amplitude"].iloc[0])
            scale_val = float(df_dup["window_scale"].iloc[0])
            window_min = float(df_dup["window_min"].iloc[0])
            window_max = float(df_dup["window_max"].iloc[0])
            ax.errorbar(df_dup["time"], df_dup["flux"], yerr=df_dup["flux_err"],
                        fmt="o", markersize=3, alpha=0.7, label="points")
            t_order = np.argsort(df_dup["time"].to_numpy())
            t_sorted = df_dup["time"].to_numpy()[t_order]
            cumulative = np.zeros(len(df_dup))
            for col in model_cols:
                cumulative = cumulative + df_dup[col].to_numpy()
                ax.plot(t_sorted, cumulative[t_order], linewidth=1.3)
            ax.set_title(f"dup {int(dup_id)}: t_0={t0_val:.2f}, A={a_val:.2f}, scale={scale_val:.3f}")
            ax.set_xlim(window_min, window_max * 1.10)
        for ax in axes_list[len(dup_items):]:
            ax.axis("off")
        fig.suptitle(f"{base} — run {run_index:03d}", fontsize=12)
        fig.supxlabel("Time since T_exp (days)")
        fig.supylabel("Flux")
        fig.subplots_adjust(hspace=0.35, wspace=0.25, top=0.92, bottom=0.08)
        dest_dir = os.path.join(lightcurve_root, base)
        os.makedirs(dest_dir, exist_ok=True)
        out_path = os.path.join(dest_dir, f"{base}_run_{run_index:03d}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.12)
        plt.close(fig)


def clean_previous_outputs(data_dir: str, output_root: str, validation_root: str) -> None:
    if os.path.isdir(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)
    for summary_name in ("summary_filtered.txt",):
        summary_path = os.path.join(validation_root, summary_name)
        try:
            os.remove(summary_path)
        except FileNotFoundError:
            pass
    try:
        names = [f for f in os.listdir(data_dir) if f.endswith('_binned.csv')]
    except FileNotFoundError:
        names = []
    for f in names:
        base = os.path.splitext(f)[0]
        d = os.path.join(validation_root, base)
        if os.path.isdir(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)


def relocate_lightcurve_dirs(source_root: str, dest_root: str, data_dir: str) -> None:
    if os.path.abspath(source_root) == os.path.abspath(dest_root):
        return
    os.makedirs(dest_root, exist_ok=True)
    try:
        names = [f for f in os.listdir(data_dir) if f.endswith('_binned.csv')]
    except FileNotFoundError:
        names = []
    for name in names:
        base = os.path.splitext(name)[0]
        src = os.path.join(source_root, base)
        dest = os.path.join(dest_root, base)
        if os.path.isdir(src) and not os.path.exists(dest):
            shutil.move(src, dest)


def clean_pycache(root_dir: str) -> None:
    for dirpath, dirnames, _ in os.walk(root_dir):
        if "__pycache__" in dirnames:
            pycache_path = os.path.join(dirpath, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
            except FileNotFoundError:
                pass


def score_model(avg_r2: float | None, occurrences: int, total_runs: int, avg_complexity: float) -> float:
    r2_term = 0.0
    if avg_r2 is not None and math.isfinite(float(avg_r2)):
        r2_clamped = max(0.0, min(float(avg_r2), 1.0 - 1e-9))
        r2_term = WEIGHT_R2 * (-math.log10(1.0 - r2_clamped))
    occurrence_term = WEIGHT_OCCURRENCE * (float(occurrences) / float(max(total_runs, 1)))
    complexity_term = WEIGHT_COMPLEXITY * float(avg_complexity)
    return r2_term + occurrence_term - complexity_term


def parametric_regression_r2(general_form: str, t_data: np.ndarray, flux_data: np.ndarray) -> float:
    expr_str = general_form.replace('^', '**')
    try:
        expr = sympify(expr_str)
    except Exception:
        return float('nan')
    x0_sym = Symbol('x0')
    free = expr.free_symbols
    if x0_sym not in free:
        return float('nan')
    param_syms = sorted(free - {x0_sym}, key=lambda s: s.name)
    n_params = len(param_syms)
    mask = np.isfinite(t_data) & np.isfinite(flux_data)
    t_fit, y_fit = t_data[mask], flux_data[mask]
    if t_fit.size < max(n_params + 1, 2):
        return float('nan')
    all_args = [x0_sym] + param_syms
    f_lam = lambdify(all_args, expr, modules='numpy')

    def model(x, *p):
        try:
            return np.asarray(f_lam(x, *p), dtype=float)
        except Exception:
            return np.full_like(x, np.nan, dtype=float)

    span = float(np.nanmax(t_fit) - np.nanmin(t_fit)) if t_fit.size > 1 else 1.0
    freq_guess = 2.0 * np.pi / span if span > 0 else 1.0
    amp_guess = float(np.nanstd(y_fit)) or 1.0
    p0_smart = []
    for s in param_syms:
        if s.name in ('e', 'f', 'h', 'k', 'n'):
            p0_smart.append(freq_guess)
        elif s.name in ('d', 'g', 'j', 'm'):
            p0_smart.append(amp_guess)
        else:
            p0_smart.append(1.0)
    p0_options = [[1.0] * n_params, p0_smart, [freq_guess] * n_params]
    best_r2 = float('nan')
    for p0 in p0_options:
        try:
            popt, _ = curve_fit(model, t_fit, y_fit, p0=p0, maxfev=20000)
            y_pred = model(t_fit, *popt)
        except Exception:
            try:
                result = least_squares(lambda p: model(t_fit, *p) - y_fit, p0, max_nfev=20000)
                y_pred = model(t_fit, *result.x)
            except Exception:
                continue
        ok = np.isfinite(y_pred)
        if ok.sum() < 2:
            continue
        yt, yp = y_fit[ok], y_pred[ok]
        sse = float(np.sum((yt - yp) ** 2))
        sst = float(np.sum((yt - np.mean(yt)) ** 2))
        r2 = 1.0 - sse / sst if sst > 0 else float('nan')
        if math.isfinite(r2) and (not math.isfinite(best_r2) or r2 > best_r2):
            best_r2 = r2
    return best_r2


def write_filtered_summary_text(
    tally: Dict[str, Counter],
    form_stats: dict,
    total_runs: int,
    summary_root: str,
    min_occ: int,
    r2_map: dict,
    *,
    verbose: bool = True,
) -> str:
    out_path = os.path.join(summary_root, "summary_filtered.txt")
    with open(out_path, "w") as f:
        f.write(f"Minimum occurrence threshold: {min_occ}\n\n")
        for base, counter in tally.items():
            entries = []
            for form, cnt in counter.most_common():
                if cnt < min_occ:
                    continue
                key = (base, form)
                stats = form_stats.get(key, {"sum_complexity": 0.0, "sum_count": cnt})
                avg_complexity = stats["sum_complexity"] / max(stats.get("sum_count", cnt), 1)
                r2 = r2_map.get(key)
                r2_for_score = r2 if (r2 is not None and math.isfinite(r2)) else None
                score = score_model(r2_for_score, cnt, total_runs, avg_complexity)
                entries.append((score, form, cnt, avg_complexity, r2))
            entries.sort(key=lambda entry: float(entry[0]), reverse=True)
            f.write(f"\n=== {base} ===\n")
            if not entries:
                f.write("No forms above threshold.\n")
                continue
            for rank, (score, form, cnt, avg_complexity, r2) in enumerate(entries, start=1):
                r2_str = "NA" if (r2 is None or not math.isfinite(r2)) else f"{r2:.5f}"
                f.write(f"{rank}. General Symbolic Form: {form}\n")
                f.write(f"  Occurrences = {cnt}, Complexity = {avg_complexity:.2f}, R^2 = {r2_str}, Score = {score:.4f}\n")
            f.write("\n")
    if verbose:
        print(f"Saved filtered summary (TXT): {out_path}")
    return out_path








def main():
    parser = argparse.ArgumentParser(description='Bootstrap ensemble and general-form tally')
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument('--out-root', default=OUTPUT_ROOT)
    parser.add_argument('--runs', type=int, default=DEFAULT_RUNS)
    parser.add_argument('--min-occurrence', type=int, default=None)
    parser.add_argument('--rsr-niterations', type=int, default=DEFAULT_RSR_NITER)
    parser.add_argument('--rsr-maxsize', type=int, default=DEFAULT_RSR_MAXSIZE)
    parser.add_argument('--rsr-maxdepth', type=int, default=DEFAULT_RSR_MAXDEPTH)
    parser.add_argument('--rsr-ncycles-per-iteration', type=int, default=DEFAULT_RSR_NCYCLES_PER_ITERATION)
    parser.add_argument('--no-bootstrap-plots', action='store_true')
    parser.add_argument('--clean-pycache-every', type=int, default=DEFAULT_CLEAN_PYCACHE_EVERY)
    args = parser.parse_args()

    args.data_dir = os.path.abspath(os.path.expanduser(args.data_dir))

    relocate_lightcurve_dirs(VALIDATION_ROOT, LIGHTCURVE_ROOT, args.data_dir)
    clean_previous_outputs(args.data_dir, args.out_root, VALIDATION_ROOT)

    os.makedirs(args.out_root, exist_ok=True)
    try:
        binned_files = [f for f in os.listdir(args.data_dir) if f.endswith('_binned.csv')]
    except FileNotFoundError:
        binned_files = []
    n_files = len(binned_files)
    expected_fits = n_files * int(args.runs) * 2
    print(f"Validation cycles: {args.runs}")
    print(f"Light curves per cycle: {n_files}")
    print(f"Expected PySR fits (up to 2 per curve): {expected_fits}")

    manifests: List[str] = []
    print(f"Processing directory: {args.data_dir}")
    for i in range(args.runs):
        m = run_rsr_once(
            data_dir=args.data_dir,
            output_root=args.out_root,
            run_index=i,
            niterations=args.rsr_niterations,
            maxsize=args.rsr_maxsize,
            maxdepth=args.rsr_maxdepth,
            ncycles_per_iteration=args.rsr_ncycles_per_iteration,
        )
        if not args.no_bootstrap_plots:
            save_bootstrap_plots(m, LIGHTCURVE_ROOT, i)
        manifests.append(m)
        if args.clean_pycache_every > 0 and ((i + 1) % args.clean_pycache_every == 0):
            clean_pycache(VALIDATION_ROOT)

    if not manifests:
        return
    tally, form_stats = collect_general_forms(manifests)
    min_occ = args.min_occurrence if args.min_occurrence is not None else int(math.sqrt(max(args.runs, 1)))

    print("Refitting discovered forms to original light curves...")
    r2_map: dict = {}
    lc_cache: dict = {}
    for base, counter in tally.items():
        if base not in lc_cache:
            lc_cache[base] = load_lightcurve_window(base, args.data_dir)
        lc = lc_cache[base]
        for form, cnt in counter.most_common():
            if cnt < min_occ:
                continue
            key = (base, form)
            if lc is None:
                r2_map[key] = float('nan')
                continue
            plot_mask = np.asarray(lc['plot_mask'], dtype=bool)
            t_plot = np.asarray(lc['x0'], dtype=float)[plot_mask]
            f_plot = np.asarray(lc['flux'], dtype=float)[plot_mask]
            r2_map[key] = parametric_regression_r2(form, t_plot, f_plot)

    for base, counter in tally.items():
        print(f"\n=== {base} ===")
        summaries = []
        for form, cnt in counter.most_common():
            key = (base, form)
            stats = form_stats.get(key, {"sum_complexity": 0.0, "sum_count": cnt})
            avg_complexity = stats["sum_complexity"] / max(stats.get("sum_count", cnt), 1)
            r2 = r2_map.get(key)
            r2_for_score = r2 if (r2 is not None and math.isfinite(r2)) else None
            sc = score_model(r2_for_score, cnt, len(manifests), avg_complexity)
            summaries.append((form, cnt, avg_complexity, r2, sc))
        summaries.sort(key=lambda t: t[-1], reverse=True)
        for form, cnt, avg_complexity, r2, sc in summaries:
            r2_str = "NA" if (r2 is None or not math.isfinite(r2)) else f"{r2:.5f}"
            print(f"General Symbolic Form: {form}")
            print(f"  Occurrences = {cnt}, Complexity = {avg_complexity:.2f}, R^2 = {r2_str}, Score = {sc:.4f}")
        if summaries:
            form, cnt, avg_complexity, r2, sc = summaries[0]
            r2_str = "NA" if (r2 is None or not math.isfinite(r2)) else f"{r2:.5f}"
            print(f"\nBest Model: {form}  (Score = {sc:.4f}, R^2 = {r2_str})")

    write_filtered_summary_text(
        tally,
        form_stats,
        len(manifests),
        VALIDATION_ROOT,
        min_occ,
        r2_map,
        verbose=True,
    )


if __name__ == '__main__':
    main()
