import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional


VALIDATION_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = VALIDATION_DIR / "catalog_split"
DEFAULT_INFO_PATH = VALIDATION_DIR / "sne_info.txt"
BIN_SIZE_DAYS = 0.25

VALUE_COLUMNS = ["BJD", "t", "CRate", "Frac", "Tmag", "CalOff"]
ERROR_COLUMNS = ["e_CRate", "e_Frac", "e_Tmag"]
OUTPUT_COLUMNS = ["BJD", "t", "CRate", "e_CRate", "Frac", "e_Frac", "Tmag", "e_Tmag", "CalOff"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bin catalog_split light curves by t and write *_binned.csv files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing split CSV files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--sn-id",
        help="Optional SN identifier to bin for a single-object test run.",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        default=BIN_SIZE_DAYS,
        help=f"Bin size in days for t (default: {BIN_SIZE_DAYS})",
    )
    return parser.parse_args()


def parse_float(value: str) -> float:
    if value is None:
        return float("nan")
    value = value.strip()
    if not value:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def load_sne_info(info_path: Path = DEFAULT_INFO_PATH) -> Dict[str, Dict[str, float]]:
    sne_info: Dict[str, Dict[str, float]] = {}
    with info_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.startswith("SN"):
                continue
            parts = raw_line.split()
            if len(parts) < 13:
                continue
            sne_info[parts[0]] = {
                "First": parse_float(parts[1]),
                "e_First": parse_float(parts[2]),
                "tRise": parse_float(parts[3]),
                "PLbeta1": parse_float(parts[4]),
                "e_PLbeta1": parse_float(parts[5]),
                "PLbeta2": parse_float(parts[6]),
                "e_PLbeta2": parse_float(parts[7]),
                "lnZ": parse_float(parts[8]),
                "e_lnZ": parse_float(parts[9]),
                "BIC": parse_float(parts[10]),
                "Ratio": parse_float(parts[11]),
                "BitMask": parts[12],
            }
    return sne_info


def load_rows(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {column: parse_float(row.get(column, "")) for column in OUTPUT_COLUMNS}
            if not math.isfinite(parsed["t"]):
                continue
            rows.append(parsed)
    return rows


def mean_or_blank(values: Iterable[float]) -> str:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return ""
    return f"{sum(finite_values) / len(finite_values):.5f}"


def combined_error_or_blank(values: Iterable[float]) -> str:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return ""
    quadrature_sum = math.sqrt(sum(value * value for value in finite_values))
    return f"{quadrature_sum / len(finite_values):.5f}"


def filter_rows_to_time_window(
    rows: List[Dict[str, float]],
    sn_id: str,
    sne_info: Dict[str, Dict[str, float]],
) -> List[Dict[str, float]]:
    sn_info = sne_info.get(sn_id)
    if sn_info is None:
        return [row for row in rows if math.isfinite(row["t"]) and row["t"] >= 0.0]

    first_light = sn_info.get("First", float("nan"))
    t_rise = sn_info.get("tRise", float("nan"))
    if not math.isfinite(first_light):
        return [row for row in rows if math.isfinite(row["t"]) and row["t"] >= 0.0]

    if not math.isfinite(t_rise):
        return [
            row for row in rows
            if math.isfinite(row["t"]) and row["t"] >= 0.0
        ]

    # t is already defined as BJD - First, so the BJD window
    # [First, First + tRise] maps to the t window [0, tRise].
    return [
        row for row in rows
        if math.isfinite(row["t"]) and 0.0 <= row["t"] <= t_rise
    ]


def bin_rows(
    rows: List[Dict[str, float]],
    bin_size: float,
    sn_id: str,
    sne_info: Dict[str, Dict[str, float]],
) -> List[Dict[str, str]]:
    if not rows:
        return []

    rows = filter_rows_to_time_window(rows, sn_id, sne_info)
    if not rows:
        return []

    rows = sorted(rows, key=lambda row: row["t"])
    t_min = rows[0]["t"]
    binned_rows: List[Dict[str, str]] = []
    current_bucket: List[Dict[str, float]] = []
    current_index = None

    for row in rows:
        bucket_index = int(math.floor((row["t"] - t_min) / bin_size))
        if current_index is None:
            current_index = bucket_index
        if bucket_index != current_index:
            binned_rows.append(summarize_bucket(current_bucket))
            current_bucket = []
            current_index = bucket_index
        current_bucket.append(row)

    if current_bucket:
        binned_rows.append(summarize_bucket(current_bucket))

    return binned_rows


def summarize_bucket(bucket: List[Dict[str, float]]) -> Dict[str, str]:
    summary: Dict[str, str] = {}
    for column in VALUE_COLUMNS:
        summary[column] = mean_or_blank(row[column] for row in bucket)
    for column in ERROR_COLUMNS:
        summary[column] = combined_error_or_blank(row[column] for row in bucket)
    return summary


def write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def iter_input_files(input_dir: Path, sn_id: Optional[str]) -> List[Path]:
    if sn_id is not None:
        return [input_dir / f"{sn_id}.csv"]
    return sorted(
        path for path in input_dir.glob("*.csv")
        if not path.name.endswith("_binned.csv")
    )


def main() -> None:
    args = parse_args()
    sne_info = load_sne_info()
    input_files = iter_input_files(args.input_dir, args.sn_id)
    if not input_files:
        raise SystemExit("No input CSV files found to bin")

    for input_path in input_files:
        if not input_path.exists():
            print(f"Missing input file: {input_path}")
            continue
        rows = load_rows(input_path)
        if not rows:
            print(f"No valid rows found in {input_path}")
            continue
        sn_id = input_path.stem
        binned_rows = bin_rows(rows, args.bin_size, sn_id, sne_info)
        if not binned_rows:
            print(f"No rows in the [First, First + tRise] window for {sn_id}")
            continue
        output_path = input_path.with_name(f"{input_path.stem}_binned.csv")
        write_rows(output_path, binned_rows)
        print(f"Wrote {len(binned_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
