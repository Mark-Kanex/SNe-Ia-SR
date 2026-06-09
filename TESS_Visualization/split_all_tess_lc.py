import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


VALIDATION_DIR = Path(__file__).resolve().parent
DEFAULT_LIGHTCURVE_PATH = VALIDATION_DIR / "all_tess_lc.txt"
DEFAULT_INFO_PATH = VALIDATION_DIR / "sne_info.txt"
DEFAULT_OUTPUT_DIR = VALIDATION_DIR / "catalog_split"

OUTPUT_COLUMNS = [
    "BJD",
    "t",
    "CRate",
    "e_CRate",
    "Frac",
    "e_Frac",
    "Tmag",
    "e_Tmag",
    "CalOff",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split validation_Data/all_tess_lc.txt into per-SN CSV files and "
            "add t = BJD - FirstLight from validation_Data/sne_info.txt."
        )
    )
    parser.add_argument(
        "--lightcurve-path",
        type=Path,
        default=DEFAULT_LIGHTCURVE_PATH,
        help=f"Path to all_tess_lc.txt (default: {DEFAULT_LIGHTCURVE_PATH})",
    )
    parser.add_argument(
        "--info-path",
        type=Path,
        default=DEFAULT_INFO_PATH,
        help=f"Path to sne_info.txt (default: {DEFAULT_INFO_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for per-SN CSV files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--sn-id",
        help="Optional SN identifier to export for a single-object test run.",
    )
    return parser.parse_args()


def load_first_light_map(info_path: Path) -> Dict[str, float]:
    first_light_by_id: Dict[str, float] = {}
    with info_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.startswith("SN"):
                continue
            parts = raw_line.split()
            if len(parts) < 2:
                continue
            sn_id = parts[0]
            first_light_by_id[sn_id] = float(parts[1])
    return first_light_by_id


def parse_lightcurve_row(raw_line: str, first_light: float) -> Dict[str, str]:
    parts = raw_line.split()
    sn_id = parts[0]
    bjd = float(parts[1])
    optional_values = parts[2:]
    padded_values = optional_values + [""] * (7 - len(optional_values))

    return {
        "BJD": parts[1],
        "t": f"{bjd - first_light:.5f}",
        "CRate": padded_values[0],
        "e_CRate": padded_values[1],
        "Frac": padded_values[2],
        "e_Frac": padded_values[3],
        "Tmag": padded_values[4],
        "e_Tmag": padded_values[5],
        "CalOff": padded_values[6],
    }


def collect_rows(
    lightcurve_path: Path,
    first_light_by_id: Dict[str, float],
    target_sn_id: Optional[str] = None,
) -> Tuple[Dict[str, List[Dict[str, str]]], Set[str]]:
    rows_by_id: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    missing_first_light: Set[str] = set()

    with lightcurve_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.startswith("SN"):
                continue

            sn_id = raw_line.split()[0]
            if target_sn_id is not None and sn_id != target_sn_id:
                continue

            first_light = first_light_by_id.get(sn_id)
            if first_light is None:
                missing_first_light.add(sn_id)
                continue

            rows_by_id[sn_id].append(parse_lightcurve_row(raw_line, first_light))

    return rows_by_id, missing_first_light


def write_rows(output_dir: Path, rows_by_id: Dict[str, List[Dict[str, str]]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sn_id, rows in rows_by_id.items():
        output_path = output_dir / f"{sn_id}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {output_path}")


def main() -> None:
    args = parse_args()
    first_light_by_id = load_first_light_map(args.info_path)
    rows_by_id, missing_first_light = collect_rows(
        lightcurve_path=args.lightcurve_path,
        first_light_by_id=first_light_by_id,
        target_sn_id=args.sn_id,
    )

    if not rows_by_id:
        if args.sn_id is not None:
            raise SystemExit(f"No rows found for {args.sn_id}")
        raise SystemExit("No rows found to export")

    write_rows(args.output_dir, rows_by_id)

    if missing_first_light:
        missing_list = ", ".join(sorted(missing_first_light))
        print(f"Skipped {len(missing_first_light)} SN IDs missing first-light data: {missing_list}")


if __name__ == "__main__":
    main()
