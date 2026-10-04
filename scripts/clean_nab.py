#### 1. Data Cleaning Script for NAB Datasets ####
"""Clean NAB datasets in `data/NAB/data` (the NAB git submodule).

Saves cleaned CSVs to `data/nab_cleaned/` and writes `data/nab_cleaned/metadata.json`.
The output lives outside the submodule (and is gitignored) so the submodule stays clean.

Usage (from the repository root):
  python scripts/clean_nab.py --input-dir data/NAB/data --output-dir data/nab_cleaned
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("clean_nab")


def detect_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    # Prefer obvious names
    candidates = [c for c in df.columns if c.lower() in ("timestamp", "time", "date")] 
    if candidates:
        return candidates[0]

    # Try to find a string column that parses as datetime for most values
    best_col = None
    best_frac = 0.0
    for c in df.columns:
        # skip numeric columns
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        parsed = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
        frac = parsed.notna().mean()
        if frac > best_frac:
            best_frac = frac
            best_col = c
    if best_frac >= 0.5:
        return best_col
    return None


def detect_value_column(df: pd.DataFrame, timestamp_col: Optional[str]) -> Optional[str]:
    # Prefer 'value' or common names
    for name in ("value", "y", "metric", "volume"):
        for c in df.columns:
            if c.lower() == name:
                return c

    # Otherwise pick the most numeric column that's not the timestamp
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if timestamp_col and timestamp_col in numeric_cols:
        numeric_cols.remove(timestamp_col)
    if numeric_cols:
        return numeric_cols[0]
    # Fallback to the last column that's not timestamp
    for c in reversed(df.columns.tolist()):
        if c != timestamp_col:
            return c
    return None


def clean_file(src: Path, dst: Path) -> dict:
    df = pd.read_csv(src)
    meta = {"file": str(src), "rows_before": len(df)}

    ts_col = detect_timestamp_column(df)
    if ts_col:
        df["_timestamp"] = pd.to_datetime(df[ts_col], errors="coerce", infer_datetime_format=True)
        df = df.drop(columns=[ts_col])
        df = df.rename(columns={"_timestamp": "timestamp"})
        df = df.dropna(subset=["timestamp"])  # drop rows with invalid timestamps
        df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    else:
        # create integer index if no timestamp
        df.index.name = "index"

    val_col = detect_value_column(df, timestamp_col=ts_col)
    if val_col is None:
        meta.update({"error": "no value column detected"})
        return meta

    # coerce value to numeric
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    if val_col != "value":     # Standardize value column name
        df = df.rename(columns={val_col: "value"})
        val_col = "value"
    missing_before = df[val_col].isna().sum()
    meta["value_column"] = val_col
    meta["missing_values_before"] = int(missing_before)

    # drop rows where both timestamp and value are missing
    if "timestamp" in df.index.names:
        # index is timestamp
        df = df[~(df[val_col].isna())]
    else:
        df = df[~(df[val_col].isna())]

    # remove duplicate timestamps keeping the last observed
    if "timestamp" in df.index.names:
        dup_count = df.index.duplicated().sum()
        if dup_count:
            df = df[~df.index.duplicated(keep="last")]
        meta["duplicate_timestamps_removed"] = int(dup_count)

    # Fill small gaps by linear interpolation, larger gaps by ffill/bfill
    total = len(df)
    if total == 0:
        meta.update({"rows_after": 0})
        return meta

    missing_after_drop = df[val_col].isna().sum()
    frac_missing = missing_after_drop / float(total)
    meta["missing_fraction_after_drop"] = float(frac_missing)

    if frac_missing > 0:
        if frac_missing <= 0.1:
            df[val_col] = df[val_col].interpolate(method="time" if "timestamp" in df.index.names else "linear")
            df[val_col] = df[val_col].fillna(method="ffill").fillna(method="bfill")
        else:
            df[val_col] = df[val_col].fillna(method="ffill").fillna(method="bfill")

    # Final cleanup: drop rows still missing value
    df = df[~df[val_col].isna()]

    meta["rows_after"] = int(len(df))
    if "timestamp" in df.index.names:
        start, end = df.index.min(), df.index.max()
        meta["start"] = str(start)
        meta["end"] = str(end)

    # Ensure output directory exists
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Save cleaned CSV
    # If index is timestamp, write index as 'timestamp'
    if "timestamp" in df.index.names:
        df.to_csv(dst, index=True, index_label="timestamp")
    else:
        df.to_csv(dst, index=False)

    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_files = list(inp.rglob("*.csv"))
    logger.info(f"Found {len(all_files)} CSV files in {inp}")

    metadata = {}
    for p in sorted(all_files):
        rel = p.relative_to(inp)
        dst = out / rel
        logger.info(f"Cleaning {rel}")
        meta = clean_file(p, dst)
        metadata[str(rel)] = meta

    # Write metadata summary
    meta_path = out / "metadata.json"
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Cleaning complete. Cleaned files and metadata at {out}")


if __name__ == "__main__":
    main()
