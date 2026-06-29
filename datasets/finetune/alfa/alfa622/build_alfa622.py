from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TARGET_FIELDS = [
    "field.angular_velocity.x",
    "field.angular_velocity.y",
    "field.angular_velocity.z",
    "field.linear_acceleration.x",
    "field.linear_acceleration.y",
    "field.linear_acceleration.z",
    "field.magnetic_field.x",
    "field.magnetic_field.y",
    "field.magnetic_field.z",
    "field.fluid_pressure",
    "field.temperature",
    "field.measured.pitch",
    "field.measured.roll",
    "field.measured.yaw",
    "field.alt_error",
    "field.aspd_error",
    "field.xtrack_error",
    "field.wp_dist",
]

EXCLUDED_FILES = {
    "carbonZ_2018-09-11-15-06-34_2_rudder_right_failure.csv": (
        "missing field.measured.pitch, field.measured.roll, field.measured.yaw"
    ),
    "carbonZ_2018-10-18-11-06-06_engine_failure.csv": (
        "missing field.measured.pitch, field.measured.roll, field.measured.yaw"
    ),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Build split CSV files for ALFA5.17 finetuning.")
    parser.add_argument("--raw-dir", type=Path, default=root / "datasets/raw/ALFA5.17")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    return parser.parse_args()


def materialize_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = [name for name in TARGET_FIELDS if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing target columns: {missing}")

    result = frame[TARGET_FIELDS].copy()
    if "fault_state" in frame.columns:
        fault_state = pd.to_numeric(frame["fault_state"], errors="coerce").fillna(0.0)
        result["label"] = (fault_state > 0).astype("int64")
    else:
        result["label"] = 0
    return result


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")

    paths = sorted(args.raw_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {args.raw_dir}")

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    test_file_frames: dict[str, pd.DataFrame] = {}
    normal_files = 0
    anomaly_files = 0
    excluded_files = 0

    for path in paths:
        if path.name in EXCLUDED_FILES:
            excluded_files += 1
            continue
        frame = materialize_frame(path)
        is_anomaly_file = bool(frame["label"].max() > 0)
        if is_anomaly_file:
            anomaly_files += 1
            test_parts.append(frame)
            test_file_frames[path.name] = frame
            continue

        normal_files += 1
        split_index = int(len(frame) * args.train_ratio)
        if split_index <= 0 or split_index >= len(frame):
            raise ValueError(
                f"{path} cannot be split with train_ratio={args.train_ratio}: {len(frame)} rows"
            )
        train_parts.append(frame.iloc[:split_index])
        val_parts.append(frame.iloc[split_index:])

    if not train_parts:
        raise ValueError("No normal files were available for train.csv")
    if not val_parts:
        raise ValueError("No normal files were available for val.csv")
    if not test_parts:
        raise ValueError("No anomaly files were available for test.csv")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train.csv": pd.concat(train_parts, ignore_index=True),
        "val.csv": pd.concat(val_parts, ignore_index=True),
        "test.csv": pd.concat(test_parts, ignore_index=True),
    }
    for name, frame in outputs.items():
        frame.to_csv(args.out_dir / name, index=False)
    for name, frame in test_file_frames.items():
        frame.to_csv(args.out_dir / name, index=False)

    print(f"raw_dir: {args.raw_dir}")
    print(f"out_dir: {args.out_dir}")
    print(f"normal_files: {normal_files}")
    print(f"anomaly_files: {anomaly_files}")
    print(f"excluded_files: {excluded_files}")
    for name, reason in EXCLUDED_FILES.items():
        print(f"excluded: {name} ({reason})")
    for name, frame in outputs.items():
        print(f"{name}: rows={len(frame)}, anomaly_labels={int(frame['label'].sum())}")
    print(f"individual_test_files: {len(test_file_frames)}")


if __name__ == "__main__":
    main()
