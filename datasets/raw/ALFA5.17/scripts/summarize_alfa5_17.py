from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


FILENAME_RE = re.compile(
    r"^carbonZ_(?P<date>\d{4}-\d{2}-\d{2})-(?P<clock>\d{2}-\d{2}-\d{2})(?:_(?P<tag>.+))?$"
)


@dataclass(frozen=True)
class FlightSummary:
    path: Path
    short_file: str
    date: str
    steps: int
    normal_count: int
    anomaly_count: int
    anomaly_segments: list[tuple[int, int]]
    anomaly_positions: list[str]
    label: str

    @property
    def ratio(self) -> str:
        return f"{self.normal_count}:{self.anomaly_count}"


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_raw_dir = script_dir.parent
    parser = argparse.ArgumentParser(description="Summarize and classify ALFA5.17 CSV flights.")
    parser.add_argument("--raw-dir", type=Path, default=default_raw_dir)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-copy", action="store_true", help="Only write the Markdown summary.")
    return parser.parse_args()


def make_short_file(path: Path) -> tuple[str, str]:
    match = FILENAME_RE.match(path.stem)
    if match is None:
        return path.stem, ""

    date = match.group("date")
    tag = match.group("tag")
    short_file = match.group("clock")
    if tag:
        short_file = f"{short_file}_{tag}"
    return short_file, date


def parse_fault_state(value: str | None) -> bool:
    if value is None or value == "":
        return False
    try:
        state = float(value)
    except ValueError:
        return False
    return math.isfinite(state) and state > 0


def first_date_from_time(value: str | None) -> str:
    if value is None:
        return ""
    match = re.match(r"(\d{4}-\d{2}-\d{2})", value)
    return match.group(1) if match else ""


def format_segments(segments: list[tuple[int, int]]) -> str:
    if not segments:
        return "-"
    return "; ".join(f"[{start},{end}]" for start, end in segments)


def format_positions(positions: list[str]) -> str:
    if not positions:
        return "-"
    return "; ".join(positions)


def summarize_csv(path: Path) -> FlightSummary:
    short_file, filename_date = make_short_file(path)
    steps = 0
    anomaly_count = 0
    segments: list[tuple[int, int]] = []
    current_start: int | None = None
    first_time_date = ""

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        has_fault_state = reader.fieldnames is not None and "fault_state" in reader.fieldnames

        for index, row in enumerate(reader):
            steps += 1
            if not first_time_date:
                first_time_date = first_date_from_time(row.get("time"))

            is_anomaly = has_fault_state and parse_fault_state(row.get("fault_state"))
            if is_anomaly:
                anomaly_count += 1
                if current_start is None:
                    current_start = index
            elif current_start is not None:
                segments.append((current_start, index - 1))
                current_start = None

    if current_start is not None:
        segments.append((current_start, steps - 1))

    positions = [
        f"{100.0 * start / steps:.1f}%~{100.0 * (end + 1) / steps:.1f}%"
        for start, end in segments
        if steps > 0
    ]
    label = "异常" if anomaly_count > 0 else "正常"
    return FlightSummary(
        path=path,
        short_file=short_file,
        date=filename_date or first_time_date or "-",
        steps=steps,
        normal_count=steps - anomaly_count,
        anomaly_count=anomaly_count,
        anomaly_segments=segments,
        anomaly_positions=positions,
        label=label,
    )


def markdown_table(rows: list[FlightSummary]) -> str:
    lines = [
        "| file | 日期 | 时间步 | 正常/异常比 | 异常片段 | 异常区间位置 | 数据集标签 |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.short_file,
                    row.date,
                    str(row.steps),
                    row.ratio,
                    format_segments(row.anomaly_segments),
                    format_positions(row.anomaly_positions),
                    row.label,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_markdown(rows: list[FlightSummary], output_path: Path) -> None:
    anomaly_rows = [row for row in rows if row.label == "异常"]
    text = "\n\n".join(
        [
            "# ALFA5.17 CSV 统计",
            (
                "说明：异常判定只依据 `fault_state > 0`。异常片段使用 0-based 时间步闭区间 "
                "`[start,end]`；异常区间位置按该片段在整段飞行序列中的百分比给出。"
            ),
            "## 全部飞行架次",
            markdown_table(rows),
            "## 异常飞行架次",
            markdown_table(anomaly_rows),
            "",
        ]
    )
    output_path.write_text(text, encoding="utf-8")


def copy_by_label(rows: list[FlightSummary], raw_dir: Path) -> None:
    norm_dir = raw_dir / "norm"
    anomaly_dir = raw_dir / "anomaly"
    norm_dir.mkdir(exist_ok=True)
    anomaly_dir.mkdir(exist_ok=True)

    for row in rows:
        target_dir = anomaly_dir if row.label == "异常" else norm_dir
        shutil.copy2(row.path, target_dir / row.path.name)


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    output_path = args.output.resolve() if args.output is not None else raw_dir / "alfa5_17_summary.md"

    csv_paths = sorted(raw_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    rows = [summarize_csv(path) for path in csv_paths]
    write_markdown(rows, output_path)
    if not args.no_copy:
        copy_by_label(rows, raw_dir)

    anomaly_count = sum(row.label == "异常" for row in rows)
    norm_count = len(rows) - anomaly_count
    print(f"Wrote {output_path}")
    print(f"Copied {norm_count} normal files and {anomaly_count} anomaly files")


if __name__ == "__main__":
    main()
