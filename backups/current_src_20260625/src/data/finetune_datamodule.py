from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import lightning as L


import numpy as np
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .csv_windows import CsvSeries, StandardScaler, WindowDataset, read_csv_series


@dataclass(frozen=True)
class FlightSplit:
    path: str
    start: int
    train_end: int
    val_end: int

    @classmethod
    def from_config(cls, path: str, value: object) -> "FlightSplit":
        if not isinstance(value, str):
            raise TypeError(f"Split for {path!r} must be a string like '0:1000:2000'")
        parts = [part.strip() for part in value.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Split for {path!r} must have format 'start:train_end:val_end'")
        try:
            start, train_end, val_end = (int(part) for part in parts)
        except ValueError as exc:
            raise ValueError(f"Split for {path!r} contains non-integer boundaries: {value!r}") from exc
        if start < 0 or train_end < 0 or val_end < 0:
            raise ValueError(f"Split for {path!r} must use non-negative boundaries")
        if not start < train_end <= val_end:
            raise ValueError(
                f"Split for {path!r} must satisfy start < train_end <= val_end, got {value!r}"
            )
        return cls(path=path, start=start, train_end=train_end, val_end=val_end)

    def validate_length(self, length: int) -> None:
        if self.val_end > length:
            raise ValueError(
                f"Split for {self.path!r} ends at {self.val_end}, but file length is {length}"
            )

    @property
    def train_slice(self) -> slice:
        return slice(self.start, self.train_end)

    @property
    def val_slice(self) -> slice:
        return slice(self.train_end, self.val_end)

    @property
    def test_slice(self) -> slice:
        return slice(self.val_end, None)


def _slice_array(values: np.ndarray | None, item: slice) -> np.ndarray | None:
    if values is None:
        return None
    return values[item]


def _slice_series(series: CsvSeries, item: slice) -> CsvSeries:
    return CsvSeries(
        path=series.path,
        values=series.values[item],
        feature_names=series.feature_names,
        labels=_slice_array(series.labels, item),
        time=_slice_array(series.time, item),
    )


class FinetuneDataModule(L.LightningDataModule):
    """DataModule for per-flight split/scaler finetuning.

    The dataset root must contain a ``config.yaml`` whose ``train`` section
    defines train/validation boundaries and whose ``test`` section defines
    test boundaries. Each selected flight is normalized by a scaler fitted on
    its own train segment.
    """

    def __init__(
        self,
        root_path: str | Path,
        feature_names: Sequence[str],
        seq_len: int,
        train_files: Sequence[str | Path] | None = None,
        val_files: Sequence[str | Path] | None = None,
        test_files: Sequence[str | Path] | None = None,
        stride: int = 1,
        eval_stride: int | None = None,
        batch_size: int = 256,
        num_workers: int = 0,
        label_col: str | None = "label",
        time_col: str | None = "time",
        scaler_type: str = "standard",
    ) -> None:
        super().__init__()
        self.root_path = Path(root_path)
        self.config_path = self.root_path / "config.yaml"
        self.feature_names = list(feature_names)
        self.seq_len = int(seq_len)
        self.train_files_arg = train_files
        self.val_files_arg = val_files
        self.test_files_arg = test_files
        self.stride = int(stride)
        self.eval_stride = int(eval_stride) if eval_stride is not None else self.stride
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.label_col = label_col
        self.time_col = time_col
        self.scaler_type = scaler_type

        self.split_config: dict[str, dict[str, FlightSplit]] = {}
        self.train_files: list[str] = []
        self.val_files: list[str] = []
        self.test_files: list[str] = []
        self.test_paths: list[Path] = []

        self.train_scalers: dict[str, StandardScaler] = {}
        self.test_scalers: dict[str, StandardScaler] = {}
        self.scaler: StandardScaler | None = None

        self.train_dataset: ConcatDataset | None = None
        self.val_dataset: ConcatDataset | None = None
        self.threshold_dataset: Dataset | None = None
        self._series_cache: dict[str, CsvSeries] = {}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "FinetuneDataModule":
        data_cfg = config["data"]
        return cls(
            root_path=data_cfg["root_path"],
            feature_names=data_cfg["target_fields"],
            seq_len=int(data_cfg["seq_len"]),
            train_files=data_cfg.get("train"),
            val_files=data_cfg.get("val"),
            test_files=data_cfg.get("test"),
            stride=int(data_cfg.get("stride", 1)),
            eval_stride=int(data_cfg.get("eval_stride", data_cfg.get("stride", 1))),
            batch_size=int(data_cfg.get("batch_size", 256)),
            num_workers=int(data_cfg.get("num_workers", 0)),
            label_col=data_cfg.get("label_col", "label"),
            time_col=data_cfg.get("timestamp_col", "time"),
            scaler_type=str(data_cfg.get("scaler_type", config.get("scaler_type", "standard"))),
        )

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return

        self._load_split_config()
        self.train_files = self._select_files("train", self.train_files_arg, field_name="train")
        self.val_files = self._select_files(
            "train",
            self.val_files_arg if self.val_files_arg is not None else self.train_files,
            field_name="val",
        )
        self.test_files = self._select_files("test", self.test_files_arg, field_name="test")
        self.test_paths = [self.root_path / path for path in self.test_files]

        train_datasets: list[WindowDataset] = []
        val_datasets: list[WindowDataset] = []
        for path in self.train_files:
            train_datasets.append(self._make_train_dataset(path))
        for path in self.val_files:
            dataset = self._make_val_dataset(path)
            if dataset is not None:
                val_datasets.append(dataset)

        if not train_datasets:
            raise ValueError("No training datasets were configured")
        self.train_dataset = ConcatDataset(train_datasets)
        self.threshold_dataset = self.train_dataset
        self.val_dataset = ConcatDataset(val_datasets) if val_datasets else None
        if len(self.train_scalers) == 1:
            self.scaler = next(iter(self.train_scalers.values()))

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader | None:
        if self.train_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting this dataloader")
        if self.val_dataset is None:
            return None
        return self._loader(self.val_dataset, shuffle=False)

    def threshold_dataloader(self) -> DataLoader:
        return self._loader(self.threshold_dataset, shuffle=False)

    def test_dataloader(self) -> list[DataLoader]:
        self._require_setup()
        return [self.test_dataloader_for(path) for path in self.test_files]

    def test_dataloader_for(self, path: str | Path) -> DataLoader:
        return self._loader(self.make_test_dataset(path), shuffle=False)

    def make_test_dataset(self, path: str | Path) -> WindowDataset:
        series = self.make_test_series(path)
        return self._window_dataset(
            path=self._normalize_file_key(path),
            split_name="test",
            values=series.values,
            labels=series.labels,
            stride=self.eval_stride,
            drop_anomaly_windows=False,
            allow_empty=False,
        )

    def make_test_series(self, path: str | Path) -> CsvSeries:
        self._require_setup()
        key = self._normalize_file_key(path)
        if key not in self.test_files:
            raise ValueError(f"{key!r} is not configured in data.test")
        split = self.split_config["test"][key]
        raw_series = self._load_series(key)
        split.validate_length(len(raw_series.values))
        scaler = self._test_scaler_for(key)
        self.scaler = scaler
        series = _slice_series(raw_series, split.test_slice)
        series.values = scaler.transform(series.values)
        return series

    def scaler_for(self, path: str | Path, section: str = "test") -> StandardScaler:
        self._require_setup()
        key = self._normalize_file_key(path)
        if section == "train":
            return self._train_scaler_for(key)
        if section == "test":
            return self._test_scaler_for(key)
        raise ValueError("section must be either 'train' or 'test'")

    def scalers_to_dict(self) -> dict[str, dict[str, dict[str, Any]]]:
        self._require_setup()
        return {
            "train": {path: scaler.to_dict() for path, scaler in self.train_scalers.items()},
            "test": {path: scaler.to_dict() for path, scaler in self.test_scalers.items()},
        }

    def _load_split_config(self) -> None:
        if self.split_config:
            return
        if not self.config_path.exists():
            raise FileNotFoundError(f"Expected split config at {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, Mapping):
            raise TypeError(f"{self.config_path} must contain a YAML mapping")

        parsed: dict[str, dict[str, FlightSplit]] = {}
        for section in ("train", "test"):
            section_raw = raw.get(section, {})
            if not isinstance(section_raw, Mapping):
                raise TypeError(f"{self.config_path}:{section} must be a mapping")
            parsed[section] = {
                str(path): FlightSplit.from_config(str(path), value)
                for path, value in section_raw.items()
            }
        self.split_config = parsed

    def _select_files(
        self,
        section: str,
        selected: Sequence[str | Path] | None,
        field_name: str,
    ) -> list[str]:
        section_config = self.split_config[section]
        if selected is None:
            return list(section_config.keys())
        if isinstance(selected, (str, Path)):
            raise TypeError(f"data.{field_name} must be a list of file names, got {selected!r}")
        files = [self._normalize_file_key(path) for path in selected]
        missing = [path for path in files if path not in section_config]
        if missing:
            raise KeyError(f"Files are missing from config.yaml:{section}: {missing}")
        return files

    def _normalize_file_key(self, path: str | Path) -> str:
        item = Path(path)
        if item.is_absolute():
            try:
                item = item.relative_to(self.root_path.resolve())
            except ValueError as exc:
                raise ValueError(f"{path!s} is outside root_path {self.root_path}") from exc
        return item.as_posix()

    def _load_series(self, path: str) -> CsvSeries:
        series = self._series_cache.get(path)
        if series is None:
            series = read_csv_series(
                self.root_path / path,
                self.feature_names,
                label_col=self.label_col,
                time_col=self.time_col,
            )
            self._series_cache[path] = series
        return series

    def _make_train_dataset(self, path: str) -> WindowDataset:
        split = self.split_config["train"][path]
        raw_series = self._load_series(path)
        split.validate_length(len(raw_series.values))
        scaler = self._train_scaler_for(path)
        series = _slice_series(raw_series, split.train_slice)
        series.values = scaler.transform(series.values)
        return self._window_dataset(
            path=path,
            split_name="train",
            values=series.values,
            labels=series.labels,
            stride=self.stride,
            drop_anomaly_windows=True,
            allow_empty=False,
        )

    def _make_val_dataset(self, path: str) -> WindowDataset | None:
        split = self.split_config["train"][path]
        raw_series = self._load_series(path)
        split.validate_length(len(raw_series.values))
        scaler = self._train_scaler_for(path)
        series = _slice_series(raw_series, split.val_slice)
        series.values = scaler.transform(series.values)
        return self._window_dataset(
            path=path,
            split_name="val",
            values=series.values,
            labels=series.labels,
            stride=self.eval_stride,
            drop_anomaly_windows=True,
            allow_empty=True,
        )

    def _train_scaler_for(self, path: str) -> StandardScaler:
        scaler = self.train_scalers.get(path)
        if scaler is not None:
            return scaler
        split = self.split_config["train"][path]
        raw_series = self._load_series(path)
        split.validate_length(len(raw_series.values))
        values = raw_series.values[split.train_slice]
        self._validate_segment_length(path, "train scaler", len(values), allow_empty=False)
        scaler = StandardScaler.fit(values, self.scaler_type)
        self.train_scalers[path] = scaler
        return scaler

    def _test_scaler_for(self, path: str) -> StandardScaler:
        scaler = self.test_scalers.get(path)
        if scaler is not None:
            return scaler
        split = self.split_config["test"][path]
        raw_series = self._load_series(path)
        split.validate_length(len(raw_series.values))
        values = raw_series.values[split.train_slice]
        self._validate_segment_length(path, "test scaler", len(values), allow_empty=False)
        scaler = StandardScaler.fit(values, self.scaler_type)
        self.test_scalers[path] = scaler
        return scaler

    def _window_dataset(
        self,
        path: str,
        split_name: str,
        values: np.ndarray,
        labels: np.ndarray | None,
        stride: int,
        drop_anomaly_windows: bool,
        allow_empty: bool,
    ) -> WindowDataset | None:
        self._validate_segment_length(path, split_name, len(values), allow_empty=allow_empty)
        if len(values) == 0:
            return None
        return WindowDataset(
            values,
            seq_len=self.seq_len,
            stride=stride,
            labels=labels,
            drop_anomaly_windows=drop_anomaly_windows,
        )

    def _validate_segment_length(
        self,
        path: str,
        split_name: str,
        length: int,
        allow_empty: bool,
    ) -> None:
        if length == 0 and allow_empty:
            return
        if length < self.seq_len:
            raise ValueError(
                f"{path!r} {split_name} segment length {length} is shorter than seq_len={self.seq_len}"
            )

    def _loader(self, dataset: Dataset | None, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting this dataloader")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def _require_setup(self) -> None:
        if self.train_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting datasets")


__all__ = ["FinetuneDataModule", "FlightSplit"]
