from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import lightning as L
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .csv_windows import (
    CsvSeries,
    FlightDataset,
    Scaler,
    fit_scaler,
    read_csv_series,
    scaler_to_dict,
)


class GlobalCsvFinetuneDataModule(L.LightningDataModule):
    """Train/test CSV DataModule with one scaler fitted on all training flights.

    Every CSV is windowed independently, so no window can cross a flight
    boundary. Validation data is intentionally unsupported: this DataModule is
    for zero-shot evaluation where the training flights provide the scaler and
    the training reconstruction-error distribution.
    """

    def __init__(
        self,
        root_path: str | Path,
        train_files: Sequence[str | Path],
        test_files: Sequence[str | Path],
        feature_names: Sequence[str],
        seq_len: int,
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
        self.train_files = self._normalize_file_list(train_files, "train")
        self.val_files: list[str] = []
        self.test_files = self._normalize_file_list(test_files, "test")
        self.test_paths = [self.root_path / path for path in self.test_files]
        self.test_file = self.test_files[0]

        self.feature_names = list(feature_names)
        if not self.feature_names:
            raise ValueError("feature_names must contain at least one field")
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.eval_stride = int(eval_stride) if eval_stride is not None else self.stride
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.label_col = label_col
        self.time_col = time_col
        self.scaler_type = str(scaler_type)

        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.eval_stride <= 0:
            raise ValueError("eval_stride must be positive")

        self.scaler: Scaler | None = None
        self.train_dataset: ConcatDataset | None = None
        self.val_dataset: None = None
        self.threshold_dataset: ConcatDataset | None = None
        self._raw_series: dict[str, CsvSeries] = {}
        self.test_series: dict[str, CsvSeries] = {}
        self._test_datasets: dict[str, FlightDataset] = {}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "GlobalCsvFinetuneDataModule":
        data_cfg = config["data"]
        val_files = data_cfg.get("val")

        return cls(
            root_path=data_cfg["root_path"],
            train_files=cls._require_file_list(data_cfg, "train"),
            test_files=cls._require_file_list(data_cfg, "test"),
            feature_names=data_cfg["target_fields"],
            seq_len=int(data_cfg["seq_len"]),
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

        train_series = [self._load_raw_series(path) for path in self.train_files]
        for path, series in zip(self.train_files, train_series):
            self._validate_length(path, "train", len(series.values))
            if series.labels is not None and np.any(series.labels != 0):
                raise ValueError(f"Training file {path!r} contains anomaly labels")

        self.scaler = fit_scaler(
            np.concatenate([series.values for series in train_series], axis=0),
            self.scaler_type,
        )

        train_datasets = [
            FlightDataset(
                self._require_scaler().transform(series.values),
                seq_len=self.seq_len,
                stride=self.stride,
                labels=series.labels,
                drop_anomaly_windows=True,
            )
            for series in train_series
        ]
        self.train_dataset = ConcatDataset(train_datasets)
        self.threshold_dataset = self.train_dataset

    def train_dataloader(self) -> DataLoader:
        self._require_setup()
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> None:
        self._require_setup()
        return None

    def threshold_dataloader(self) -> DataLoader:
        self._require_setup()
        return self._loader(self.threshold_dataset, shuffle=False)

    def test_dataloader(self) -> list[DataLoader]:
        self._require_setup()
        return [self.test_dataloader_for(path) for path in self.test_files]

    def test_dataloader_for(self, path: str | Path) -> DataLoader:
        return self._loader(self.make_test_dataset(path), shuffle=False)

    def make_test_dataset(self, path: str | Path) -> FlightDataset:
        self._require_setup()
        key = self._configured_test_key(path)
        dataset = self._test_datasets.get(key)
        if dataset is None:
            series = self.make_test_series(key)
            dataset = FlightDataset(
                series.values,
                seq_len=self.seq_len,
                stride=self.eval_stride,
                labels=series.labels,
                drop_anomaly_windows=False,
            )
            self._test_datasets[key] = dataset
        return dataset

    def make_test_series(self, path: str | Path) -> CsvSeries:
        self._require_setup()
        key = self._configured_test_key(path)
        series = self.test_series.get(key)
        if series is None:
            raw = self._load_raw_series(key)
            self._validate_length(key, "test", len(raw.values))
            series = CsvSeries(
                path=raw.path,
                values=self._require_scaler().transform(raw.values),
                feature_names=raw.feature_names,
                labels=raw.labels,
                time=raw.time,
            )
            self.test_series[key] = series
        return series

    def scaler_for(self, path: str | Path, section: str = "test") -> Scaler:
        self._require_setup()
        key = self._normalize_file_key(path)
        if section == "train":
            if key not in self.train_files:
                raise ValueError(f"{key!r} is not configured in data.train")
        elif section == "test":
            if key not in self.test_files:
                raise ValueError(f"{key!r} is not configured in data.test")
        else:
            raise ValueError("section must be either 'train' or 'test'")
        return self._require_scaler()

    def scalers_to_dict(self) -> dict[str, dict[str, dict[str, Any]]]:
        self._require_setup()
        scaler_state = scaler_to_dict(self._require_scaler(), self.scaler_type)
        return {
            "train": {path: scaler_state for path in self.train_files},
            "val": {},
            "test": {path: scaler_state for path in self.test_files},
        }

    @staticmethod
    def _require_file_list(
        data_cfg: Mapping[str, Any],
        field_name: str,
    ) -> list[str | Path]:
        if field_name not in data_cfg:
            raise KeyError(f"Expected data.{field_name} in global_csv config")
        value = data_cfg[field_name]
        if not isinstance(value, list) or not value:
            raise TypeError(f"data.{field_name} must be a non-empty list of CSV files")
        invalid = [item for item in value if not isinstance(item, (str, Path))]
        if invalid:
            raise TypeError(f"data.{field_name} contains invalid file names: {invalid!r}")
        return value

    def _normalize_file_list(
        self,
        paths: Sequence[str | Path],
        field_name: str,
    ) -> list[str]:
        normalized = [self._normalize_file_key(path) for path in paths]
        duplicates = sorted({path for path in normalized if normalized.count(path) > 1})
        if duplicates:
            raise ValueError(f"data.{field_name} contains duplicate files: {duplicates}")
        return normalized

    def _normalize_file_key(self, path: str | Path) -> str:
        item = Path(path)
        if item.is_absolute():
            try:
                item = item.resolve().relative_to(self.root_path.resolve())
            except ValueError as exc:
                raise ValueError(f"{path!s} is outside root_path {self.root_path}") from exc
        return item.as_posix()

    def _configured_test_key(self, path: str | Path) -> str:
        key = self._normalize_file_key(path)
        if key not in self.test_files:
            raise ValueError(f"{key!r} is not configured in data.test")
        return key

    def _load_raw_series(self, key: str) -> CsvSeries:
        series = self._raw_series.get(key)
        if series is not None:
            return series
        path = self.root_path / key
        if not path.is_file():
            raise FileNotFoundError(path)
        series = read_csv_series(
            path,
            self.feature_names,
            label_col=self.label_col,
            time_col=self.time_col,
        )
        self._raw_series[key] = series
        return series

    def _validate_length(self, path: str, split_name: str, length: int) -> None:
        if length < self.seq_len:
            raise ValueError(
                f"{path!r} {split_name} length {length} is shorter than seq_len={self.seq_len}"
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

    def _require_scaler(self) -> Scaler:
        if self.scaler is None:
            raise RuntimeError("DataModule.setup() must be called before requesting the scaler")
        return self.scaler


__all__ = ["GlobalCsvFinetuneDataModule"]
