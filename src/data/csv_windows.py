from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import lightning as L

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .utils import Timeseries


@dataclass
class CsvSeries:
    path: Path
    values: np.ndarray
    feature_names: list[str]
    labels: np.ndarray | None = None
    time: np.ndarray | None = None


Scaler = StandardScaler | MinMaxScaler


def make_scaler(scaler_type: str) -> Scaler:
    if scaler_type == "standard":
        return StandardScaler()
    if scaler_type == "minmax":
        return MinMaxScaler()
    raise ValueError("scaler_type must be either 'standard' or 'minmax'")


def fit_scaler(values: np.ndarray, scaler_type: str) -> Scaler:
    if values.ndim != 2:
        raise ValueError(f"Expected scaler input shape [time, features], got {values.shape}")
    scaler = make_scaler(scaler_type)
    scaler.fit(values)
    return scaler


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def scaler_to_dict(scaler: Scaler, scaler_type: str) -> dict[str, Any]:
    state: dict[str, Any] = {
        "scaler_type": scaler_type,
        "class": type(scaler).__name__,
    }
    for name in (
        "copy",
        "with_mean",
        "with_std",
        "feature_range",
        "clip",
        "mean_",
        "var_",
        "scale_",
        "min_",
        "data_min_",
        "data_max_",
        "data_range_",
        "n_features_in_",
        "n_samples_seen_",
    ):
        if hasattr(scaler, name):
            state[name] = _jsonable(getattr(scaler, name))
    return state



def _clean_numeric_frame(frame: pd.DataFrame) -> np.ndarray:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.interpolate(method="linear", limit_direction="both")
    numeric = numeric.bfill().ffill()
    numeric = numeric.fillna(0.0)
    return numeric.to_numpy(dtype=np.float32)

class UnavailableFeatureError(ValueError):
      pass

def read_csv_series(
    path: str | Path,
    feature_names: Iterable[str],
    label_col: str | None = "label",
    time_col: str | None = "time",
) -> CsvSeries:
    path = Path(path)
    features = list(feature_names)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = [name for name in features if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required feature columns: {missing}")

    values = _clean_numeric_frame(frame[features])

    labels = None
    if label_col and label_col in frame.columns:
        labels = pd.to_numeric(frame[label_col], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        labels = (labels > 0).astype(np.int64)

    time = None
    if time_col and time_col in frame.columns:
        time = pd.to_numeric(frame[time_col], errors="coerce").to_numpy(dtype=np.float64)
        if np.isnan(time).any():
            time = np.arange(len(frame), dtype=np.float64)

    return CsvSeries(path=path, values=values, feature_names=features, labels=labels, time=time)


class FlightDataset(Dataset):
    def __init__(
        self,
        series: np.ndarray,
        seq_len: int,
        stride: int = 1,
        labels: np.ndarray | None = None,
        drop_anomaly_windows: bool = False,
    ) -> None:

        if len(series) < seq_len:
            raise ValueError(f"Series length {len(series)} is shorter than seq_len={seq_len}")

        self.series = series.astype(np.float32) # [T, C]
        self.labels = labels.astype(np.int64) if labels is not None else None
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.drop_anomaly_windows = drop_anomaly_windows

        starts = np.arange(0, len(series) - self.seq_len + 1, self.stride, dtype=np.int64) # index
        if drop_anomaly_windows and self.labels is not None:
            starts = np.asarray(
                [start for start in starts if self.labels[start : start + self.seq_len].max() == 0],
                dtype=np.int64,
            )
        if len(starts) == 0:
            raise ValueError("No valid sliding windows were produced")
        self.starts = starts

    def __len__(self) -> int:
        return int(len(self.starts))

    def __getitem__(self, index: int) -> Timeseries:
        start = int(self.starts[index])
        end = start + self.seq_len
        label = (
            torch.from_numpy(self.labels[start:end])
            if self.labels is not None
            else torch.empty(0, dtype=torch.long)
        )
        return Timeseries(
            series=torch.from_numpy(self.series[start:end]),
            label=label,
        )


class PretrainDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: str | Path,
        pattern: str,
        feature_names: list[str],
        seq_len: int,
        stride: int = 1,
        split_ratio: float = 0.9,
        batch_size: int = 256,
        num_workers: int = 0,
        label_col: str | None = "label",
        time_col: str | None = "time",
        scaler_type: str = "standard",
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.pattern = pattern
        self.feature_names = feature_names
        self.seq_len = seq_len
        self.stride = stride
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label_col = label_col
        self.time_col = time_col
        self.scaler_type = scaler_type
        self.train_dataset: ConcatDataset | None = None
        self.val_dataset: ConcatDataset | None = None
        self.csv_paths: list[Path] = []
        # self.scaler = make_scaler(scaler_type)

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        paths = sorted(self.root.rglob(self.pattern)) # sead: 876
        if not paths:
            raise FileNotFoundError(f"No CSV files matched {self.root}/{self.pattern}")
        # self.csv_paths = paths
        self.csv_paths = []
        skipped_paths: list[Path] = []

        train_datasets: list[FlightDataset] = []
        val_datasets: list[FlightDataset] = []
        feature_set = set(self.feature_names)
        for path in paths:
            # 组装batch需要同样的shape，这里有缺失字段的都简单去除
            candidate = pd.read_csv(
                path,
                usecols=lambda column: column in feature_set,
            )

            missing_columns = [
            field
            for field in self.feature_names
            if field not in candidate.columns
            ]

            unavailable_columns = []
            if not missing_columns:
                numeric = candidate.apply(pd.to_numeric, errors="coerce")
                numeric = numeric.replace([np.inf, -np.inf], np.nan)

                unavailable_columns = [
                    field
                    for field in self.feature_names
                    if not numeric[field].notna().any()
                ]

            if missing_columns or unavailable_columns:
                skipped_paths.append(path)
                continue

            del candidate
            
            series = read_csv_series(path, self.feature_names, self.label_col, self.time_col)
            self.csv_paths.append(path)
            split_index = int(len(series.values) * self.split_ratio)
            split_index = max(self.seq_len, min(split_index, len(series.values) - self.seq_len))
            train_values = series.values[:split_index]
            val_values = series.values[split_index:]

            # scaler = fit_scaler(train_values, self.scaler_type)
            scaler = make_scaler(self.scaler_type)
            scaler.fit(train_values)
            train_datasets.append(
                FlightDataset(
                    scaler.transform(train_values),
                    seq_len=self.seq_len,
                    stride=self.stride,
                    labels=series.labels[:split_index] if series.labels is not None else None,
                    drop_anomaly_windows=True,
                )
            )
            val_datasets.append(
                FlightDataset(
                    scaler.transform(val_values),
                    seq_len=self.seq_len,
                    stride=self.stride,
                    labels=series.labels[split_index:] if series.labels is not None else None,
                    drop_anomaly_windows=True,
                )
            )
            
        if not train_datasets:
            raise RuntimeError(
                "No usable CSV files remained after feature validation"
            )

        print(
            f"Pretraining CSV files: used={len(self.csv_paths)}, "
            f"skipped={len(skipped_paths)}"
        )
        # exit()

        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = ConcatDataset(val_datasets)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def _loader(self, dataset: Dataset | None, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting a dataloader")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


class FinetuneDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_path: str | Path,
        test_paths: list[str | Path],
        feature_names: list[str],
        seq_len: int,
        stride: int = 1,
        eval_stride: int | None = None,
        split_ratio: float = 0.9,
        batch_size: int = 256,
        num_workers: int = 0,
        label_col: str | None = "label",
        time_col: str | None = "time",
        scaler_type: str = "standard",
    ) -> None:
        super().__init__()
        self.train_path = Path(train_path)
        self.test_paths = [Path(path) for path in test_paths]
        self.feature_names = feature_names
        self.seq_len = seq_len
        self.stride = stride
        self.eval_stride = eval_stride if eval_stride is not None else stride
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label_col = label_col
        self.time_col = time_col
        self.scaler_type = scaler_type
        self.train_series: CsvSeries | None = None
        self.train_dataset: FlightDataset | None = None
        self.val_dataset: FlightDataset | None = None
        self.threshold_dataset: FlightDataset | None = None
        self.scaler: Scaler | None = None
        self.test_files = [path.as_posix() for path in self.test_paths]
        self.scaler = make_scaler(scaler_type)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "FinetuneDataModule":
        data_cfg = config["data"]
        feature_names = data_cfg.get("target_fields", config.get("features"))
        if not feature_names:
            raise KeyError("Expected non-empty data.target_fields or top-level features in config")

        root_path = Path(data_cfg["root_path"])
        train_file = data_cfg["train"]
        test_files = cls._require_file_list(data_cfg, "test")
        train_path = root_path / train_file
        test_paths = [root_path / path for path in test_files]

        if isinstance(test_paths, (str, Path)):
            test_paths = [test_paths]
        elif not isinstance(test_paths, list):
            raise TypeError(f"data.test_paths must be a file name or list of file names, got {test_paths!r}")
        if not test_paths:
            raise ValueError("data.test_paths must contain at least one file")

        return cls(
            train_path=train_path,
            test_paths=test_paths,
            feature_names=list(feature_names),
            seq_len=int(data_cfg["seq_len"]),
            stride=int(data_cfg.get("stride", 1)),
            eval_stride=int(data_cfg.get("eval_stride", data_cfg.get("stride", 1))),
            split_ratio=float(data_cfg.get("train_val_split", data_cfg.get("split_ratio", 0.9))),
            batch_size=int(data_cfg.get("batch_size", 256)),
            num_workers=int(data_cfg.get("num_workers", 0)),
            label_col=data_cfg.get("label_col", "label"),
            time_col=data_cfg.get("timestamp_col", data_cfg.get("time_col", "time")),
            scaler_type=str(data_cfg.get("scaler_type", config.get("scaler_type", "standard"))),
        )

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None:
            return

        self.train_series = read_csv_series(self.train_path, self.feature_names, self.label_col, self.time_col)

        # train scaler
        self.scaler.fit(self.train_series.values)
        values = self.scaler.transform(self.train_series.values)
        labels = self.train_series.labels

        split_index = int(len(values) * self.split_ratio)
        split_index = max(self.seq_len, min(split_index, len(values) - self.seq_len))
        self.train_dataset = FlightDataset(
            values[:split_index],
            seq_len=self.seq_len,
            stride=self.stride,
            labels=labels[:split_index] if labels is not None else None,
            drop_anomaly_windows=True,
        )
        self.val_dataset = FlightDataset(
            values[split_index:],
            seq_len=self.seq_len,
            stride=self.eval_stride,
            labels=labels[split_index:] if labels is not None else None,
            drop_anomaly_windows=True,
        )
        self.threshold_dataset = FlightDataset(
            values,
            seq_len=self.seq_len,
            stride=self.eval_stride,
            labels=labels,
            drop_anomaly_windows=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def threshold_dataloader(self) -> DataLoader:
        return self._loader(self.threshold_dataset, shuffle=False)

    def make_test_series(self, path: str | Path) -> CsvSeries:
        if self.scaler is None:
            raise RuntimeError("DataModule.setup() must be called before creating test datasets")
        series = read_csv_series(path, self.feature_names, self.label_col, self.time_col)
        series.values = self.scaler.transform(series.values)
        return series

    def make_test_dataset(self, path: str | Path) -> FlightDataset:
        series = self.make_test_series(path)
        return FlightDataset(
            series.values,
            seq_len=self.seq_len,
            stride=self.eval_stride,
            labels=series.labels,
            drop_anomaly_windows=False,
        )

    def test_dataloader_for(self, path: str | Path) -> DataLoader:
        return self._loader(self.make_test_dataset(path), shuffle=False)

    def test_dataloader(self) -> list[DataLoader]:
        return [self.test_dataloader_for(path) for path in self.test_paths]

    def scalers_to_dict(self) -> dict[str, dict[str, Any]]:
        if self.scaler is None:
            raise RuntimeError("DataModule.setup() must be called before exporting scalers")
        scaler_state = scaler_to_dict(self.scaler, self.scaler_type)
        return {
            "train": {self.train_path.as_posix(): scaler_state},
            "test": {path.as_posix(): scaler_state for path in self.test_paths},
        }

    def _loader(self, dataset: Dataset | None, shuffle: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting a dataloader")
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    @staticmethod
    def _require_file_list(data_cfg: Mapping[str, Any], field_name: str) -> list[str | Path]:
        if field_name not in data_cfg:
            raise KeyError(f"Expected data.{field_name} in csv_windows finetune config")
        value = data_cfg[field_name]
        if isinstance(value, (str, Path)):
            return [value]
        if isinstance(value, list):
            if not value:
                raise ValueError(f"data.{field_name} must contain at least one file")
            bad = [item for item in value if not isinstance(item, (str, Path))]
            if bad:
                raise TypeError(f"data.{field_name} entries must be file names, got {bad!r}")
            return value
        raise TypeError(f"data.{field_name} must be a file name or list of file names, got {value!r}")
