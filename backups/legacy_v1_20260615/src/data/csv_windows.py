from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import lightning as L
except ImportError:  # pragma: no cover - kept for older environments
    import pytorch_lightning as L

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset


@dataclass
class CsvSeries:
    path: Path
    values: np.ndarray
    feature_names: list[str]
    labels: np.ndarray | None = None
    time: np.ndarray | None = None


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray
    scaler_type: str = "standard"
    eps: float = 1e-6

    @classmethod
    def fit(cls, values: np.ndarray, scaler_type: str = "standard") -> "StandardScaler":
        if scaler_type == "standard":
            mean = np.nanmean(values, axis=0)
            std = np.nanstd(values, axis=0)
        elif scaler_type == "minmax":
            mean = np.nanmin(values, axis=0)
            std = np.nanmax(values, axis=0) - mean
        else:
            raise ValueError("scaler_type must be either 'standard' or 'minmax'")
        std = np.where(std < cls.eps, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32), scaler_type=scaler_type)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (values * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {"scaler_type": self.scaler_type, "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, state: dict[str, list[float]]) -> "StandardScaler":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
            scaler_type=str(state.get("scaler_type", "standard")),
        )


def _clean_numeric_frame(frame: pd.DataFrame) -> np.ndarray:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.interpolate(method="linear", limit_direction="both")
    numeric = numeric.bfill().ffill()
    numeric = numeric.fillna(0.0)
    return numeric.to_numpy(dtype=np.float32)


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


class WindowDataset(Dataset):
    def __init__(
        self,
        series: np.ndarray,
        seq_len: int,
        stride: int = 1,
        labels: np.ndarray | None = None,
        drop_anomaly_windows: bool = False,
    ) -> None:
        if series.ndim != 2:
            raise ValueError(f"Expected series shape [time, features], got {series.shape}")
        if len(series) < seq_len:
            raise ValueError(f"Series length {len(series)} is shorter than seq_len={seq_len}")
        if labels is not None and len(labels) != len(series):
            raise ValueError("Labels and series must have the same time length")

        self.series = series.astype(np.float32)
        self.labels = labels.astype(np.int64) if labels is not None else None
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.drop_anomaly_windows = drop_anomaly_windows

        starts = np.arange(0, len(series) - self.seq_len + 1, self.stride, dtype=np.int64)
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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[index])
        end = start + self.seq_len
        item = {
            "x": torch.from_numpy(self.series[start:end]),
            "start": torch.tensor(start, dtype=torch.long),
        }
        if self.labels is not None:
            item["label"] = torch.from_numpy(self.labels[start:end])
        return item


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

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        paths = sorted(self.root.rglob(self.pattern))
        if not paths:
            raise FileNotFoundError(f"No CSV files matched {self.root}/{self.pattern}")
        self.csv_paths = paths

        train_datasets: list[WindowDataset] = []
        val_datasets: list[WindowDataset] = []
        for path in paths:
            series = read_csv_series(path, self.feature_names, self.label_col, self.time_col)
            split_index = int(len(series.values) * self.split_ratio)
            split_index = max(self.seq_len, min(split_index, len(series.values) - self.seq_len))
            train_values = series.values[:split_index]
            val_values = series.values[split_index:]

            scaler = StandardScaler.fit(train_values, self.scaler_type)
            train_datasets.append(
                WindowDataset(
                    scaler.transform(train_values),
                    seq_len=self.seq_len,
                    stride=self.stride,
                    labels=series.labels[:split_index] if series.labels is not None else None,
                    drop_anomaly_windows=True,
                )
            )
            val_datasets.append(
                WindowDataset(
                    scaler.transform(val_values),
                    seq_len=self.seq_len,
                    stride=self.stride,
                    labels=series.labels[split_index:] if series.labels is not None else None,
                    drop_anomaly_windows=True,
                )
            )

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
        self.train_dataset: WindowDataset | None = None
        self.val_dataset: WindowDataset | None = None
        self.threshold_dataset: WindowDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        self.train_series = read_csv_series(self.train_path, self.feature_names, self.label_col, self.time_col)

        # train scaler
        self.scaler = StandardScaler.fit(self.train_series.values, self.scaler_type)
        values = self.scaler.transform(self.train_series.values)
        labels = self.train_series.labels

        split_index = int(len(values) * self.split_ratio)
        split_index = max(self.seq_len, min(split_index, len(values) - self.seq_len))
        self.train_dataset = WindowDataset(
            values[:split_index],
            seq_len=self.seq_len,
            stride=self.stride,
            labels=labels[:split_index] if labels is not None else None,
            drop_anomaly_windows=True,
        )
        self.val_dataset = WindowDataset(
            values[split_index:],
            seq_len=self.seq_len,
            stride=self.eval_stride,
            labels=labels[split_index:] if labels is not None else None,
            drop_anomaly_windows=True,
        )
        self.threshold_dataset = WindowDataset(
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

    def make_test_dataset(self, path: str | Path) -> WindowDataset:
        series = self.make_test_series(path)
        return WindowDataset(
            series.values,
            seq_len=self.seq_len,
            stride=self.eval_stride,
            labels=series.labels,
            drop_anomaly_windows=False,
        )

    def test_dataloader_for(self, path: str | Path) -> DataLoader:
        return self._loader(self.make_test_dataset(path), shuffle=False)

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
