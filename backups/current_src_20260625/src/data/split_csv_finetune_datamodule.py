from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import lightning as L
from torch.utils.data import DataLoader, Dataset

from .csv_windows import CsvSeries, StandardScaler, WindowDataset, read_csv_series


class SplitCsvFinetuneDataModule(L.LightningDataModule):
    """DataModule for materialized train/val/test CSV finetuning.

    This DataModule expects the split files to already exist under ``root_path``.
    The scaler is fitted only on ``train`` and then reused for ``val`` and
    ``test``.
    """

    def __init__(
        self,
        root_path: str | Path,
        train_file: str | Path,
        test_file: str | Path | list[str | Path],
        feature_names: Sequence[str],
        seq_len: int,
        val_file: str | Path | None = None,
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
        self.train_file = self._normalize_file_key(train_file)
        self.val_file = self._normalize_file_key(val_file) if val_file is not None else None
        if isinstance(test_file, (str, Path)):
            self.test_files = [self._normalize_file_key(test_file)]
        elif isinstance(test_file, list):
            if not test_file:
                raise ValueError("data.test must contain at least one file")
            self.test_files = [self._normalize_file_key(path) for path in test_file]
        else:
            raise TypeError(f"data.test must be a file name or list of file names, got {test_file!r}")
        self.test_file = self.test_files[0]
        self.feature_names = list(feature_names)
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.eval_stride = int(eval_stride) if eval_stride is not None else self.stride
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.label_col = label_col
        self.time_col = time_col
        self.scaler_type = scaler_type

        self.train_files = [self.train_file]
        self.val_files = [self.val_file] if self.val_file is not None else []
        self.test_paths = [self.root_path / path for path in self.test_files]

        self.scaler: StandardScaler | None = None
        self.train_series: CsvSeries | None = None
        self.val_series: CsvSeries | None = None
        self.test_series: dict[str, CsvSeries] = {}
        self.train_dataset: WindowDataset | None = None
        self.val_dataset: WindowDataset | None = None
        self.threshold_dataset: WindowDataset | None = None
        self._test_datasets: dict[str, WindowDataset] = {}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SplitCsvFinetuneDataModule":
        data_cfg = config["data"]
        train_file = cls._require_single_file(data_cfg, "train")
        val_file = cls._optional_single_file(data_cfg, "val")
        test_file = cls._require_file_or_file_list(data_cfg, "test")
        return cls(
            root_path=data_cfg["root_path"],
            train_file=train_file,
            val_file=val_file,
            test_file=test_file,
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

        self.train_series = self._load_series(self.train_file)
        self._validate_segment_length(self.train_file, "train", len(self.train_series.values))
        self.scaler = StandardScaler.fit(self.train_series.values, self.scaler_type)

        train_values = self.scaler.transform(self.train_series.values)
        self.train_dataset = WindowDataset(
            train_values,
            seq_len=self.seq_len,
            stride=self.stride,
            labels=self.train_series.labels,
            drop_anomaly_windows=True,
        )
        self.threshold_dataset = WindowDataset(
            train_values,
            seq_len=self.seq_len,
            stride=self.eval_stride,
            labels=self.train_series.labels,
            drop_anomaly_windows=True,
        )

        if self.val_file is not None:
            self.val_series = self._load_series(self.val_file)
            self._validate_segment_length(self.val_file, "val", len(self.val_series.values))
            self.val_series.values = self.scaler.transform(self.val_series.values)
            self.val_dataset = WindowDataset(
                self.val_series.values,
                seq_len=self.seq_len,
                stride=self.eval_stride,
                labels=self.val_series.labels,
                drop_anomaly_windows=True,
            )
            self.threshold_dataset = self.val_dataset

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
        key = self._normalize_file_key(path)
        if key not in self.test_files:
            raise ValueError(f"{key!r} is not configured in data.test")
        dataset = self._test_datasets.get(key)
        if dataset is None:
            series = self.make_test_series(key)
            dataset = WindowDataset(
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
        key = self._normalize_file_key(path)
        if key not in self.test_files:
            raise ValueError(f"{key!r} is not configured in data.test")
        series = self.test_series.get(key)
        if series is None:
            series = self._load_series(key)
            self._validate_segment_length(key, "test", len(series.values))
            series.values = self._require_scaler().transform(series.values)
            self.test_series[key] = series
        return series

    def scaler_for(self, path: str | Path, section: str = "test") -> StandardScaler:
        self._require_setup()
        key = self._normalize_file_key(path)
        if section == "train" and key != self.train_file:
            raise ValueError(f"{key!r} is not configured as data.train")
        if section == "val" and key != self.val_file:
            raise ValueError(f"{key!r} is not configured as data.val")
        if section == "test" and key not in self.test_files:
            raise ValueError(f"{key!r} is not configured as data.test")
        if section not in {"train", "val", "test"}:
            raise ValueError("section must be one of 'train', 'val', or 'test'")
        return self._require_scaler()

    def scalers_to_dict(self) -> dict[str, dict[str, dict[str, Any]]]:
        self._require_setup()
        scaler_state = self._require_scaler().to_dict()
        states = {
            "train": {self.train_file: scaler_state},
            "val": {},
            "test": {path: scaler_state for path in self.test_files},
        }
        if self.val_file is not None:
            states["val"][self.val_file] = scaler_state
        return states

    @staticmethod
    def _require_single_file(data_cfg: Mapping[str, Any], field_name: str) -> str | Path:
        if field_name not in data_cfg:
            raise KeyError(f"Expected data.{field_name} in split-csv finetune config")
        value = data_cfg[field_name]
        if isinstance(value, (str, Path)):
            return value
        raise TypeError(f"data.{field_name} must be a single file name, got {value!r}")

    @staticmethod
    def _require_file_or_file_list(
        data_cfg: Mapping[str, Any],
        field_name: str,
    ) -> str | Path | list[str | Path]:
        if field_name not in data_cfg:
            raise KeyError(f"Expected data.{field_name} in split-csv finetune config")
        value = data_cfg[field_name]
        if isinstance(value, (str, Path)):
            return value
        if isinstance(value, list):
            if not value:
                raise ValueError(f"data.{field_name} must contain at least one file")
            bad = [item for item in value if not isinstance(item, (str, Path))]
            if bad:
                raise TypeError(f"data.{field_name} entries must be file names, got {bad!r}")
            return value
        raise TypeError(f"data.{field_name} must be a file name or list of file names, got {value!r}")

    @staticmethod
    def _optional_single_file(data_cfg: Mapping[str, Any], field_name: str) -> str | Path | None:
        if field_name not in data_cfg or data_cfg[field_name] is None:
            return None
        value = data_cfg[field_name]
        if isinstance(value, (str, Path)):
            return value
        raise TypeError(f"data.{field_name} must be a single file name, got {value!r}")

    def _normalize_file_key(self, path: str | Path) -> str:
        item = Path(path)
        if item.is_absolute():
            try:
                item = item.relative_to(self.root_path.resolve())
            except ValueError as exc:
                raise ValueError(f"{path!s} is outside root_path {self.root_path}") from exc
        return item.as_posix()

    def _path_for(self, key: str) -> Path:
        return self.root_path / key

    def _load_series(self, key: str) -> CsvSeries:
        path = self._path_for(key)
        if not path.exists():
            raise FileNotFoundError(path)
        return read_csv_series(
            path,
            self.feature_names,
            label_col=self.label_col,
            time_col=self.time_col,
        )

    def _validate_segment_length(self, path: str, split_name: str, length: int) -> None:
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

    def _require_scaler(self) -> StandardScaler:
        if self.scaler is None:
            raise RuntimeError("DataModule.setup() must be called before requesting the scaler")
        return self.scaler


__all__ = ["SplitCsvFinetuneDataModule"]
