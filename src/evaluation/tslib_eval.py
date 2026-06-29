from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
import lightning as L
from torch.utils.data import DataLoader

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from src.data import data_factory
from src.evaluation.utils import adjustment
from src.scripts.utils import save_json
from src.data.data_factory import data_provider



def _make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def _scores_from_outputs(outputs: dict[str, np.ndarray], score_mode: str) -> np.ndarray:
    if score_mode == "mae":
        return outputs["scores"]
    if score_mode == "mse":
        error = outputs["reconstruction"] - outputs["target"]
        return np.mean(error ** 2, axis=-1).astype(np.float32)
    raise ValueError(f"Unsupported evaluation.tslib.score_mode: {score_mode!r}")


def _binary_metrics(gt: np.ndarray, pred: np.ndarray, threshold: float) -> dict[str, float]:
    accuracy = accuracy_score(gt, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt,
        pred,
        average="binary",
        zero_division=0,
    )
    return {
        "tslib_threshold": float(threshold),
        "tslib_accuracy": float(accuracy),
        "tslib_precision": float(precision),
        "tslib_recall": float(recall),
        "tslib_f1": float(f1),
    }


def tslib_eval(model: L.LightningModule, data, config, output_dir: str | Path):
    
    setting = 'anomaly_detection_{}'.format(config.model_id)
    
    tslib_cfg = config.get("evaluation", {}).get("tslib", {})
    
    anomaly_ratio = float(tslib_cfg["anomaly_ratio"])
    score_mode = str(tslib_cfg.get("score_mode", "mae"))
    anomaly_criterion = nn.MSELoss(reduce=False)
    
    device = model.device
    model.to(device)
    model.eval()
    
    data_cfg = config.get("data", {})
    batch_size = int(data_cfg.get("batch_size", 256))
    num_workers = int(data_cfg.get("num_workers", 0))
    output_dir = Path(output_dir)
    
    threshold_dataset = getattr(data, "train_dataset", None)
    train_loader= _make_loader(threshold_dataset, batch_size, num_workers)

    # (1) stastic on the train set
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            # reconstruction
            outputs = model(batch_x)
            # criterion
            score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)

    attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
    train_energy = np.array(attens_energy)
    
    # rows = []
    for test_path in data.test_files:
        # (2) find the threshold
        test_path = Path(test_path)
        dataset_name = test_path.stem

        dataset = data.make_test_dataset(test_path)
        series = data.make_test_series(test_path)
        test_loader = _make_loader(dataset, batch_size, num_workers)
        
        attens_energy = []
        test_labels = []
        for i, batch in enumerate(test_loader):
            batch_x = batch.series.float().to(device)
            # reconstruction
            outputs = model(batch_x)
            # criterion
            score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch.label)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - anomaly_ratio)
        print("Threshold :", threshold)
        
        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))


        f = open(output_dir / "result_anomaly_detection.txt", 'a')
        setting += dataset_name + test_path
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()
        return
    
    