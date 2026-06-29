from __future__ import annotations

import argparse

from src.data import PretrainDataModule
from src.model_timesbert import TimesBERTPretrainLitModule
from src.model import PretrainLitModule
from src.scripts.utils import (
    load_config,
    make_run_dir,
    model_config,
    model_version,
    save_config,
    save_json,
    seed_everything,
    trainer_and_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain UAV reconstruction encoder.")
    parser.add_argument(
        "--config",
        default="src/configs/pretrain_config.yaml",
        help="Path to pretraining YAML.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    run_dir = make_run_dir(config["outputs"]["root_dir"], "pretrain")
    save_config(config, run_dir / "config.yaml")

    data_cfg = config["data"]
    pretrain_cfg = config["pretrain"]
    dm = PretrainDataModule(
        root=pretrain_cfg["root"],
        pattern=pretrain_cfg.get("pattern", "ThorFlight*.csv"),
        feature_names=config["features"],
        seq_len=int(data_cfg["seq_len"]),
        stride=int(data_cfg.get("stride", 1)),
        split_ratio=float(data_cfg.get("train_val_split", 0.9)),
        batch_size=int(pretrain_cfg.get("batch_size", data_cfg.get("batch_size", 256))),
        num_workers=int(data_cfg.get("num_workers", 0)),
        label_col=data_cfg.get("label_col", "label"),
        time_col=data_cfg.get("time_col", "time"),
        scaler_type=str(data_cfg.get("scaler_type", config.get("scaler_type", "standard"))),
    )
    dm.setup("fit")

    version = model_version(config)
    if version in {"v2", "timesbert"}:
        module = TimesBERTPretrainLitModule(
            model=model_config(config),
            lr=float(pretrain_cfg.get("lr", config["optimization"].get("lr", 2e-4))),
            weight_decay=float(config["optimization"].get("weight_decay", 1e-2)),
            loss=str(config["optimization"].get("loss", "mse")),
            mask_ratio=float(pretrain_cfg.get("mask_ratio", 0.25)),
        )
    else:
        module = PretrainLitModule(
            model=model_config(config),
            lr=float(pretrain_cfg.get("lr", config["optimization"].get("lr", 1e-3))),
            weight_decay=float(config["optimization"].get("weight_decay", 1e-4)),
            loss=str(config["optimization"].get("loss", "mae")),
            mask_ratio=float(pretrain_cfg.get("mask_ratio", 0.25)),
            mask_unit_size=int(pretrain_cfg.get("mask_unit_size", 4)),
        )
    trainer_cfg = dict(config["trainer"])
    trainer_cfg.update(pretrain_cfg.get("trainer", {}))
    trainer, checkpoint = trainer_and_checkpoint(run_dir, trainer_cfg)
    trainer.fit(module, datamodule=dm)

    summary = {
        "run_dir": str(run_dir),
        "best_checkpoint": checkpoint.best_model_path,
        "best_val_loss": float(checkpoint.best_model_score.cpu()) if checkpoint.best_model_score is not None else None,
        "csv_paths": [str(path) for path in dm.csv_paths],
    }
    save_json(summary, run_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
