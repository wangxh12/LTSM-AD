from __future__ import annotations

from pathlib import Path
from typing import Any


from src.scripts.utils import load_config
from src.scripts import pretrain


def configure_output_paths(config: dict[str, Any]) -> tuple[str, Path, Path]:
    model_id = str(config["model_id"])
    data_cfg = config["data"]
    root_path = data_cfg.get("root_path", data_cfg.get("root"))
    if root_path is None:
        raise KeyError("Expected data.root_path or data.root in config")

    dataset_name = str(data_cfg.get("dataset_name"))

    checkpoint_cfg = config.setdefault("checkpoint", {})
    checkpoint_root = Path(checkpoint_cfg.get("root_dir", checkpoint_cfg.get("dirpath", "checkpoints")))
    checkpoint_dir = checkpoint_root / dataset_name / model_id
    checkpoint_cfg["dirpath"] = str(checkpoint_dir)

    logging_cfg = config.setdefault("logging", {})
    logging_save_dir = Path(logging_cfg.get("save_dir", "lightning_logs"))
    logging_name = Path(dataset_name) / model_id
    logging_cfg["save_dir"] = str(logging_save_dir)
    logging_cfg["name"] = str(logging_name)
    logging_dir = logging_save_dir / logging_name

    return dataset_name, checkpoint_dir, logging_dir


def run_pretraining(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    dataset_name, checkpoint_dir, logging_dir = configure_output_paths(config)

    print(f">>>>> Starting pretraining <<<<<")
    datamodule = pretrain.get_datamodule(config, setup=True)
    module = pretrain.init_lightning(config)
    
    
    # pretrain
    _, best_checkpoint, best_val_loss = pretrain.train(module, datamodule, config)

    # save pretrained model
    print(f">>>>> Saving pretrained model <<<<<")
    if best_checkpoint:
        export_module = pretrain.load_pretraining_module(config, best_checkpoint)
    else:
        raise RuntimeError("No checkpoint was saved during pretraining. Check checkpoint config.")
    
    pretrained_model_dir = pretrain.save_pretrained(
        module=export_module,
        config=config,
        checkpoint_path=best_checkpoint,
        best_val_loss=best_val_loss,
        dataset_name=dataset_name,
        checkpoint_dir=checkpoint_dir,
        logging_dir=logging_dir,
        csv_paths=datamodule.csv_paths,
    )

    summary = {
        "dataset_name": dataset_name,
        "best_checkpoint": best_checkpoint,
        "best_val_loss": best_val_loss,
        "pretrained_model_dir": str(pretrained_model_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "logging_dir": str(logging_dir),
        "csv_paths": [str(path) for path in datamodule.csv_paths],
    }
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Pretrain UAV time-series model.")
    parser.add_argument("--config", default="src/configs/timesbert_pretrain_config.yaml", help="Path to pretraining YAML.")
    args = parser.parse_args()
    summary = run_pretraining(args.config)
    print(summary)


if __name__ == "__main__":
    main()
