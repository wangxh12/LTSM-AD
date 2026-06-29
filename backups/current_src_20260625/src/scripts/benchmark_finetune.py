from __future__ import annotations

import argparse
from pathlib import Path
from src.evaluation.evaluate import evaluate_model
from src.scripts import finetune


from src.scripts.utils import (
    load_config,
    save_config,
    save_json,
    seed_everything
)


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune and evaluate UAV reconstruction anomaly detector.")
    parser.add_argument(
        "--config",
        default="src/configs/timesbert_finetune_config.yaml",
        help="Path to finetuning/evaluation YAML.",
    )
    parser.add_argument("--is_finetuning", type=_bool_arg, default=True)
    parser.add_argument("--pretrained_checkpoint", default=None, help="Optional pretraining checkpoint for finetuning init.")
    args = parser.parse_args()

    # load config
    config = load_config(args.config)
    
    data_cfg = config["data"]
    dataset_name = str(data_cfg["dataset_name"])
    pretrained_model = config.get("pretrained_model")
    if pretrained_model is not None and not isinstance(pretrained_model, str):
        raise TypeError(f"pretrained_model must be a string path or null, got {pretrained_model!r}")
    base_model_name = (
        Path(pretrained_model).name
        if pretrained_model is not None
        else str(config.get("model_id", config.get("model_family", "model")))
    )
    mode_name = "finetuning" if args.is_finetuning else "zero_shot"
    model_name = f"{base_model_name}_{mode_name}"

    # result path
    results_root = Path(config.get("results", {}).get("root_dir"))
    result_dir = results_root / dataset_name / model_name
    result_dir.mkdir(parents=True, exist_ok=True)

    # checkpoint path
    checkpoint_cfg = config.setdefault("checkpoint", {})
    checkpoint_root = Path(checkpoint_cfg.get("root_dir", checkpoint_cfg.get("dirpath", "checkpoints")))
    checkpoint_dir = checkpoint_root / dataset_name / model_name
    
    checkpoint_cfg["dirpath"] = str(checkpoint_dir)
    logging_cfg = config.setdefault("logging", {})
    logging_cfg["name"] = str(Path(dataset_name) / model_name)
    
    save_config(config, result_dir / "config.yaml")
    
    # seed
    seed_everything(int(config.get("seed", 42)))
    
    # Initialize Lightning module and datamodule
    lightning_module = finetune.init_lightning(config)
    datamodule = finetune.get_datamodule(config, setup=True, output_dir=result_dir)
    
    # Train or run zero-shot
    checkpoint_path: str | None = None
    if args.is_finetuning:
        _, best_ckpt_path, best_val_loss = finetune.train(lightning_module, datamodule, config)

        if best_ckpt_path is None:
            raise RuntimeError("No checkpoint was saved during training. Check checkpoint config.")

        # Load best finetuned model checkpoint
        trained_model = finetune.load_finetuned_module(
            pretrained_model,
            best_ckpt_path,
            config,
            lightning_module.device,
        )
        checkpoint_path = best_ckpt_path
    else:
        trained_model = lightning_module
        best_val_loss = None
    save_config(config, result_dir / "config.yaml")

    print("Best validation loss: ", best_val_loss)
    
    # Evaluate model
    result = evaluate_model(
        trained_model,
        datamodule,
        config,
        output_dir=result_dir
    )

    threshold_percentile = float(config["evaluation"].get("threshold_percentile", 95))
    summary = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "is_finetuning": args.is_finetuning,
        "pretrained_model": pretrained_model,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "result_dir": str(result_dir),
        "checkpoint_dir": str(config["checkpoint"]["dirpath"]),
        "best_val_loss": best_val_loss,
        "threshold_percentile": threshold_percentile,
        "threshold": result.threshold,
        "metrics": result.metrics,
    }
    save_json(summary, result_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
