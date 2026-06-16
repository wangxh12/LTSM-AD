from __future__ import annotations

from pathlib import Path
from typing import Any


from src.scripts.utils import load_config, make_run_dir, save_config, save_json, seed_everything
from src.scripts import pretrain


def run_pretraining(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    run_dir = make_run_dir(config["outputs"]["root_dir"], "pretrain")
    config.setdefault("runtime", {})
    config["runtime"]["run_dir"] = str(run_dir)
    save_config(config, run_dir / "config.yaml")

    print(f">>>>> Starting pretraining <<<<<")
    datamodule = pretrain.get_datamodule(config, setup=True)
    module = pretrain.init_lightning(config)
    
    
    # pretrain
    _, best_checkpoint, best_val_loss = pretrain.train(module, datamodule, config)

    # save pretrained model
    print(f">>>>> Saving pretrained model <<<<<")
    if best_checkpoint:
        # export_module = ModelForPreTraining.load_from_checkpoint(...)
        export_module = pretrain.load_pretraining_module(config, best_checkpoint)
    else:
        raise RuntimeError("No checkpoint was saved during pretraining. Check checkpoint config.")
    
    pretrained_model_dir = pretrain.save_pretrained(
        module=export_module,
        config=config,
        checkpoint_path=best_checkpoint,
        best_val_loss=best_val_loss,
    )

    # Save summary
    summary = {
        "run_dir": str(run_dir),
        "best_checkpoint": best_checkpoint,
        "best_val_loss": best_val_loss,
        "pretrained_model_dir": str(pretrained_model_dir),
        "csv_paths": [str(path) for path in datamodule.csv_paths],
    }
    save_json(summary, run_dir / "summary.json")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Pretrain UAV time-series model.")
    parser.add_argument("--config", default="src/configs/pretrain_config.yaml", help="Path to pretraining YAML.")
    args = parser.parse_args()
    summary = run_pretraining(args.config)
    print(summary)


if __name__ == "__main__":
    main()

