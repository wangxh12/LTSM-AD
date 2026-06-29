from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.evaluate import evaluate_model
from src.scripts import finetune
from src.scripts.utils import device_from_trainer_config, load_config, save_config, save_json, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved finetuning checkpoint without retraining.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to a Lightning .ckpt file.")
    parser.add_argument("--output_name", required=True, help="Explicit result directory name under results/<dataset>.")
    parser.add_argument("--threshold_percentile", type=float, default=None)
    parser.add_argument("--score_method", default=None)
    parser.add_argument("--reconstruction_weight", type=float, default=None)
    parser.add_argument("--ridge_multiplier", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.threshold_percentile is not None:
        config["evaluation"]["threshold_percentile"] = float(args.threshold_percentile)
    if args.score_method is not None:
        config["evaluation"].setdefault("score", {})["method"] = args.score_method
    if args.reconstruction_weight is not None:
        score_cfg = config["evaluation"].setdefault("score", {})
        if score_cfg.get("method") != "reconstruction_mahalanobis":
            raise ValueError("--reconstruction_weight requires evaluation.score.method=reconstruction_mahalanobis")
        score_cfg["reconstruction_weight"] = float(args.reconstruction_weight)
    if args.ridge_multiplier is not None:
        score_cfg = config["evaluation"].setdefault("score", {})
        if score_cfg.get("method") != "reconstruction_mahalanobis":
            raise ValueError("--ridge_multiplier requires evaluation.score.method=reconstruction_mahalanobis")
        score_cfg["ridge_multiplier"] = float(args.ridge_multiplier)

    seed_everything(int(config.get("seed", 42)))
    dataset_name = str(config["data"]["dataset_name"])
    results_root = Path(config.get("results", {}).get("root_dir"))
    result_dir = results_root / dataset_name / args.output_name
    result_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, result_dir / "config.yaml")

    datamodule = finetune.get_datamodule(config, setup=True, output_dir=result_dir)
    device = device_from_trainer_config(config)
    model = finetune.load_finetuned_module(
        config.get("pretrained_model"),
        args.checkpoint,
        config,
        map_location=device,
    ).to(device)

    result = evaluate_model(model, datamodule, config, output_dir=result_dir)
    summary = {
        "dataset_name": dataset_name,
        "model_name": args.output_name,
        "checkpoint": args.checkpoint,
        "result_dir": str(result_dir),
        "threshold_percentile": float(config["evaluation"].get("threshold_percentile", 95)),
        "threshold": result.threshold,
        "metrics": result.metrics,
    }
    save_json(summary, result_dir / "summary.json")
    print(summary)


if __name__ == "__main__":
    main()
