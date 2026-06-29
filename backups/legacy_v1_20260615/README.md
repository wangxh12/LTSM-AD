# Legacy V1 Backup

Source snapshot: `7915973` (`feat: add pretraining export to HuggingFace-style format`).

This backup reproduces the legacy `model_v1` reconstruction experiment protocol
used for the June 15-16, 2026 Minnesota and ALFA runs. The original source
imports the package as `src`, so the backup keeps that package name at
`backups/legacy_v1_20260615/src/`.

Legacy reproduction configs:

- `src/configs/legacy_v1_minnesota121_config.yaml`
- `src/configs/legacy_v1_alfa615_config.yaml`

The snapshot contains one compatibility edit in `src/model_timesbert/__init__.py`:
it exports `TimesBERTLitModule` so the old runner's unused unconditional import
does not prevent a `model_version: v1` run.


重跑：
• 可以重跑。注意不要用 python -m src...，那会导入当前 src。要用 legacy 脚本路径，并把 PYTHONPATH 指到备份目录。

  我刚修了 legacy 备份里的几个 runner 兼容问题，并做了本地语法检查。远程 --help 检查因为审批通道被拒没跑成，所以你可以直接在服务器上跑：

  cd /workspace/code/LTSM-AD

  env PYTHONPATH=/workspace/code/LTSM-AD/backups/legacy_v1_20260615 \
  CUDA_VISIBLE_DEVICES=0 \
  conda run --no-capture-output -n ltsm \
  python backups/legacy_v1_20260615/src/scripts/benchmark_finetune.py \
    --config backups/legacy_v1_20260615/src/configs/legacy_v1_minnesota121_config.yaml \
    --is_finetuning true

  如果只是先检查导入：

  cd /workspace/code/LTSM-AD

  env PYTHONPATH=/workspace/code/LTSM-AD/backups/legacy_v1_20260615 \
  conda run -n ltsm \
  python backups/legacy_v1_20260615/src/scripts/benchmark_finetune.py --help

  输出会写到：

  outputs/uav_ad_model_v1_legacy_minnesota/finetune_<timestamp>/

  图会在类似这里：

  outputs/uav_ad_model_v1_legacy_minnesota/finetune_<timestamp>/test/ThorFlight121/ThorFlight121_scores.png

  这对应你图里的 ThorFlight121。之前已有一次 legacy run 的 summary 在：

  outputs/uav_ad_model_v1_legacy_minnesota/finetune_20260622_155331/summary.json

  那次 best_f1=0.9969，应该就是你想复现的那类效果。