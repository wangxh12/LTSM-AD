# LTSM-AD

run

pretrain
```bash
ssh root@h100 "cd /workspace/code/LTSM-AD && conda run -n ltsm python -m --config <config_patch>"

# eg
ssh root@h100 "cd /workspace/code/LTSM-AD && conda run -n ltsm python -m src.scripts.benchmark_pretrain"

```

finetune
```bash

ssh root@h100 "cd /workspace/code/LTSM-AD && conda run -n ltsm python -m <config_path> --is_finetuning true --pretrained_checkpoint <pretrain_ckpt>"

# eg

ssh root@h100 "cd /workspace/code/LTSM-AD && conda run -n ltsm python -m src.scripts.benchmark_finetune --is_finetuning true --pretrained_checkpoint <pretrain_ckpt>"

```