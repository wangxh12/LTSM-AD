# TIME SPACE 交替

# patch嵌入

# 重构

微调范围由 `optimization.finetune_scope` 显式控制：

- `reconstruction_head`：冻结 backbone，仅训练 `reconstruction_head`
- `all`：训练全部参数

预训练使用 TimesBERT 对应的 step-wise cosine annealing：

```yaml
optimization:
  lr: 0.0001
  betas: [0.9, 0.99]
  scheduler:
    type: cosine_annealing
    t_max_steps: 30000
    eta_min: 0.0000002
trainer:
  max_steps: 30000
```
