# model_itransformer

`model_itransformer` 是一个 iTransformer 风格的重构模型，用于和 `model_v1`、`model_timesbert` 做异常检测实验对比。

## 核心结构

输入窗口：

```text
x: [batch, seq_len, feature_count]
```

iTransformer 的关键变化是把变量当作 token，而不是把时间点当作 token：

```text
[B, T, F]
  -> transpose
[B, F, T]
  -> Linear(seq_len -> d_model)
[B, F, d_model]
  -> TransformerEncoder over variables
[B, F, d_model]
  -> Linear(d_model -> seq_len)
[B, F, T]
  -> transpose
[B, T, F]
```

输出和输入同形状，用于计算 reconstruction error。

## 预训练

`ModelForPreTraining` 使用 point mask：

- `point_mask: [B, T]`
- 被 mask 的时间点整行特征替换为可学习 `mask_token: [F]`
- loss 只在 masked point 上计算

如果 batch 中没有 `point_mask`，LightningModule 会按照 `trainer.mask_ratio` 随机生成。

## 微调

`ModelForFinetuning` 不使用 point mask，直接做全窗口重构：

```text
MSE(reconstruction, x)
```

## 标准接口

该目录遵守 `src/scripts/README.md` 中的模型接口标准：

```python
from src.model_itransformer.model import Model
from src.model_itransformer.lightning import ModelForPreTraining, ModelForFinetuning
```

配置中使用：

```yaml
model_family: model_itransformer
```

模型参数示例：

```yaml
model:
  d_model: 128
  num_layers: 2
  num_heads: 4
  d_ffn: 256
  dropout: 0.1
  activation: gelu
  norm_first: false
```

## 和其他模型的区别

| 模型 | token 粒度 | attention 维度 | 预训练 mask |
|---|---|---|---|
| `model_v1` | 时间点 | 时间维度 | point mask |
| `model_timesbert` | 变量 patch | 变量和 patch token | patch mask |
| `model_itransformer` | 变量 | 变量维度 | point mask |
