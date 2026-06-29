# model_v1 模型说明

`model_v1` 是一个点级重构模型：每个时间步的完整多变量特征向量被编码成一个 token，然后在时间维度上做双向 Transformer 编码，最后逐时间步重构原始特征向量。

它和 `model_timesbert` 的最大区别是：

- `model_v1` 的 token 粒度是「一个时间步一个 token」。
- `model_timesbert` 的 token 粒度是「每个变量按 patch 切分后的 token」，并把多变量序列组织成类似文档的结构。
- `model_v1` 使用 Lightning checkpoint 保存和加载；`model_timesbert` 使用 `save_pretrained/from_pretrained` 风格的本地模型目录。

## 文件结构

```text
src/model_v1/
├── transformer.py  # 网络结构
├── lightning.py    # LightningModule、loss、checkpoint 加载和预训练权重初始化
└── README.md
```

## 输入输出

模型输入是归一化后的滑窗序列：

```text
values:     [batch, seq_len, feature_count]
valid_mask: [batch, seq_len]
point_mask: [batch, seq_len] | None
```

输出是重构结果：

```text
reconstruction: [batch, seq_len, feature_count]
```

当前 `WindowDataset` 返回 `Timeseries`，模型通过 `batch.series` 读取窗口：

```python
values = batch.series
```

所以现有 DataModule 不需要为了 `model_v1` 改 batch 字段。

## 网络结构

整体结构：

```text
values [B, T, F]
  -> PointWiseTokenizer
  -> optional point mask replacement
  -> TimeEncoder
  -> ReconstructionHead
  -> reconstruction [B, T, F]
```

### PointWiseTokenizer

位置：`transformer.py::PointWiseTokenizer`

每个时间步的完整特征向量 `x_t` 会被映射成一个 hidden token：

```text
value_projection(x_t) * silu(gate_projection(x_t))
```

特点：

- 不按变量拆 token。
- 不按 patch 拆 token。
- 每个 token 直接代表一个时间点的所有传感器状态。

### Mask Token

位置：`transformer.py::ReconstructionModel`

如果传入 `point_mask`，被 mask 的时间步 token 会被替换成可学习的 `mask_token`：

```python
tokens = torch.where(point_mask[..., None], mask_token, tokens)
```

注意：当前通用 `WindowDataset` 不会自动生成 `point_mask`。如果要做真正的 masked point pretraining，需要在 LightningModule 或 Dataset 中生成 `point_mask`。

### TimeEncoder

位置：`transformer.py::TimeEncoder`

由多层 `EncoderLayer` 组成，每层结构：

```text
RMSNorm
  -> bidirectional self-attention over time with RoPE
  -> residual
  -> RMSNorm
  -> SwiGLU FFN
  -> residual
```

注意力只沿时间维度做：

```text
tokens: [batch, seq_len, hidden_size]
```

它不会显式建模「变量 token」之间的 attention；变量之间的交互被压缩在每个时间步的 feature-vector token 内。

### TimeSelfAttention

位置：`transformer.py::TimeSelfAttention`

特点：

- 双向 self-attention。
- 使用 RoPE 时间位置编码。
- 使用 `valid_mask` 屏蔽无效时间步。
- 不是 causal attention。

### SwiGLUFeedForward

位置：`transformer.py::SwiGLUFeedForward`

FFN 形式：

```text
silu(gate_projection(x)) * up_projection(x)
  -> down_projection
```

### ReconstructionHead

位置：`transformer.py::ReconstructionHead`

逐时间步把 hidden token 投影回原始特征维度：

```text
[batch, seq_len, hidden_size] -> [batch, seq_len, feature_count]
```

## LightningModule

位置：`lightning.py::ReconstructionLightningModule`

初始化参数：

```python
ReconstructionLightningModule(
    feature_columns=[...],
    model_config={
        "hidden_size": 256,
        "num_layers": 4,
        "num_heads": 8,
        "ffn_size": 1024,
        "dropout": 0.1,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
    },
    objective="pretrain" | "finetune",
    learning_rate=...,
    weight_decay=...,
)
```

loss：

```text
MSE(reconstruction, target)
```

如果 `objective == "pretrain"` 且 batch 中存在 `point_mask`，只在 masked point 上计算 loss。

如果没有 `point_mask`，则会退化为全点重构 loss。

## 和 model_timesbert 的差异

| 项目 | model_v1 | model_timesbert |
|---|---|---|
| token 粒度 | 每个时间步一个 token | 每个变量的 patch 一个 token |
| 变量交互 | 在 feature-vector token 内混合 | 变量 token 之间通过 Transformer 交互 |
| 时间位置编码 | RoPE | learned patch position embedding |
| 归一化 | RMSNorm | TransformerEncoderLayer 内部 norm / output LayerNorm |
| FFN | SwiGLU | TransformerEncoderLayer FFN |
| 预训练 mask | 需要 `point_mask` | Lightning 内部生成 patch mask |
| 保存格式 | Lightning `.ckpt` | `save_pretrained` 本地模型目录 |
| 加载方式 | `load_checkpoint_module()` / `initialize_from_pretrained()` | `Model.from_pretrained(path)` |

## 标准脚本接口

`model_v1` 现在按 `src/scripts/README.md` 中的模型包标准对外暴露接口。实验配置通过 `model_family` 选择模型包：

```yaml
model_family: model_v1
```

模型结构参数使用 `model_v1` 字段：

```yaml
model:
  hidden_size: 128
  num_layers: 2
  num_heads: 4
  ffn_size: 256
  dropout: 0.1
  rms_norm_eps: 0.000001
  rope_theta: 10000.0
```

`model.py` 暴露：

```python
class Model(nn.Module, ModelHubMixin):
    ...
```

并满足：

```python
Model(config).model
Model.from_pretrained(path).model
```

`lightning.py` 暴露：

```python
class ModelForPreTraining(L.LightningModule):
    def __init__(self, backbone, config): ...

class ModelForFinetuning(L.LightningModule):
    def __init__(self, pretrained_backbone, config): ...
```

因此 benchmark 中统一逻辑可以直接工作：

```python
Model, ModelForPreTraining, ModelForFinetuning = load_model_package(config["model_family"])
```

预训练导出格式与其他标准模型一致：

```text
PretrainedModels/<model_id>/
├── config.json
├── model.safetensors
├── metadata.json
└── training_config.yaml
```

## 评估逻辑

当前评估代码调用：

```python
reconstruction = model(x)
```

`ReconstructionLightningModule.forward(values, valid_mask=None, point_mask=None)` 可以兼容这个调用，所以评估部分原则上不需要改。

## 适合的实验问题

`model_v1` 更适合先作为一个简单明确的 baseline：

- 不做变量级 patch。
- 不做变量间 token attention。
- 只验证「时间维度 Transformer + 重构误差」是否足够有效。

如果它效果接近 `model_timesbert`，说明 patch/变量 token 设计不是主要瓶颈；如果差距明显，说明变量级建模或 patch 级 mask 可能是关键因素。
