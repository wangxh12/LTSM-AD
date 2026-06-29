# 模型接口标准

本文档说明 `src/scripts` 下预训练和微调脚本应依赖的模型包标准。目标是新增 `model_xxx` 时只改配置，不改 benchmark 主流程。

## 配置字段

每个实验配置应声明模型包名：

```yaml
model_family: model_timesbert
```

或：

```yaml
model_family: model_v1
```

`model_family` 对应 `src/` 下的模型目录名：

```text
src/model_timesbert/
src/model_v1/
src/model_xxx/
```

## 目录规范

每个模型目录至少提供：

```text
src/model_xxx/
├── model.py
├── lightning.py
└── backbone.py
```

内部文件可以继续拆分，但对外必须满足下面的 import contract。

## model.py 标准

`model.py` 必须暴露：

```python
class Model(nn.Module, ModelHubMixin):
    ...
```

并满足：

```python
wrapper = Model(config)
backbone = wrapper.model
```

以及：

```python
wrapper = Model.from_pretrained(model_path)
backbone = wrapper.model
```

`Model` 负责：

- 从完整实验配置中解析网络结构参数。
- 构造真正的 backbone，并放在 `self.model`。
- 支持 `save_pretrained(path)`。
- 支持 `from_pretrained(path)`。

## lightning.py 标准

`lightning.py` 必须暴露：

```python
class ModelForPreTraining(L.LightningModule):
    def __init__(self, backbone, config): ...

class ModelForFinetuning(L.LightningModule):
    def __init__(self, pretrained_backbone, config): ...
```

其中：

- `backbone` 是 `Model(config).model`。
- `pretrained_backbone` 是 `Model.from_pretrained(path).model`。
- 两个 LightningModule 都应接收 batch 中的 `batch["x"]`。
- `forward(x)` 应返回与 `x` 同形状的 reconstruction。

## 动态加载

脚本侧可以统一这样加载模型包：

```python
from importlib import import_module


def load_model_package(model_family: str):
    model_module = import_module(f"src.{model_family}.model")
    lightning_module = import_module(f"src.{model_family}.lightning")
    return (
        model_module.Model,
        lightning_module.ModelForPreTraining,
        lightning_module.ModelForFinetuning,
    )
```

预训练：

```python
Model, ModelForPreTraining, _ = load_model_package(config["model_family"])
backbone = Model(config).model
module = ModelForPreTraining(backbone=backbone, config=config)
```

微调：

```python
Model, _, ModelForFinetuning = load_model_package(config["model_family"])
pretrained_backbone = Model.from_pretrained(config["pretrained_model"]).model
module = ModelForFinetuning(pretrained_backbone=pretrained_backbone, config=config)
```

## 新增模型 checklist

新增 `src/model_xxx` 时确认：

- `src/model_xxx/model.py` 存在。
- `model.py` 暴露 `Model`。
- `Model(config).model` 可用。
- `Model.from_pretrained(path).model` 可用。
- `src/model_xxx/lightning.py` 存在。
- `lightning.py` 暴露 `ModelForPreTraining`。
- `lightning.py` 暴露 `ModelForFinetuning`。
- 两个 LightningModule 的构造参数和标准一致。
- `forward(x)` 返回 `[batch, seq_len, num_features]`。
- 配置里设置 `model_family: model_xxx`。

## 不建议

不要在 benchmark 主流程里为每个模型写大量分支：

```python
if model_family == "model_a":
    ...
elif model_family == "model_b":
    ...
```

模型差异应封装在各自的 `model.py` 和 `lightning.py` 里，脚本只依赖统一接口。
