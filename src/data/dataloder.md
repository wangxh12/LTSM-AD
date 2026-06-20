## 一些关于数据加载的设想

### 方案1
每一架次用自己的train尺度去归一化val和test
dir:
```
root_path
├── config.yaml
├── flight1.csv
├── flight2.csv
├── flight3.csv
```

其中 yaml 文件把数据集划分好了
```yaml
train: # 训练集和测试集从这里选取
    flight1.csv: 0:1000:2000 # 表示[0-1000)是训练集, [1000, 2000)是验证集,剩下的是测试集,但是这是在train下面，所以没有用到
    flight2.csv: 0:3000:3000 # 表示该架次没有验证集, 验证集可以为空，但是训练集不能为空, 否则该架次没有归一化尺度
    flight3.csv: 0:3000:3000 
    # ...

test:
    flight1.csv: 0:1000:2000 # 用train:[0, 1000)的尺度来归一化test[2000:-1], 后用于测试
    flight5.csv: 0:1000:2000
    # ...

```

配置中已经把数据范围和归一化尺度都确定了，按照数据集范围，生成滑窗样本后放入数据集

那这份LightningDataModule只需要根据加载config来处理数据集就好了
finetune配置中只需要在data下配置一个root_path就好了，默认规则就是读取root_path/config.yaml, 也就无需配置别的参数，保持DataModule的干净接口

finetune_config.yaml可以包含如下结构：
```yaml
data:
    root_path: xxx
    train: 
        - flight1.csv
        - flight2.csv
        - flight3.csv
    val: 
        - flight1.csv
        - flight2.csv
        - flight3.csv
    test: 
        - flight1.csv
        - flight5.csv
```
train val test中定义了训练集、验证集、测试集用到的文件，然后对于这个文件就去config中找对应的配置决定用那些片段，这样子可以灵活配置

### 方案2
采取拼接方案，
- 将正常飞行架次和异常飞行架次的 pre-fault 正常段组成 train dataset 和 val dataset: train.csv和val.csv
- 将异常片段拼接为test.csv

这种情况下数据集目录：
```
root_path
├── train.csv
├── val.csv
├── test.csv
```

这个方案可操作空间很大，具体如何拼接会有很多种方式：
- 拼接时只采用 部分 架次，还是 全部 架次，是只选用某天还是跨天
- 需要按日期来拼接吗？

我的一个想法是这样拼接
```
假如用的是这三个文件
划分:
flight1: |train_1|val_1|test_1|
flight2: |train_2|val_2|test_2|
flight3: |train_3|val_3|test_3|

最后数据集为：
train.csv: = concat(train_1, train_2, train_3)
val.csv: = concat(val_1, val_2, val_3)
test.csv: = concat(test_1, test_2, test_3)

在DataModule中用train.csv的尺度去归一化val和test
```

finetune配置中，需要在data下配置一个root_path、train、val、test, 如:
```yaml
data:
    root_path: xxx
    train: train.csv
    val: val.csv
    test: test.csv
```
