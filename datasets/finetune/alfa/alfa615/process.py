import pandas as pd
import os

# 校验文件是否存在
required_files = ["train.csv", "val.csv", "test.csv", "label.csv"]
for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"当前目录缺失文件：{file}")

# ========== 1. 处理 train.csv：末尾新增 label 列，值全为 0 ==========
train_df = pd.read_csv("train.csv")
train_df["label"] = 0
# 强制 label 列在最后一位
cols_order = [col for col in train_df.columns if col != "label"] + ["label"]
train_df = train_df[cols_order]

# ========== 2. 处理 val.csv：末尾新增 label 列，值全为 0 ==========
val_df = pd.read_csv("val.csv")
val_df["label"] = 0
cols_order = [col for col in val_df.columns if col != "label"] + ["label"]
val_df = val_df[cols_order]

# ========== 3. 处理 test.csv：合并 label.csv 到最后一列 ==========
test_df = pd.read_csv("test.csv")
label_df = pd.read_csv("label.csv")

# 校验行数匹配，防止数据错位
if len(test_df) != len(label_df):
    raise ValueError(
        f"行数不匹配：test.csv 共 {len(test_df)} 行，label.csv 共 {len(label_df)} 行"
    )

# 取 label.csv 第一列作为标签，追加到最后一列
test_df["label"] = label_df.iloc[:, 0].values
cols_order = [col for col in test_df.columns if col != "label"] + ["label"]
test_df = test_df[cols_order]

# ========== 4. 保存结果（默认不覆盖原文件） ==========
train_df.to_csv("train_labeled.csv", index=False)
val_df.to_csv("val_labeled.csv", index=False)
test_df.to_csv("test_labeled.csv", index=False)

# 如需直接覆盖原文件，注释上面三行，取消下面三行注释
# train_df.to_csv("train.csv", index=False)
# val_df.to_csv("val.csv", index=False)
# test_df.to_csv("test.csv", index=False)

print("处理完成！已生成带标签的文件：")
print("  - train_labeled.csv")
print("  - val_labeled.csv")
print("  - test_labeled.csv")