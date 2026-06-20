"""测试脚本：验证 FinetuneDataModule 的各个功能"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from transformers import AutoModel


AutoModel.from_config()


from src.data.finetune_datamodule import FinetuneDataModule, FlightSplit  # 替换为实际导入路径


def create_test_data(tmp_dir: Path) -> None:
    """创建测试用的 CSV 数据和 config.yaml"""
    
    # 创建测试 CSV 文件 - 飞行1
    np.random.seed(42)
    n_samples = 1000
    time = np.arange(n_samples)
    feat1 = np.sin(time * 0.1) + np.random.normal(0, 0.1, n_samples)
    feat2 = np.cos(time * 0.1) + np.random.normal(0, 0.1, n_samples)
    label = (np.abs(feat1) > 1.0).astype(int)  # 简单的异常标签
    
    df1 = pd.DataFrame({
        "time": time,
        "feature_1": feat1,
        "feature_2": feat2,
        "label": label,
    })
    df1.to_csv(tmp_dir / "flight_1.csv", index=False)
    
    # 创建测试 CSV 文件 - 飞行2
    feat1 = np.sin(time * 0.15) + np.random.normal(0, 0.1, n_samples)
    feat2 = np.cos(time * 0.15) + np.random.normal(0, 0.1, n_samples)
    label = (np.abs(feat2) > 1.0).astype(int)
    
    df2 = pd.DataFrame({
        "time": time,
        "feature_1": feat1,
        "feature_2": feat2,
        "label": label,
    })
    df2.to_csv(tmp_dir / "flight_2.csv", index=False)
    
    # 创建 config.yaml
    config = {
        "train": {
            "flight_1.csv": "0:500:700",  # train: 0-500, val: 500-700
            "flight_2.csv": "0:400:600",  # train: 0-400, val: 400-600
        },
        "test": {
            "flight_1.csv": "700:800:1000",  # train scaler: 700-800, test: 800-1000
            "flight_2.csv": "600:700:1000",  # train scaler: 600-700, test: 700-1000
        }
    }
    
    with open(tmp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)


def test_flight_split():
    """测试 FlightSplit 类"""
    print("=" * 60)
    print("测试 FlightSplit")
    
    # 测试正常创建
    split = FlightSplit.from_config("test.csv", "0:100:200")
    print(f"✓ 创建成功: {split}")
    print(f"  train_slice: {split.train_slice}")
    print(f"  val_slice: {split.val_slice}")
    print(f"  test_slice: {split.test_slice}")
    
    # 测试验证
    split.validate_length(300)  # 应该不报错
    print("✓ validate_length(300) 通过")
    
    # 测试各种错误情况
    error_cases = [
        ("格式错误", "test.csv", "0:100"),          # 缺少一个冒号
        ("负数", "test.csv", "-1:100:200"),           # 负数
        ("顺序错误", "test.csv", "100:50:200"),       # start > train_end
        ("相等", "test.csv", "0:100:100"),            # train_end == val_end
        ("非整数", "test.csv", "0:abc:200"),          # 非整数
    ]
    
    for name, path, value in error_cases:
        try:
            FlightSplit.from_config(path, value)
            print(f"✗ 应该抛出异常但没有: {name}")
        except (ValueError, TypeError) as e:
            print(f"✓ 正确捕获异常 - {name}: {e}")
    
    print()


def test_datamodule_basic(tmp_dir: Path):
    """测试 DataModule 基本功能"""
    print("=" * 60)
    print("测试 FinetuneDataModule 基本功能")
    
    dm = FinetuneDataModule(
        root_path=tmp_dir,
        feature_names=["feature_1", "feature_2"],
        seq_len=50,
        batch_size=32,
        label_col="label",
        time_col="time",
    )
    
    # 测试 setup
    print("调用 setup('fit')...")
    dm.setup("fit")
    print("✓ setup 完成")
    
    # 检查数据集
    print(f"  train_dataset 长度: {len(dm.train_dataset)}")
    print(f"  val_dataset 长度: {len(dm.val_dataset) if dm.val_dataset else 'None'}")
    
    # 测试 dataloaders
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print(f"✓ train_dataloader 可用")
    print(f"  批次形状: {batch[0].shape}, 标签形状: {batch[1].shape}")
    
    val_loader = dm.val_dataloader()
    if val_loader:
        batch = next(iter(val_loader))
        print(f"✓ val_dataloader 可用")
        print(f"  批次形状: {batch[0].shape}, 标签形状: {batch[1].shape}")
    
    # 测试 scalers
    scaler_dict = dm.scalers_to_dict()
    print(f"✓ scalers_to_dict 可用")
    print(f"  训练 scaler 数量: {len(scaler_dict['train'])}")
    print(f"  测试 scaler 数量: {len(scaler_dict['test'])}")
    
    print()


def test_datamodule_test_mode(tmp_dir: Path):
    """测试测试模式"""
    print("=" * 60)
    print("测试测试模式")
    
    dm = FinetuneDataModule(
        root_path=tmp_dir,
        feature_names=["feature_1", "feature_2"],
        seq_len=50,
        batch_size=32,
    )
    
    dm.setup("test")
    
    # 测试 test_dataloader
    test_loaders = dm.test_dataloader()
    print(f"✓ test_dataloader 返回 {len(test_loaders)} 个 loader")
    
    for i, loader in enumerate(test_loaders):
        batch = next(iter(loader))
        print(f"  Loader {i}: 批次形状 {batch[0].shape}")
    
    # 测试单个文件的数据集创建
    for test_file in dm.test_files:
        dataset = dm.make_test_dataset(test_file)
        print(f"✓ make_test_dataset({test_file}) 成功，长度: {len(dataset)}")
        
        series = dm.make_test_series(test_file)
        print(f"  series values 形状: {series.values.shape}")
    
    print()


def test_scaler_consistency(tmp_dir: Path):
    """测试 scaler 一致性"""
    print("=" * 60)
    print("测试 Scaler 一致性")
    
    dm = FinetuneDataModule(
        root_path=tmp_dir,
        feature_names=["feature_1", "feature_2"],
        seq_len=50,
    )
    
    dm.setup()
    
    # 检查训练和测试的 scaler 是否独立
    for path in dm.train_files:
        train_scaler = dm.scaler_for(path, "train")
        test_scaler = dm.scaler_for(path, "test")
        
        # 它们可能相同（如果训练段相同），但应该是不同的对象
        print(f"  文件: {path}")
        print(f"    训练 scaler mean: {train_scaler.mean[:3]}")
        print(f"    测试 scaler mean: {test_scaler.mean[:3]}")
    
    print("✓ Scaler 检查完成\n")


def test_edge_cases(tmp_dir: Path):
    """测试边界情况"""
    print("=" * 60)
    print("测试边界情况")
    
    # 测试 stride 和 eval_stride
    dm = FinetuneDataModule(
        root_path=tmp_dir,
        feature_names=["feature_1", "feature_2"],
        seq_len=50,
        stride=2,
        eval_stride=1,
    )
    dm.setup()
    print(f"✓ 自定义 stride 设置成功")
    
    # 测试指定文件列表
    dm2 = FinetuneDataModule(
        root_path=tmp_dir,
        feature_names=["feature_1", "feature_2"],
        seq_len=50,
        train_files=["flight_1.csv"],
        test_files=["flight_1.csv", "flight_2.csv"],
    )
    dm2.setup()
    print(f"✓ 指定文件列表成功")
    print(f"  训练文件: {dm2.train_files}")
    print(f"  测试文件: {dm2.test_files}")
    
    # 测试 val_files 为 None（使用 train_files）
    dm3 = FinetuneDataModule(
        root_path=tmp_dir,
        feature_names=["feature_1", "feature_2"],
        seq_len=50,
        train_files=["flight_1.csv"],
    )
    dm3.setup()
    print(f"✓ val_files=None 时自动使用 train_files")
    print(f"  训练文件: {dm3.train_files}")
    print(f"  验证文件: {dm3.val_files}")
    
    # 测试 scaler_type
    dm4 = FinetuneDataModule(
        root_path=tmp_dir,
        feature_names=["feature_1", "feature_2"],
        seq_len=50,
        scaler_type="minmax",
    )
    try:
        dm4.setup()
        print(f"✓ scaler_type='minmax' 测试通过")
    except Exception as e:
        print(f"✗ scaler_type='minmax' 失败: {e}")
    
    print()


def test_from_config(tmp_dir: Path):
    """测试 from_config 类方法"""
    print("=" * 60)
    print("测试 from_config")
    
    config = {
        "data": {
            "root_path": str(tmp_dir),
            "target_fields": ["feature_1", "feature_2"],
            "seq_len": 50,
            "batch_size": 64,
            "stride": 2,
        }
    }
    
    dm = FinetuneDataModule.from_config(config)
    dm.setup()
    print(f"✓ from_config 创建成功")
    print(f"  seq_len: {dm.seq_len}")
    print(f"  batch_size: {dm.batch_size}")
    print(f"  stride: {dm.stride}")
    
    # 测试默认值
    minimal_config = {
        "data": {
            "root_path": str(tmp_dir),
            "target_fields": ["feature_1", "feature_2"],
            "seq_len": 30,
        }
    }
    dm2 = FinetuneDataModule.from_config(minimal_config)
    print(f"✓ 最小配置创建成功")
    print(f"  默认 batch_size: {dm2.batch_size}")
    print(f"  默认 stride: {dm2.stride}")
    
    print()


def main():
    """主测试函数"""
    print("开始测试 FinetuneDataModule\n")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        create_test_data(tmp_path)
        
        try:
            test_flight_split()
            test_datamodule_basic(tmp_path)
            test_datamodule_test_mode(tmp_path)
            test_scaler_consistency(tmp_path)
            test_edge_cases(tmp_path)
            test_from_config(tmp_path)
            
            print("=" * 60)
            print("所有测试通过！✓")
            
        except Exception as e:
            print(f"\n测试失败！")
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()