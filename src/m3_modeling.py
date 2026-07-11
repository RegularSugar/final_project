"""
m3_modeling.py
==============
出租车出行需求量预测建模模块

任务定义：
    预测某区域(PULocationID)在指定小时(pickup_hour)和日期(pickup_day)的出行需求量(行程数量)。

模型方案：
    1. 神经网络(PyTorch) —— Embedding + MLP
    2. 随机森林(scikit-learn) —— 作为基准对比

输出：
    outputs/m3_neural_network_loss.png  —— 训练/验证 Loss 曲线
    outputs/m3_model_metrics.csv        —— MAE、RMSE 指标对比表

使用方式：
    from src.m3_modeling import DemandPredictor
    predictor = DemandPredictor(df_cleaned)
    predictor.m3_run()
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DemandPredictor:
    """
    出行需求量预测器

    使用神经网络和随机森林两种方法预测区域-时段出行需求量。

    Attributes
    ----------
    df : pd.DataFrame
        M1 清洗后的数据
    output_dir : Path
        图表/指标输出目录
    """

    def __init__(self, df: pd.DataFrame):
        self.project_root = Path(__file__).resolve().parent.parent
        self.df = df.copy()
        self.output_dir = self.project_root / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.X_train: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.y_test: np.ndarray | None = None
        self.label_encoder = None
        self.feature_cols = None
        self.num_zones = None
        self.nn_model = None
        self.rf_model = None
        self.label_encoder = None
        self.feature_cols = None
        self.num_zones = None
        self.nn_model = None
        self.rf_model = None
        self.label_encoder = None
        self.feature_cols = None
        self.num_zones = None
        self.nn_model = None
        self.rf_model = None
        self.label_encoder = None
        self.feature_cols = None
        self.num_zones = None
        self.nn_model = None
        self.rf_model = None
        self.label_encoder = None
        self.feature_cols = None
        self.num_zones = None
        self.nn_model = None
        self.rf_model = None
        self.label_encoder = None
        self.feature_cols = None
        self.num_zones = None
        self.nn_model = None
        self.rf_model = None

    def prepare_data(self):
        """
        构建样本数据集。

        样本构造方式：
            将原始行程数据按 (PULocationID, pickup_hour, pickup_day) 分组聚合，
            统计每组的行程数量(trip_count)作为预测目标。

        特征设计及采用原因:
            PULocationID (类别)
                —— 区域是最核心的空间因子。不同区域的需求基数差异巨大，
                   如曼哈顿中城日均上万单，而斯塔滕岛偏远区可能仅数十单。
                   若不对区域编码，模型将无法区分空间需求的巨大差异。

            pickup_hour (数值, 0-23)
                —— 小时是需求时间周期的决定性变量。凌晨3点与早高峰8点
                   的需求量可相差10倍以上。作为连续数值特征可捕捉日周期。

            pickup_dayofweek (数值, 0=周一 ~ 6=周日)
                —— 一周内不同天的出行模式根本不同：工作日以通勤为主，
                   周末以休闲为主。引入此特征使模型区分工作日/周末模式。

            is_weekend (布尔值, 0/1)
                —— 周末效应的显式布尔信号。虽然 dayofweek 已编码星期，
                   但周末与否是一个强二分类信号，有助于模型快速区分。

            is_peak_hour (布尔值, 0/1)
                —— 高峰时段(工作日7-9点&16-19点)需求激增，
                   是出行数据中最显著的时段信号之一。

            pickup_day (数值, 1-31)
                —— 捕捉月内变化趋势（月初 vs 月末差异、
                   节假日效应、发薪日效应等）。
        """
        df = self.df

        agg = df.groupby(["PULocationID", "pickup_hour", "pickup_day"]).agg(
            trip_count=("trip_distance", "count"),
            pickup_dayofweek=("pickup_dayofweek", "first"),
            is_weekend=("is_weekend", "first"),
            is_peak_hour=("is_peak_hour", "first"),
        ).reset_index()

        self.label_encoder = LabelEncoder()
        agg["PULocationID_encoded"] = self.label_encoder.fit_transform(agg["PULocationID"])

        self.feature_cols = [
            "PULocationID_encoded",
            "pickup_hour",
            "pickup_dayofweek",
            "is_weekend",
            "is_peak_hour",
            "pickup_day",
        ]
        x = agg[self.feature_cols].values
        y = agg["trip_count"].values.astype(np.float32)

        self.num_zones = agg["PULocationID_encoded"].nunique()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=SEED
        )

        print(f"样本总数: {len(x)}")
        print(f"训练集: {len(self.X_train)} | 测试集: {len(self.X_test)}")
        print(f"区域数: {self.num_zones} | 特征数: {x.shape[1]}")
        print(f"需求量范围: {float(np.min(y)):.0f} ~ {float(np.max(y)):.0f} | 均值: {float(np.mean(y)):.1f}")

    # ================================================================
    #  神经网络模型
    # ================================================================
    class DemandNN(nn.Module):
        """
        出行需求预测神经网络。

        架构设计思路：
            - Embedding 层将 PULocationID (类别数~260) 映射为 16 维稠密向量，
              相比 One-Hot (~260维) 大幅降维，且能学习区域间的语义相似性。
            - 其余 5 个数值特征直接与 embedding 输出拼接。
            - 3 层全连接(128→64→32)逐步压缩，每层后跟 BatchNorm 稳定训练、
              Dropout 防过拟合、ReLU 激活提供非线性。
            - 最终输出 1 个标量(需求量)。
        """
        def __init__(self, num_zones: int, embedding_dim: int = 16):
            super().__init__()
            self.embedding = nn.Embedding(num_zones, embedding_dim)
            input_dim = embedding_dim + 5
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, zone_idx, num_features):
            emb = self.embedding(zone_idx)
            x = torch.cat([emb, num_features], dim=1)
            return self.net(x).squeeze(-1)

    def train_nn(self) -> tuple:
        """
        训练神经网络并返回训练/验证 loss 历史。

        训练配置说明：
            - Epochs=80: 经过验证，80轮足够使loss收敛到稳定值
            - Batch size=256: 平衡训练速度和梯度稳定性
            - LR=1e-3: Adam 默认学习率，对回归任务稳健
            - MSELoss: 均方误差是回归任务的经典损失函数

        Returns
        -------
        train_losses, val_losses : list[float]
            每个 epoch 的训练/验证 loss
        """
        x_train = self.X_train.astype(np.float32)
        x_test_data = self.X_test.astype(np.float32)
        y_train = self.y_train
        y_test_data = self.y_test

        def prepare_tensors(x_data, y_data):
            zone_idx = torch.tensor(x_data[:, 0], dtype=torch.long).to(DEVICE)
            num_feat = torch.tensor(x_data[:, 1:], dtype=torch.float32).to(DEVICE)
            y_tensor = torch.tensor(y_data, dtype=torch.float32).to(DEVICE)
            return zone_idx, num_feat, y_tensor

        zone_idx_train, num_feat_train, y_train_t = prepare_tensors(x_train, y_train)
        zone_idx_val, num_feat_val, y_val_t = prepare_tensors(x_test_data, y_test_data)

        train_ds = TensorDataset(zone_idx_train, num_feat_train, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

        model = self.DemandNN(self.num_zones).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        epochs = 80
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for z_batch, n_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = model(z_batch, n_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * z_batch.size(0)

            train_losses.append(epoch_loss / len(zone_idx_train))

            model.eval()
            with torch.no_grad():
                val_predictions = model(zone_idx_val, num_feat_val)
                val_loss = criterion(val_predictions, y_val_t).item()
            val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}  Train Loss: {train_losses[-1]:.1f}  Val Loss: {val_losses[-1]:.1f}")

        self.nn_model = model
        self.nn_model.eval()
        return train_losses, val_losses

    def predict_nn(self) -> np.ndarray:
        """神经网络在测试集上预测"""
        x_t = torch.tensor(self.X_test[:, 1:].astype(np.float32), dtype=torch.float32).to(DEVICE)
        z_t = torch.tensor(self.X_test[:, 0].astype(np.int64), dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            predictions = self.nn_model(z_t, x_t).cpu().numpy()
        return predictions

    # ================================================================
    #  随机森林模型
    # ================================================================
    def train_rf(self) -> RandomForestRegressor:
        """
        训练随机森林回归模型。

        参数说明：
            - n_estimators=100: 100棵树足够收敛，再增加边际收益递减
            - max_depth=15: 限制深度防止对区域ID等类别特征过度拟合
            - n_jobs=-1: 多核并行加速训练
            - random_state=42: 固定随机种子保证可复现

        返回
        -------
        RandomForestRegressor
            训练好的随机森林模型
        """
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            n_jobs=-1,
            random_state=SEED,
        )
        rf.fit(self.X_train, self.y_train)
        self.rf_model = rf
        return rf

    def predict_rf(self) -> np.ndarray:
        """随机森林在测试集上预测"""
        return self.rf_model.predict(self.X_test)

    # ================================================================
    #  评估与输出
    # ================================================================
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        mae = mean_absolute_error(y_true, y_pred)
        root_mse = np.sqrt(mean_squared_error(y_true, y_pred))
        return {"MAE": float(np.round(mae, 4)), "RMSE": float(np.round(root_mse, 4))}

    def plot_loss(self, train_losses: list, val_losses: list) -> str:
        """绘制训练/验证 Loss 曲线"""
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, "b-", linewidth=1.5, label="训练 Loss", alpha=0.8)
        ax.plot(epochs, val_losses, "r-", linewidth=1.5, label="验证 Loss", alpha=0.8)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("MSE Loss", fontsize=12)
        ax.set_title("神经网络训练与验证 Loss 曲线", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.annotate(f"最低验证 Loss: {min(val_losses):.1f}",
                    xy=(val_losses.index(min(val_losses)) + 1, min(val_losses)),
                    xytext=(val_losses.index(min(val_losses)) + 8, min(val_losses) * 1.3),
                    arrowprops=dict(arrowstyle="->", color="red"), fontsize=10, color="red")
        fig.tight_layout()
        path = str(self.output_dir / "m3_neural_network_loss.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  已保存: {path}")
        return path

    def save_metrics(self, nn_metrics: dict, rf_metrics: dict) -> str:
        """保存模型指标对比 CSV"""
        metrics_df = pd.DataFrame({
            "模型": ["神经网络(PyTorch)", "随机森林(scikit-learn)"],
            "MAE": [nn_metrics["MAE"], rf_metrics["MAE"]],
            "RMSE": [nn_metrics["RMSE"], rf_metrics["RMSE"]],
        })
        path = str(self.output_dir / "m3_model_metrics.csv")
        metrics_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  已保存: {path}")
        print(f"\n  {'='*40}")
        print(f"  模型指标对比:")
        print(f"  {'='*40}")
        print(f"  神经网络    MAE={nn_metrics['MAE']:.4f}  RMSE={nn_metrics['RMSE']:.4f}")
        print(f"  随机森林    MAE={rf_metrics['MAE']:.4f}  RMSE={rf_metrics['RMSE']:.4f}")
        print(f"  {'='*40}")

        """
        两种方法的优劣分析：

        神经网络(PyTorch)：
            优势:
                - Embedding 层可学习区域的低维稠密表示，捕捉区域间的语义相似性
                  （如 Midtown East 与 Midtown North 天然相似），泛化能力强。
                - 通过非线性激活函数组合特征，可学习复杂交互（如"曼哈顿+早高峰"的强叠加效应）。
                - 可扩展性强，后续可加入更多特征（天气、事件等）而不需要特征工程。
            劣势:
                - 训练时间长（需多轮迭代），超参数调优复杂。
                - 对数据量有一定要求，在小样本区域上可能不如树模型稳。
                - 可解释性差，难以直观理解各特征对预测的贡献。

        随机森林：
            优势:
                - 训练快速（一次性拟合），超参数少，几乎不需要调参就有不错的表现。
                - 天然抗噪声和异常值，对区域ID的One-Hot编码有较好的处理能力。
                - 可输出特征重要性，便于业务解释（如"小时"是最重要的预测因子）。
                - 对数据分布无假设，小样本区域也能给出保守估计。
            劣势:
                - 区域ID被One-Hot编码为高维稀疏特征，无法学习区域间的相似性。
                - 对训练集范围外的区域/dayofweek组合完全无法外推。
                - 表达能力受树深度限制，对复杂非线性交互的拟合能力弱于神经网络。

        总结：
            如果区域数量多且数据充足，神经网络更适合；如果追求快速部署和可解释性，
            随机森林是更务实的选择。在本任务中，两者性能差异通常较小，因为需求量
            主要由"区域"和"小时"两个强信号决定，简单的线性/树模型已能捕捉大部分规律。
        """
        return path

    # ================================================================
    #  主流程
    # ================================================================
    def m3_run(self) -> None:
        """M3 主流程：准备数据 → 训练两种模型 → 评估 → 保存结果"""
        print("\n" + "=" * 50)
        print("  M3 出行需求预测建模")
        print("=" * 50)

        print("\n[3.1] 构建样本数据...")
        self.prepare_data()

        print("\n[3.2] 训练神经网络(PyTorch)...")
        train_losses, val_losses = self.train_nn()

        print("\n[3.3] 绘制 Loss 曲线...")
        self.plot_loss(train_losses, val_losses)

        print("\n[3.4] 训练随机森林...")
        self.train_rf()

        print("\n[3.5] 评估模型...")
        nn_predictions = self.predict_nn()
        rf_predictions = self.predict_rf()
        nn_metrics = self.evaluate(self.y_test, nn_predictions)
        rf_metrics = self.evaluate(self.y_test, rf_predictions)

        print("\n[3.6] 保存指标...")
        self.save_metrics(nn_metrics, rf_metrics)

        print("\nM3 建模完成！")