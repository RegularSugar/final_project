"""
m1_data_processing.py
=====================
出租车行程数据质量报告生成模块

功能：
    1. 加载 parquet 格式的黄牌出租车行程数据
    2. 生成数据质量报告（缺失率、异常值统计等）
    3. 输出报告至 outputs/data_quality_report.csv

使用方式：
    from src.m1_data_processing import DataQualityAnalyzer
    analyzer = DataQualityAnalyzer(data_path, output_path)
    analyzer.m1_run()
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataQualityAnalyzer:
    """
    数据质量分析器

    负责加载出租车 parquet 数据，计算缺失率与异常值统计，生成质量报告并输出 CSV。

    Attributes
    ----------
    data_path : Path
        输入 parquet 文件路径
    output_path : Path
        输出 CSV 报告路径
    df : pd.DataFrame
        加载后的原始数据
    """

    def __init__(self, data_path: str, output_path: str):
        """
        初始化分析器

        Parameters
        ----------
        data_path : str
            parquet 文件路径
        output_path : str
            报告输出路径
        """
        self.project_root = Path(__file__).resolve().parent.parent
        self.data_path = self.project_root / data_path
        self.output_path = self.project_root / output_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        加载 parquet 格式数据

        Returns
        -------
        pd.DataFrame
            加载后的 DataFrame
        """
        self.df = pd.read_parquet(self.data_path)
        print(f"数据加载完成，共 {len(self.df)} 行，{len(self.df.columns)} 列")
        return self.df

    def compute_missing_stats(self) -> pd.DataFrame:
        """
        计算各列缺失率统计

        Returns
        -------
        pd.DataFrame
            缺失率统计表，包含列名、总数、缺失数、缺失率
        """
        total = len(self.df)
        missing_count = self.df.isnull().sum()
        missing_rate = (missing_count / total * 100).round(2)

        stats = pd.DataFrame({
            "column": self.df.columns,
            "total_rows": total,
            "missing_count": missing_count.values,
            "missing_rate_pct": missing_rate.values
        })
        return stats

    def compute_outliers(self, col: str) -> dict:
        """
        使用 IQR 方法检测数值列异常值

        Parameters
        ----------
        col : str
            数值列名称

        Returns
        -------
        dict
            包含异常值统计信息的字典
        """
        series = self.df[col].dropna()
        if len(series) == 0:
            return {
                "column": col,
                "min": np.nan, "max": np.nan,
                "Q1": np.nan, "Q3": np.nan, "IQR": np.nan,
                "lower_bound": np.nan, "upper_bound": np.nan,
                "outlier_count": 0, "outlier_rate_pct": 0.0
            }

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_count = len(outliers)
        outlier_rate = round(outlier_count / len(series) * 100, 2)

        return {
            "column": col,
            "min": series.min(),
            "max": series.max(),
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": outlier_count,
            "outlier_rate_pct": outlier_rate
        }

    def generate_report(self) -> pd.DataFrame:
        """
        生成完整数据质量报告

        Returns
        -------
        pd.DataFrame
            合并后的数据质量报告
        """
        missing_stats = self.compute_missing_stats()

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_results = [self.compute_outliers(col) for col in numeric_cols]
        outlier_stats = pd.DataFrame(outlier_results)

        if not outlier_stats.empty:
            report = missing_stats.merge(outlier_stats, on="column", how="left")
        else:
            report = missing_stats

        return report

    def save_report(self, report: pd.DataFrame) -> None:
        """
        保存报告至 CSV 文件

        Parameters
        ----------
        report : pd.DataFrame
            待保存的报告
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(self.output_path, index=False, encoding="utf-8-sig")
        print(f"报告已保存至: {self.output_path}")

    def clean_data(self) -> pd.DataFrame:
        """
        数据清洗主流程

        按顺序执行以下步骤，每步输出清洗统计信息：

        过滤步骤（删除行）：
            1. 去重 —— 删除完全重复的行，避免虚增样本量
            2. 时间异常 —— 下车时间 ≤ 上车时间不可能发生，属录入错误
            3. 乘客数异常 —— 0人或超过5人不合法
               (NYC TLC规定：4乘客出租车最多4人，5乘客出租车最多5人)
            4. 距离异常 —— ≤0未行驶，>200英里远超合理范围
            5. 费用异常 —— fare_amount ≤ 0 属退款/调整记录
            6. 费率码异常 —— RatecodeID=99 官方定义为 Null/unknown
            7. 供应商异常 —— VendorID 不在 {1,2,6,7}
               (官方字典仅1=Creative, 2=Curb, 6=Myle, 7=Helix)

        填充步骤（修改值）：
            8. store_and_fwd_flag 缺失 → 填充"N"
               (缺失通常意味"否"而非未知)
            9. congestion_surcharge/Airport_fee 缺失 → 填充0
               (属旧版本数据未包含字段，0最安全)
            10. 负值附加费裁剪 → 裁剪为0
                (附加费/税费/小费不应为负，小额负值属舍入误差)

        重算步骤：
            11. 重算 total_amount = 所有费用字段求和
                (清洗后各字段值可能变动，保证一致性)

        Returns
        -------
        pd.DataFrame
            清洗后的 DataFrame
        """
        print(f"\n开始数据清洗，当前数据量: {len(self.df)} 行")

        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)
        print(f"  步骤1 去重: 删除 {before - after} 行 ({((before - after) / before * 100):.2f}%)，剩余 {after} 行")

        before = len(self.df)
        time_invalid = self.df["tpep_dropoff_datetime"] <= self.df["tpep_pickup_datetime"]
        self.df = self.df[~time_invalid]
        after = len(self.df)
        print(f"  步骤2 时间异常: 删除 {before - after} 行 ({((before - after) / before * 100):.2f}%)，剩余 {after} 行")

        before = len(self.df)
        passenger_invalid = (self.df["passenger_count"] == 0) | (self.df["passenger_count"] > 5)
        self.df = self.df[~passenger_invalid]
        after = len(self.df)
        print(f"  步骤3 乘客数异常: 删除 {before - after} 行 ({((before - after) / before * 100):.2f}%)，剩余 {after} 行")

        before = len(self.df)
        distance_invalid = (self.df["trip_distance"] <= 0) | (self.df["trip_distance"] > 200)
        self.df = self.df[~distance_invalid]
        after = len(self.df)
        print(f"  步骤4 距离异常: 删除 {before - after} 行 ({((before - after) / before * 100):.2f}%)，剩余 {after} 行")

        before = len(self.df)
        fare_invalid = self.df["fare_amount"] <= 0
        self.df = self.df[~fare_invalid]
        after = len(self.df)
        print(f"  步骤5 费用异常: 删除 {before - after} 行 ({((before - after) / before * 100):.2f}%)，剩余 {after} 行")

        before = len(self.df)
        self.df = self.df[self.df["RatecodeID"] != 99]
        after = len(self.df)
        print(f"  步骤6 费率码异常: 删除 {before - after} 行 ({((before - after) / before * 100):.2f}%)，剩余 {after} 行")

        before = len(self.df)
        valid_vendors = {1, 2, 6, 7}
        self.df = self.df[self.df["VendorID"].isin(valid_vendors)]
        after = len(self.df)
        print(f"  步骤7 供应商异常: 删除 {before - after} 行 ({((before - after) / before * 100):.2f}%)，剩余 {after} 行")

        before = self.df["store_and_fwd_flag"].isnull().sum()
        self.df["store_and_fwd_flag"] = self.df["store_and_fwd_flag"].fillna("N")
        print(f"  步骤8 store_and_fwd_flag: 填充 {before} 个缺失值为 'N'")

        missing_cols = ["congestion_surcharge", "Airport_fee"]
        for col in missing_cols:
            before = self.df[col].isnull().sum()
            self.df[col] = self.df[col].fillna(0)
            print(f"  步骤9 {col}: 填充 {before} 个缺失值为 0")

        fee_cols = [
            "extra", "mta_tax", "tip_amount", "tolls_amount",
            "improvement_surcharge", "congestion_surcharge",
            "Airport_fee", "cbd_congestion_fee"
        ]
        for col in fee_cols:
            if col in self.df.columns:
                negative_count = (self.df[col] < 0).sum()
                self.df[col] = self.df[col].clip(lower=0)
                print(f"  步骤10 {col}: 裁剪 {negative_count} 个负值为 0")

        amount_cols = [
            "fare_amount", "extra", "mta_tax", "tip_amount",
            "tolls_amount", "improvement_surcharge", "congestion_surcharge",
            "Airport_fee", "cbd_congestion_fee"
        ]
        self.df["total_amount"] = self.df[amount_cols].sum(axis=1)
        print(f"  步骤11 total_amount: 已重新计算")

        print(f"\n数据清洗完成，最终数据量: {len(self.df)} 行\n")
        return self.df

    def add_time_features(self) -> pd.DataFrame:
        """
        从行程时间中提取基础时间特征与衍生特征

        Returns
        -------
        pd.DataFrame
            添加特征后的 DataFrame
        """
        pickup_dt = pd.to_datetime(self.df["tpep_pickup_datetime"])

        # 上车小时 (0-23)，用于分析各时段出行量
        self.df["pickup_hour"] = pickup_dt.dt.hour
        # 星期几 (0=周一, 6=周日)，用于分析工作日/周末差异
        self.df["pickup_dayofweek"] = pickup_dt.dt.dayofweek
        # 月份 (1-12)，用于分析月度趋势
        self.df["pickup_month"] = pickup_dt.dt.month
        # 日期 (1-31)，用于分析日间波动
        self.df["pickup_day"] = pickup_dt.dt.day

        # 是否周末 (周六或周日)，方便分组统计
        self.df["is_weekend"] = self.df["pickup_dayofweek"].isin([5, 6])

        # 是否高峰时段 (工作日 7:00-9:00 或 16:00-19:00)，用于分析通勤时段特征
        self.df["is_peak_hour"] = (
            ~self.df["is_weekend"]
            & (
                (self.df["pickup_hour"] >= 7) & (self.df["pickup_hour"] <= 9)
                | (self.df["pickup_hour"] >= 16) & (self.df["pickup_hour"] <= 19)
            )
        )

        dropoff_dt = pd.to_datetime(self.df["tpep_dropoff_datetime"])

        # 行程时长(分钟)：一次行程实际耗费的时间，由下车时间减去上车时间得到
        # 作用：结合 trip_distance 可计算速度、分析拥堵；异常值时长可为数据质量提供辅助判断
        self.df["trip_duration_minutes"] = (
            (dropoff_dt - pickup_dt).dt.total_seconds() / 60
        )

        valid_duration = (
            (self.df["trip_duration_minutes"] > 0)
            & (self.df["trip_duration_minutes"] < 180)
        )

        # 平均速度(英里/小时)：该行程的平均行驶速度，由距离 ÷ 时间计算
        # 作用：直接反映路况拥堵程度；极低速度暗示严重拥堵或怠速等待；极高速度(>80mph)可能为数据异常
        self.df["avg_speed_mph"] = np.nan
        self.df.loc[valid_duration, "avg_speed_mph"] = (
            self.df.loc[valid_duration, "trip_distance"]
            / (self.df.loc[valid_duration, "trip_duration_minutes"] / 60)
        )

        # 小费比例：tip_amount ÷ fare_amount，反映乘客支付小费占车费的比重
        # 作用：是乘客满意度与服务质量的代理指标——深夜/机场线小费偏高、短途小费比例偏低；
        #       后续可视化与建模中的高价值分析变量
        self.df["tip_ratio"] = np.nan
        valid_fare = self.df["fare_amount"] > 0
        self.df.loc[valid_fare, "tip_ratio"] = (
            self.df.loc[valid_fare, "tip_amount"]
            / self.df.loc[valid_fare, "fare_amount"]
        )

        print("基础时间特征已添加: pickup_hour, pickup_dayofweek, pickup_month, pickup_day")
        print("基础时间特征已添加: is_weekend, is_peak_hour")
        print("衍生特征已添加: trip_duration_minutes, avg_speed_mph, tip_ratio")

        return self.df

    def m1_run(self) -> None:
        """
        M1 模块主流程：加载数据 → 生成质量报告 → 保存报告 → 数据清洗 → 特征工程
        """
        print(f"正在加载数据: {self.data_path}")
        self.load_data()

        print("正在生成数据质量报告...")
        report = self.generate_report()

        self.save_report(report)

        print("正在执行数据清洗...")
        self.clean_data()

        print("正在提取时间特征...")
        self.add_time_features()




if __name__ == "__main__":
    analyzer = DataQualityAnalyzer(
        data_path="data/yellow_tripdata_2026-01.parquet",
        output_path="outputs/data_quality_report.csv"
    )
    analyzer.m1_run()
