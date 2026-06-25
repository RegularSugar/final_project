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

    def m1_run(self) -> None:
        """
        M1 模块主流程：加载数据 → 生成报告 → 保存输出
        """
        print(f"正在加载数据: {self.data_path}")
        self.load_data()

        print("正在生成数据质量报告...")
        report = self.generate_report()

        self.save_report(report)


if __name__ == "__main__":
    analyzer = DataQualityAnalyzer(
        data_path="data/yellow_tripdata_2026-01.parquet",
        output_path="outputs/data_quality_report.csv"
    )
    analyzer.m1_run()
