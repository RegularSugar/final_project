"""
m2_visualization.py
====================
出租车行程数据可视化模块

功能：
    1. 出行需求时间规律 —— 工作日/周末分小时订单量对比、每日订单量趋势
    2. 区域热度分析 —— 上下客 TOP10 柱状图、热门区域小时热力图
    3. 车费影响因素 —— 距离-车费散点图、时段/人数与车费关系箱线图
    4. 地理空间可视化 —— 利用 taxi_zones.shp 绘制区域分级设色地图

输出：
    outputs/m2_1_demand_time.png
    outputs/m2_2_zone_popularity.png
    outputs/m2_2_zone_heatmap.png
    outputs/m2_3_fare_factors.png
    outputs/m2_4_geospatial_map.png

使用方式：
    from src.m2_visualization import TaxiVisualizer
    viz = TaxiVisualizer(df_cleaned)
    viz.m2_run()
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class TaxiVisualizer:
    """
    出租车数据可视化器

    基于 M1 清洗后的 DataFrame 生成各类分析图表。

    Attributes
    ----------
    df : pd.DataFrame
        M1 清洗并添加特征后的数据
    output_dir : Path
        图表输出目录
    zone_lookup : pd.DataFrame or None
        区域 ID → 区域名称查找表
    """

    def __init__(self, df: pd.DataFrame):
        self.project_root = Path(__file__).resolve().parent.parent
        self.df = df.copy()
        self.output_dir = self.project_root / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.zone_lookup = self._load_zone_lookup()

    def _load_zone_lookup(self) -> pd.DataFrame:
        path = self.project_root / "data" / "taxi_zone_lookup.csv"
        if path.exists():
            return pd.read_csv(path)
        return None

    def _zone_name(self, loc_ids: pd.Series) -> pd.Series:
        if self.zone_lookup is None:
            return pd.Series(loc_ids).astype(str)
        mapping = dict(zip(self.zone_lookup["LocationID"], self.zone_lookup["Zone"]))
        s = pd.Series(loc_ids)
        result = s.map(mapping)
        return result.fillna(s.astype(str))

    # ================================================================
    #  1. 出行需求时间规律
    # ================================================================
    def plot_demand_time(self) -> str:
        """
        工作日/周末分小时订单量对比折线图 + 每日订单量趋势
        输出: m2_1_demand_time.png
        """
        df = self.df

        hourly = df.groupby(["pickup_hour", "is_weekend"]).size().unstack(fill_value=0)
        weekday_line = hourly[False] / hourly[False].sum() * 100
        weekend_line = hourly[True] / hourly[True].sum() * 100

        daily = df.groupby("pickup_day").size()

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        ax1 = axes[0]
        ax1.plot(weekday_line.index, weekday_line.values, "o-",
                 color="#2196F3", linewidth=2, markersize=5, label="工作日")
        ax1.plot(weekend_line.index, weekend_line.values, "s--",
                 color="#F44336", linewidth=2, markersize=5, label="周末")
        ax1.set_xlabel("小时", fontsize=12)
        ax1.set_ylabel("订单占比(%)", fontsize=12)
        ax1.set_title("工作日 vs 周末 分小时订单量占比对比", fontsize=14, fontweight="bold")
        ax1.set_xticks(range(0, 24))
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        peak_morning_start, peak_morning_end = 7, 9
        peak_evening_start, peak_evening_end = 16, 19
        ax1.axvspan(peak_morning_start, peak_morning_end, alpha=0.08, color="orange")
        ax1.axvspan(peak_evening_start, peak_evening_end, alpha=0.08, color="orange")
        ax1.annotate("早高峰", xy=(8, ax1.get_ylim()[1] * 0.95), fontsize=9, color="orange",
                     ha="center", fontweight="bold")
        ax1.annotate("晚高峰", xy=(17.5, ax1.get_ylim()[1] * 0.95), fontsize=9, color="orange",
                     ha="center", fontweight="bold")

        ax2 = axes[1]
        days = daily.index.astype(int)
        ax2.fill_between(days, daily.values, alpha=0.35, color="#4CAF50")
        ax2.plot(days, daily.values, "o-", color="#2E7D32", linewidth=1.5, markersize=4)
        ax2.set_xlabel("日期", fontsize=12)
        ax2.set_ylabel("订单量", fontsize=12)
        ax2.set_title(f"2026年1月 每日订单量趋势 (共 {daily.sum():,} 单)", fontsize=14, fontweight="bold")
        ax2.set_xticks(days)
        ax2.set_xticklabels(days, rotation=45, fontsize=8)
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax2.axhline(y=daily.mean(), color="red", linestyle="--", linewidth=1, alpha=0.7,
                    label=f"日均 {daily.mean():,.0f}")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        path = str(self.output_dir / "m2_1_demand_time.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  已保存: {path}")
        return path


