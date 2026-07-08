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

    # ================================================================
    #  2. 区域热度分析
    # ================================================================
    def plot_zone_popularity(self) -> str:
        """
        上下客量 TOP10 区域柱状图
        输出: m2_2_zone_popularity.png
        """
        df = self.df

        pickup_counts = df["PULocationID"].value_counts().head(10)
        dropoff_counts = df["DOLocationID"].value_counts().head(10)

        pickup_names = self._zone_name(pickup_counts.index)
        dropoff_names = self._zone_name(dropoff_counts.index)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        colors_pickup = sns.color_palette("Blues_d", len(pickup_counts))[::-1]
        ax1 = axes[0]
        bars1 = ax1.barh(range(len(pickup_counts)), pickup_counts.values, color=colors_pickup, edgecolor="white")
        ax1.set_yticks(range(len(pickup_counts)))
        ax1.set_yticklabels(pickup_names.values, fontsize=9)
        ax1.invert_yaxis()
        ax1.set_xlabel("订单量", fontsize=12)
        ax1.set_title("上车量 TOP 10 区域", fontsize=14, fontweight="bold")
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        for bar, val in zip(bars1, pickup_counts.values):
            ax1.text(bar.get_width() + max(pickup_counts.values) * 0.005,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:,}", va="center", fontsize=8, color="black")

        colors_dropoff = sns.color_palette("Oranges_d", len(dropoff_counts))[::-1]
        ax2 = axes[1]
        bars2 = ax2.barh(range(len(dropoff_counts)), dropoff_counts.values, color=colors_dropoff, edgecolor="white")
        ax2.set_yticks(range(len(dropoff_counts)))
        ax2.set_yticklabels(dropoff_names.values, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_xlabel("订单量", fontsize=12)
        ax2.set_title("下车量 TOP 10 区域", fontsize=14, fontweight="bold")
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        for bar, val in zip(bars2, dropoff_counts.values):
            ax2.text(bar.get_width() + max(dropoff_counts.values) * 0.005,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:,}", va="center", fontsize=8, color="black")

        fig.suptitle("NYC 黄牌出租车 区域热度分析", fontsize=16, fontweight="bold", y=1.01)
        fig.tight_layout()
        path = str(self.output_dir / "m2_2_zone_popularity.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  已保存: {path}")
        return path

    def plot_zone_hourly_heatmap(self, top_n: int = 8) -> str:
        """
        热门上车区域 × 小时 订单量热力图
        输出: m2_2_zone_heatmap.png
        """
        df = self.df

        top_zones = df["PULocationID"].value_counts().head(top_n).index.tolist()
        zone_names = self._zone_name(pd.Series(top_zones))

        heat_data = df[df["PULocationID"].isin(top_zones)]
        pivot = heat_data.pivot_table(
            index="PULocationID", columns="pickup_hour",
            values="trip_distance", aggfunc="count"
        )
        pivot = pivot.reindex(top_zones)
        pivot.index = zone_names.values

        fig, ax = plt.subplots(figsize=(16, 6))
        sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".0f",
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"label": "订单量", "shrink": 0.85},
                    xticklabels=[f"{h}:00" for h in range(24)],
                    ax=ax)
        ax.set_title(f"热门上车区域 TOP{top_n} 分小时订单量热力图", fontsize=14, fontweight="bold")
        ax.set_xlabel("小时", fontsize=12)
        ax.set_ylabel("区域", fontsize=12)
        ax.tick_params(axis="both", labelsize=8)

        fig.tight_layout()
        path = str(self.output_dir / "m2_2_zone_heatmap.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  已保存: {path}")
        return path

    # ================================================================
    #  3. 地理空间可视化
    # ================================================================
    def plot_geospatial_map(self) -> str:
        """
        利用 geopandas 加载 taxi_zones.shp 绘制区域分级设色地图
        按上车量分为5级着色
        输出: m2_4_geospatial_map.png
        """
        shp_path = self.project_root / "data" / "taxi_zones.shp"
        gdf = gpd.read_file(shp_path)

        pickup_counts = self.df["PULocationID"].value_counts().reset_index()
        pickup_counts.columns = ["LocationID", "pickup_count"]
        gdf_merged = gdf.merge(pickup_counts, left_on="LocationID",
                               right_on="LocationID", how="left")
        gdf_merged["pickup_count"] = gdf_merged["pickup_count"].fillna(0)

        gdf_merged["pickup_level"] = pd.cut(
            gdf_merged["pickup_count"],
            bins=5,
            labels=["极低", "较低", "中等", "较高", "极高"]
        )

        fig, ax = plt.subplots(figsize=(14, 14))
        gdf_merged.plot(
            column="pickup_count", ax=ax, legend=True,
            cmap="Blues", edgecolor="white", linewidth=0.3,
            legend_kwds={
                "label": "上车订单量",
                "orientation": "horizontal",
                "shrink": 0.65,
                "pad": 0.02
            },
            missing_kwds={"color": "#eeeeee", "label": "无数据"}
        )
        ax.set_title("NYC 黄牌出租车 2026年1月 各区域上车订单量分布",
                     fontsize=16, fontweight="bold", pad=10)
        ax.axis("off")

        top5 = pickup_counts.head(5)
        for _, row in top5.iterrows():
            match = gdf[gdf["LocationID"] == row["LocationID"]]
            if len(match) > 0:
                centroid = match.geometry.centroid.iloc[0]
                zone_name = self.zone_lookup[
                    self.zone_lookup["LocationID"] == row["LocationID"]
                ]["Zone"].values[0] if self.zone_lookup is not None else str(row["LocationID"])
                ax.annotate(zone_name, (centroid.x, centroid.y),
                           fontsize=7, ha="center", color="red", fontweight="bold",
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                    edgecolor="red", alpha=0.7))

        fig.tight_layout()
        path = str(self.output_dir / "m2_4_geospatial_map.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  已保存: {path}")
        return path


    # ================================================================
    #  4. 车费影响因素分析
    # ================================================================
    def plot_fare_factors(self) -> str:
        """
        车费影响因素: 距离-车费散点图 + 时段车费箱线图 + 乘客数-车费箱线图
        输出: m2_3_fare_factors.png
        """
        df = self.df

        sample = df.sample(n=min(20000, len(df)), random_state=42)

        fig = plt.figure(figsize=(16, 12))

        ax1 = fig.add_subplot(2, 2, (1, 2))
        sc = ax1.scatter(sample["trip_distance"], sample["fare_amount"],
                         c=sample["trip_duration_minutes"], cmap="viridis",
                         alpha=0.4, s=3, vmin=0, vmax=60)
        cbar = fig.colorbar(sc, ax=ax1, shrink=0.7)
        cbar.set_label("行程时长(分钟)", fontsize=10)
        ax1.set_xlabel("行程距离(英里)", fontsize=12)
        ax1.set_ylabel("车费($)", fontsize=12)
        ax1.set_title("行程距离 vs 车费 散点图 (N=20,000 随机样本)", fontsize=14, fontweight="bold")
        ax1.set_xlim(0, sample["trip_distance"].quantile(0.99))
        ax1.set_ylim(0, sample["fare_amount"].quantile(0.99))
        ax1.grid(True, alpha=0.2)
        z = np.polyfit(sample["trip_distance"].dropna(), sample["fare_amount"].dropna(), 1)
        p = np.poly1d(z)
        x_range = np.linspace(0, sample["trip_distance"].max(), 100)
        ax1.plot(x_range, p(x_range), "r--", linewidth=1.5, alpha=0.8, label="趋势线")
        ax1.legend(fontsize=10)

        ax2 = fig.add_subplot(2, 2, 3)
        df_period = df.copy()
        conditions = [
            df_period["is_peak_hour"],
            df_period["is_weekend"],
        ]
        default = "非高峰工作日"
        df_period["time_period"] = np.select(
            [conditions[0], conditions[1] & ~conditions[0]],
            ["高峰时段", "周末"],
            default
        )

        order = ["高峰时段", "非高峰工作日", "周末"]
        colors_period = {"高峰时段": "#E53935", "非高峰工作日": "#2196F3", "周末": "#4CAF50"}
        bp2 = df_period.boxplot(
            column="fare_amount", by="time_period", ax=ax2,
            patch_artist=True, showfliers=False, widths=0.55
        )
        for patch, label in zip(
            [item for item in bp2.artists],
            [item.get_text() for item in ax2.get_xticklabels()]
        ):
            if label in colors_period:
                patch.set_facecolor(colors_period[label])
                patch.set_alpha(0.7)
        medians = df_period.groupby("time_period")["fare_amount"].median()
        for i, period in enumerate(medians.index):
            ax2.text(i + 1, medians[period] + 1, f"${medians[period]:.1f}",
                     ha="center", fontsize=9, fontweight="bold", color="black")
        ax2.set_title("不同时段 车费分布", fontsize=14, fontweight="bold")
        ax2.set_xlabel("时段", fontsize=12)
        ax2.set_ylabel("车费($)", fontsize=12)
        ax2.set_xticklabels(medians.index, fontsize=10)
        fig.delaxes(fig.axes[-1]) if len(fig.axes) > 3 else None
        ax2.grid(True, alpha=0.3, axis="y")

        ax3 = fig.add_subplot(2, 2, 4)
        df_pax = df[df["passenger_count"].between(1, 5)]
        pax_data = [df_pax[df_pax["passenger_count"] == i]["fare_amount"].dropna()
                    for i in range(1, 6)]
        bp3 = ax3.boxplot(pax_data, patch_artist=True, showfliers=False, widths=0.55)
        colors_pax = sns.color_palette("Set2", 5)
        for patch, color in zip(bp3["boxes"], colors_pax):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        pax_medians = df_pax.groupby("passenger_count")["fare_amount"].median()
        for i, val in enumerate(pax_medians.values):
            ax3.text(i + 1, val + 1, f"${val:.1f}",
                     ha="center", fontsize=9, fontweight="bold", color="black")
        ax3.set_title("乘客人数 vs 车费分布", fontsize=14, fontweight="bold")
        ax3.set_xlabel("乘客人数", fontsize=12)
        ax3.set_ylabel("车费($)", fontsize=12)
        ax3.set_xticklabels([f"{i}人" for i in range(1, 6)], fontsize=10)
        ax3.grid(True, alpha=0.3, axis="y")

        fig.suptitle("NYC 黄牌出租车 车费影响因素分析", fontsize=16, fontweight="bold", y=1.01)
        fig.tight_layout()
        path = str(self.output_dir / "m2_3_fare_factors.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  已保存: {path}")
        return path


    # ================================================================
    #  主流程
    # ================================================================
    def m2_run(self) -> None:
        """M2 主流程：依次生成所有可视化图表"""
        print("\n" + "=" * 50)
        print("  M2 数据可视化")
        print("=" * 50)

        print("\n[1/4] 出行需求时间规律...")
        self.plot_demand_time()

        print("\n[2/4] 区域热度分析...")
        self.plot_zone_popularity()
        self.plot_zone_hourly_heatmap()

        print("\n[3/4] 车费影响因素分析...")
        self.plot_fare_factors()

        print("\n[4/4] 地理空间可视化...")
        self.plot_geospatial_map()

        print("\nM2 可视化全部完成！")


