"""
main.py
=======
项目主入口，串联各功能模块
"""

from src.m1_data_processing import DataQualityAnalyzer
from src.m2_visualization import TaxiVisualizer
from src.m3_modeling import DemandPredictor


def main():
    print("=" * 60)
    print("  出租车行程数据分析项目")
    print("=" * 60)

    # ========================
    #  M1: 数据加载与清洗
    # ========================
    analyzer = DataQualityAnalyzer(
        data_path="data/yellow_tripdata_2026-01.parquet",
        output_path="outputs/data_quality_report.csv"
    )
    analyzer.m1_run()

    df_cleaned = analyzer.df

    # ========================
    #  M2: 数据可视化
    # ========================
    viz = TaxiVisualizer(df_cleaned)
    viz.m2_run()

    # ========================
    #  M3: 出行需求预测建模
    # ========================
    predictor = DemandPredictor(df_cleaned)
    predictor.m3_run()



if __name__ == "__main__":
    main()
