"""
main.py
=======
项目主入口，串联各功能模块
"""

from src.m1_data_processing import DataQualityAnalyzer


def main():
    print("=" * 60)
    print("  出租车行程数据分析项目")
    print("=" * 60)

    analyzer = DataQualityAnalyzer(
        data_path="data/yellow_tripdata_2026-01.parquet",
        output_path="outputs/data_quality_report.csv"
    )
    analyzer.m1_run()

    df_cleaned = analyzer.df



if __name__ == "__main__":
    main()
