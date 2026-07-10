# NYC 出租车出行数据分析与智能问答系统

## 项目简介

《人工智能编程语言》期末大作业。

基于纽约市黄色出租车公开数据集（2026年1月），完整实现四个功能模块：

- **M1** 数据处理：加载数据、生成质量报告、数据清洗、特征工程
- **M2** 分析可视化：出行需求时间规律、区域热度分析、车费影响因素、费率类型深度分析
- **M3** 预测模型：PyTorch 神经网络 + 随机森林对比实验，预测区域时段出行需求量
- **M4** 智能问答系统：规则匹配意图识别 + Gradio 交互界面，可选接入大模型 API 兜底

## 项目结构

```
final_project/
├── data/                      # 原始数据（需要自行下载）
│   ├── yellow_tripdata_2026-01.parquet
│   ├── taxi_zone_lookup.csv
│   └── taxi_zones.*            # 地理空间数据（.shp等）
├── outputs/                    # 输出文件
│   ├── data_quality_report.csv
│   ├── m2_1_demand_time.png
│   ├── m2_2_zone_popularity.png
│   ├── m2_2_zone_heatmap.png
│   ├── m2_3_fare_factors.png
│   ├── m2_4_geospatial_map.png
│   ├── m2_4_ratecode_analysis.png
│   ├── m3_model_metrics.csv
│   └── m3_neural_network_loss.png
├── src/                        # 功能模块代码
│   ├── m1_data_processing.py   # M1: 数据处理
│   ├── m2_visualization.py     # M2: 可视化
│   ├── m3_modeling.py          # M3: 预测模型
│   └── m4_qa_system.py         # M4: 问答系统
├── main.py                     # 主入口
├── requirements.txt            # Python 依赖
├── .gitignore                  # Git 忽略配置
└── README.md                   # 本文件
```

## 环境要求

- Python 3.9+
- PyTorch 2.0+（用于神经网络训练）

## 安装依赖

```bash
pip install -r requirements.txt
```

依赖列表：
```
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0          # parquet 数据读取
matplotlib>=3.7.0
seaborn>=0.12.0
geopandas>=0.14.0        # 地理空间可视化（可选）
scikit-learn>=1.3.0
torch>=2.0.0
gradio>=4.0.0            # M4 可视化界面
openai>=1.0.0            # 大模型 API 客户端
python-dotenv>=1.0.0     # 环境变量管理
```

## 数据准备

1. 下载数据：
   - 主数据：从 [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) 下载 `yellow_tripdata_2026-01.parquet`
   - 区域数据：`taxi_zone_lookup.csv` 和 `taxi_zones.*` 已经在 `data/` 目录

2. 将下载的 parquet 文件放入 `data/` 目录：
   ```
   data/yellow_tripdata_2026-01.parquet
   ```

## 运行方法

### 一键运行全流程

```bash
python main.py
```

执行顺序：
1. **M1** 数据加载 → 生成质量报告 → 数据清洗 → 特征工程
2. **M2** 生成所有分析图表
3. **M3** 训练神经网络和随机森林 → 保存指标和 Loss 曲线
4. **M4** 启动 Gradio 问答界面（自动在浏览器打开）

### 单独运行某个模块

```python
# M1
from src.m1_data_processing import DataQualityAnalyzer
analyzer = DataQualityAnalyzer(
    data_path="data/yellow_tripdata_2026-01.parquet",
    output_path="outputs/data_quality_report.csv"
)
analyzer.m1_run()

# M2
from src.m2_visualization import TaxiVisualizer
viz = TaxiVisualizer(df_cleaned)
viz.m2_run()

# M3
from src.m3_modeling import DemandPredictor
predictor = DemandPredictor(df_cleaned)
predictor.m3_run()

# M4
from src.m4_qa_system import QASystem
qa = QASystem(df_cleaned)
qa.launch_ui()
```

## M4 问答系统说明

### 支持的问题类型

| 问题类型 | 示例问题 |
|----------|----------|
| 时段需求查询 | "早上8点有多少订单？"，"周末订单多还是工作日多？" |
| 区域热度排名 | "最热门的10个上车区域？"，"top5下车区域" |
| 需求预测 | "预测区域100在15点有多少订单？" |
| 费用估算分析 | "平均车费是多少？"，"10英里预估车费多少钱？" |
| 费率类型分析 | "JFK机场费率平均车费是多少？"，"不同费率对比" |
| 数据总览 | "总共有多少条记录？"，"数据质量报告" |

### 大模型 API 配置（可选）

M4 支持接入 GLM / DeepSeek / Qwen 等大模型，当问题无法规则匹配时由大模型生成解释性回复。

**推荐方式：在 Gradio 网页界面中直接填写**

启动 `python main.py` 后，在网页右侧的 **LLM 设置** 面板中直接填写：

1. 选择平台（DeepSeek / 通义千问 / 智谱GLM / 自定义），自动填充 Base URL 和模型名
2. 填入 API Key
3. 点击 **保存配置**

配置后立即生效，无需重启。

**备选方式：通过 `.env` 文件**

在项目根目录创建 `.env` 文件：

```env
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

说明：
- 使用 `openai>=1.0.0` 客户端，兼容所有 OpenAI 格式的 API
- 不配置时系统自动降级为纯规则问答模式，完全可用
- `.env` 已在 `.gitignore` 中，API Key 不会被提交到 Git

## 输出说明

| 输出文件 | 说明 |
|----------|------|
| `outputs/data_quality_report.csv` | M1 数据质量报告，包含各列缺失率、异常值统计 |
| `outputs/m2_1_demand_time.png` | 出行需求时间规律：工作日/周末分小时对比 + 每日趋势 |
| `outputs/m2_2_zone_popularity.png` | 区域热度：上下客 TOP 10 柱状图 |
| `outputs/m2_2_zone_heatmap.png` | 区域热度：热门区域小时订单量热力图 |
| `outputs/m2_3_fare_factors.png` | 车费影响因素：距离-车费散点图 + 时段/乘客数箱线图 |
| `outputs/m2_4_geospatial_map.png` | 地理空间可视化：区域分级设色地图（加分项） |
| `outputs/m2_4_ratecode_analysis.png` | 费率类型深度分析（自选分析） |
| `outputs/m3_neural_network_loss.png` | 神经网络训练/验证 Loss 曲线 |
| `outputs/m3_model_metrics.csv` | 神经网络和随机森林 MAE/RMSE 指标对比 |




