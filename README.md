# NYC 出租车出行数据分析与智能问答系统

> 《人工智能编程语言》期末大作业 · 基于 2026 年 1 月黄牌出租车行程数据（约 372 万条记录）

---

## 项目简介

本项目围绕纽约市出租车公开数据集，构建了一个完整的 **数据处理 → 可视化分析 → 预测建模 → 智能问答** 流水线。

| 模块 | 功能 | 核心产出 |
|:----:|------|----------|
| **M1** | 数据处理 | 质量报告 CSV · 11 步清洗流水线 · 3 个衍生特征 |
| **M2** | 可视化分析 | 6 张图表：时间规律 · 区域热度 · 车费因素 · 地理空间 · 费率分析 |
| **M3** | 预测建模 | PyTorch 神经网络 + 随机森林对比 · MAE/RMSE 指标 |
| **M4** | 智能问答 | Gradio 交互界面 · 6 种意图识别 · 大模型 API 兜底 |

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据 → 放入 data/ 目录（详见下方「数据准备」）

# 3. 一键运行
python main.py
```

执行流程：`M1 数据清洗` → `M2 图表生成` → `M3 模型训练` → `M4 问答界面（自动打开浏览器）`

---

## 项目结构

```
final_project/
│
├── 📁 data/                        ← 原始数据（需自行下载 parquet）
│   ├── yellow_tripdata_2026-01.parquet
│   ├── taxi_zone_lookup.csv
│   └── taxi_zones.*                ← 地理空间数据（.shp / .dbf / .prj 等）
│
├── 📁 outputs/                     ← 所有输出文件（自动生成）
│   ├── data_quality_report.csv     ← M1 质量报告
│   ├── m2_1_demand_time.png        ← 出行需求时间规律
│   ├── m2_2_zone_popularity.png    ← 区域热度 TOP10
│   ├── m2_2_zone_heatmap.png       ← 热门区域热力图
│   ├── m2_3_fare_factors.png       ← 车费影响因素
│   ├── m2_4_geospatial_map.png     ← 地理空间分级设色地图
│   ├── m2_4_ratecode_analysis.png  ← 费率类型深度分析
│   ├── m3_neural_network_loss.png  ← 神经网络 Loss 曲线
│   └── m3_model_metrics.csv        ← 模型指标对比
│
├── 📁 src/                         ← 功能模块
│   ├── m1_data_processing.py       ← 数据加载 · 清洗 · 特征工程
│   ├── m2_visualization.py         ← 6 张分析图表
│   ├── m3_modeling.py              ← 神经网络 + 随机森林
│   └── m4_qa_system.py             ← 问答系统 + Gradio 界面
│
├── 📄 main.py                      ← 主入口（一键运行全流程）
├── 📄 requirements.txt             ← Python 依赖清单
├── 📄 .gitignore                   ← Git 忽略规则
└── 📄 README.md                    ← 本文件
```

---

## 环境要求

| 依赖 | 最低版本 | 用途 |
|------|:-------:|------|
| Python | 3.9 | — |


## 安装依赖

```bash
pip install -r requirements.txt
```

```
pandas>=2.0.0           # 数据处理
numpy>=1.24.0           # 数值计算
pyarrow>=14.0.0         # parquet 文件读取
matplotlib>=3.7.0       # 绑图
seaborn>=0.12.0         # 统计可视化
geopandas>=0.14.0       # 地理空间可视化
scikit-learn>=1.3.0     # 随机森林 + 指标计算
torch>=2.0.0            # 神经网络
gradio>=4.0.0           # M4 可视化界面
openai>=1.0.0           # 大模型 API 客户端
python-dotenv>=1.0.0    # 环境变量管理
```

---

## 数据准备

> **主数据文件**（约 300 MB）需自行下载，区域数据已在 `data/` 目录中。

1. 访问 [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
2. 下载 **Yellow Taxi Trip Records** → `2026-01` → `yellow_tripdata_2026-01.parquet`
3. 放入 `data/` 目录：

```
data/
├── yellow_tripdata_2026-01.parquet    ← 下载后放这里
├── taxi_zone_lookup.csv               ← 已提供
└── taxi_zones.*                       ← 已提供
```

---

## 运行方法

### 一键运行

```bash
python main.py
```

```
═══════════════════════════════════════════════════════
  [M1] 数据加载与清洗       →  outputs/data_quality_report.csv
  [M2] 数据可视化           →  outputs/m2_*.png（6 张）
  [M3] 出行需求预测建模     →  outputs/m3_*（Loss 曲线 + 指标）
  [M4] 启动智能问答系统     →  🌐 自动打开浏览器
═══════════════════════════════════════════════════════
```

### 单独运行某个模块

```python
# —— M1 数据处理 ——
from src.m1_data_processing import DataQualityAnalyzer
analyzer = DataQualityAnalyzer(
    data_path="data/yellow_tripdata_2026-01.parquet",
    output_path="outputs/data_quality_report.csv"
)
analyzer.m1_run()

# —— M2 可视化 ——
from src.m2_visualization import TaxiVisualizer
viz = TaxiVisualizer(df_cleaned)
viz.m2_run()

# —— M3 预测建模 ——
from src.m3_modeling import DemandPredictor
predictor = DemandPredictor(df_cleaned)
predictor.m3_run()

# —— M4 问答系统 ——
from src.m4_qa_system import QASystem
qa = QASystem(df_cleaned)
qa.launch_ui()
```

---

## M4 智能问答系统

### 支持的问题类型

| 类型 | 示例 |
|------|------|
| 时段需求查询 | "早上 8 点有多少订单？" · "周末和工作日哪个多？" |
| 区域热度排名 | "最热门的 10 个上车区域？" · "前 5 下车区域" |
| 需求预测 | "预测区域 100 在 15 点有多少订单？" |
| 费用估算 | "平均车费是多少？" · "10 英里预估多少钱？" |
| 费率分析 | "JFK 机场费率平均车费？" · "不同费率对比" |
| 数据总览 | "总共有多少条记录？" · "数据质量报告" |

### 大模型 API 配置（可选）

> 接入后，无法规则匹配的问题将由大模型生成解释性回复。**不配置也能正常使用**。

**推荐方式：网页界面直接填写**

启动后在右侧面板操作：

1. 选择平台 → 自动填充 Base URL 和模型名
2. 填入 API Key
3. 点击 **保存配置** → 立即生效

| 平台 | 模型示例 |
|------|----------|
| DeepSeek | `deepseek-chat` |
| 通义千问 (Qwen) | `qwen-plus` |
| 智谱 GLM | `glm-4-flash` |
| 自定义 | 任意兼容 OpenAI 格式的 API |

**备选方式：`.env` 文件**

```env
LLM_API_KEY=sk-your-key-here
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

> `.env` 已在 `.gitignore` 中，不会被提交到 Git。

---

## 输出文件一览

| 文件 | 模块 | 内容 |
|------|:---:|------|
| `data_quality_report.csv` | M1 | 各列缺失率 · 异常值统计 |
| `m2_1_demand_time.png` | M2 | 工作日/周末分小时订单量对比 · 每日趋势 |
| `m2_2_zone_popularity.png` | M2 | 上下客 TOP 10 区域柱状图 |
| `m2_2_zone_heatmap.png` | M2 | 热门区域 × 小时 订单量热力图 |
| `m2_3_fare_factors.png` | M2 | 距离-车费散点图 · 时段/乘客数箱线图 |
| `m2_4_geospatial_map.png` | M2 | 区域分级设色地图（加分项） |
| `m2_4_ratecode_analysis.png` | M2 | 费率类型深度分析（自选分析） |
| `m3_neural_network_loss.png` | M3 | 神经网络训练/验证 Loss 曲线 |
| `m3_model_metrics.csv` | M3 | 神经网络 vs 随机森林 MAE/RMSE 对比 |

---

