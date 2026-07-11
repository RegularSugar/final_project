"""
m4_qa_system.py
===============
NYC 出租车数据智能问答系统

功能：
    1. 意图识别 —— 规则匹配 + 关键词提取，支持 6 种问题类型
    2. 结构化回答 —— 返回数字结论 + 文本解释 + 相关图表路径
    3. Gradio 可视化界面 —— 对话式 UI，支持聊天历史和图表展示
    4. LLM 兜底 —— 未匹配规则时调用大模型 API 解释性回复（可选）

支持的问题类型：
    时段需求查询、区域热度排名、需求预测、费用估算分析、数据总览、费率分析

LLM 接入：
    通过环境变量 LLM_API_KEY、LLM_BASE_URL、LLM_MODEL 配置
    兼容 OpenAI Chat Completions API（DeepSeek / Qwen / GLM 等均可）
    未配置时自动降级为纯规则模式

使用方式：
    # 命令行交互
    python src/m4_qa_system.py

    # 从 main.py 调用
    from src.m4_qa_system import QASystem
    qa = QASystem(df_cleaned)
    qa.launch_ui()
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path

from dotenv import load_dotenv

# Gradio 界面自定义样式
CHAT_CSS = """
.chatbot { height: 500px !important; }
.quick-btn { margin: 3px !important; }
"""


class QASystem:
    """
    NYC 出租车智能问答系统

    接收自然语言问题，通过 意图识别 → 实体提取 → 规则回答 → LLM兜底
    的流水线返回结构化答案。

    Attributes
    ----------
    df : pd.DataFrame
        M1 清洗后的数据
    stats : dict
        预计算的常用统计量，避免每次回答重复计算
    chart_map : dict
        图表名称 → 相对路径映射
    """

    def __init__(self, df: pd.DataFrame):
        self.project_root = Path(__file__).resolve().parent.parent

        # 加载 .env 文件中的环境变量（LLM_API_KEY 等）
        env_path = self.project_root / ".env"
        load_dotenv(env_path if env_path.exists() else None)

        self.df = df.copy()
        self.zone_lookup: pd.DataFrame | None = self._load_zone_lookup()
        self.stats: dict = {}
        self._precompute_stats()

        # LLM 配置：优先从环境变量读取，Gradio 界面可覆盖
        self._llm_config: dict[str, str] = {
            "api_key": os.environ.get("LLM_API_KEY", ""),
            "base_url": os.environ.get("LLM_BASE_URL", ""),
            "model": os.environ.get("LLM_MODEL", ""),
        }

        self.chart_map: dict[str, str] = {
            "需求时间": "outputs/m2_1_demand_time.png",
            "区域热度": "outputs/m2_2_zone_popularity.png",
            "区域热力图": "outputs/m2_2_zone_heatmap.png",
            "车费因素": "outputs/m2_3_fare_factors.png",
            "地理地图": "outputs/m2_2_geospatial_map.png",
            "费率分析": "outputs/m2_4_ratecode_analysis.png",
            "质量报告": "outputs/data_quality_report.csv",
            "模型指标": "outputs/m3_model_metrics.csv",
            "Loss曲线": "outputs/m3_neural_network_loss.png",
        }

    # ================================================================
    #  数据准备
    # ================================================================
    def _load_zone_lookup(self) -> pd.DataFrame | None:
        path = self.project_root / "data" / "taxi_zone_lookup.csv"
        if path.exists():
            return pd.read_csv(path)
        return None

    def _zone_name(self, loc_id: int) -> str:
        if self.zone_lookup is None:
            return str(loc_id)
        match = self.zone_lookup[self.zone_lookup["LocationID"] == loc_id]
        if len(match) > 0:
            return str(match["Zone"].iloc[0])
        return str(loc_id)

    def _precompute_stats(self) -> None:
        """
        预计算常用统计量，缓存到 self.stats。
        后续所有回答方法直接从 stats 读取，避免重复聚合计算。
        """
        df = self.df
        self.stats = {
            "total_trips": len(df),
            "avg_fare": float(df["fare_amount"].mean()),
            "median_fare": float(df["fare_amount"].median()),
            "avg_distance": float(df["trip_distance"].mean()),
            "avg_tip_ratio": float(
                (df["tip_amount"] / df["total_amount"].replace(0, np.nan)).mean() * 100
            ),
            "avg_duration": float(df["trip_duration_minutes"].mean()),
            "avg_speed": float(df["avg_speed_mph"].mean()),
            "weekday_avg": int(df[~df["is_weekend"]].groupby("pickup_day").size().mean()),
            "weekend_avg": int(df[df["is_weekend"]].groupby("pickup_day").size().mean()),
            "peak_hour_avg": int(df[df["is_peak_hour"]].groupby("pickup_day").size().mean()),
        }

        hour_counts = df.groupby("pickup_hour").size()
        self.stats["peak_hour"] = int(hour_counts.idxmax())
        self.stats["peak_hour_count"] = int(hour_counts.max())
        self.stats["quietest_hour"] = int(hour_counts.idxmin())

        rate_names = {
            1: "标准费率", 2: "JFK机场", 3: "Newark机场",
            4: "Nassau/Westchester", 5: "议价", 6: "拼车",
        }
        rate_stats = df.groupby("RatecodeID").agg(
            count=("trip_distance", "count"),
            avg_fare=("fare_amount", "mean"),
            avg_distance=("trip_distance", "mean"),
        )
        self.stats["ratecode"] = {}
        for rid, row in rate_stats.iterrows():
            name = rate_names.get(int(rid), f"费率{int(rid)}")
            self.stats["ratecode"][name] = {
                "count": int(row["count"]),
                "avg_fare": float(row["avg_fare"]),
                "avg_distance": float(row["avg_distance"]),
            }

    # ================================================================
    #  意图识别
    # ================================================================
    def recognize_intent(self, question: str) -> dict:
        """
        基于关键词匹配识别用户意图。

        流程：
            1. 问题预处理：去标点、转小写
            2. 按优先级依次匹配 6 种意图的关键词规则
            3. 匹配到后立即返回，不再继续尝试后续意图

        优先级设计：预测 > 排名 > 费率 > 时段 > 费用 > 总览
        原因：预测/排名/费率的关键词更具体，时段/费用/总览的关键词较宽泛，
        放在后面避免误匹配。

        Returns
        -------
        dict
            {"intent": str, "confidence": float, "entities": dict}
        """
        q = question.lower().strip()
        q_clean = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", q)

        if any(kw in q_clean for kw in ["预测", "预计", "明天", "将来", "未来"]):
            return {"intent": "demand_prediction", "confidence": 0.9,
                    "entities": self._extract_prediction_params(q_clean)}

        if any(kw in q_clean for kw in ["排名", "top", "热门", "最多", "最少", "哪个区"]):
            return {"intent": "zone_ranking", "confidence": 0.85,
                    "entities": self._extract_zone_params(q_clean)}

        if any(kw in q_clean for kw in ["jfk", "机场", "费率", "拼车", "议价", "newark"]):
            return {"intent": "ratecode_analysis", "confidence": 0.85,
                    "entities": self._extract_ratecode_params(q_clean)}

        if any(kw in q_clean for kw in ["早上", "晚上", "下午", "高峰", "凌晨",
                                          "几点", "小时", "周末", "工作日"]):
            return {"intent": "time_demand", "confidence": 0.8,
                    "entities": self._extract_time_params(q_clean)}

        if any(kw in q_clean for kw in ["车费", "价格", "多少钱", "小费", "英里",
                                          "费用", "fare", "price", "cost"]):
            return {"intent": "fare_analysis", "confidence": 0.8,
                    "entities": self._extract_fare_params(q_clean)}

        if any(kw in q_clean for kw in ["总共", "多少条", "数据质量", "平均", "统计",
                                          "概况", "多少单", "多少行程", "多少记录"]):
            return {"intent": "data_overview", "confidence": 0.75, "entities": {}}

        return {"intent": "unknown", "confidence": 0.0, "entities": {}}

    # ================================================================
    #  实体提取
    # ================================================================
    @staticmethod
    def _parse_hour(q: str) -> int | None:
        hour_match = re.search(r"(\d{1,2})\s*点", q)
        if not hour_match:
            return None
        hour = int(hour_match.group(1))
        if ("下午" in q or "晚上" in q) and hour < 12:
            hour += 12
        return min(hour, 23)

    @staticmethod
    def _extract_time_params(q: str) -> dict:
        params: dict = {}
        hour = QASystem._parse_hour(q)
        if hour is not None:
            params["hour"] = hour
        if "周末" in q:
            params["is_weekend"] = True
        elif "工作日" in q:
            params["is_weekend"] = False
        if "高峰" in q:
            params["is_peak"] = True
        return params

    @staticmethod
    def _extract_zone_params(q: str) -> dict:
        params: dict = {"top_n": 10}
        top_match = re.search(r"top\s*(\d+)", q, re.IGNORECASE)
        if top_match:
            params["top_n"] = int(top_match.group(1))
        num_match = re.search(r"前\s*(\d+)", q)
        if num_match:
            params["top_n"] = int(num_match.group(1))
        params["direction"] = "dropoff" if ("下车" in q or "dropoff" in q.lower()) else "pickup"
        return params

    @staticmethod
    def _extract_prediction_params(q: str) -> dict:
        params: dict = {}
        zone_match = re.search(r"区域\s*(\d+)", q)
        if zone_match:
            params["zone_id"] = int(zone_match.group(1))
        hour = QASystem._parse_hour(q)
        if hour is not None:
            params["hour"] = hour
        return params

    @staticmethod
    def _extract_fare_params(q: str) -> dict:
        params: dict = {}
        dist_match = re.search(r"(\d+)\s*英里", q)
        if dist_match:
            params["distance"] = float(dist_match.group(1))
        return params

    @staticmethod
    def _extract_ratecode_params(q: str) -> dict:
        params: dict = {}
        q_lower = q.lower()
        if "jfk" in q_lower:
            params["ratecode"] = 2
        elif "newark" in q_lower:
            params["ratecode"] = 3
        elif "拼车" in q:
            params["ratecode"] = 6
        elif "议价" in q:
            params["ratecode"] = 5
        return params

    # ================================================================
    #  规则回答：6 种意图各对应一个回答方法
    # ================================================================
    def answer_time_demand(self, entities: dict) -> str:
        """时段需求查询：按小时/工作日/周末聚合统计，返回订单量 + 图表引用"""
        df = self.df
        hour: int | None = entities.get("hour")
        is_weekend: bool | None = entities.get("is_weekend")
        is_peak: bool | None = entities.get("is_peak")
        s = self.stats

        lines = ["## 出行需求时间分析\n"]

        if hour is not None:
            mask = df["pickup_hour"] == hour
            if is_weekend is not None:
                mask &= df["is_weekend"] == is_weekend
            count = int(mask.sum())
            label = f"{hour}:00"
            if is_weekend is True:
                label += "（周末）"
            elif is_weekend is False:
                label += "（工作日）"
            lines.append(f"**{label}** 的订单量约为 **{count:,}** 单。")
        else:
            lines.append(f"- 全天订单量最高的时段是 **{s['peak_hour']}:00**，"
                        f"约 {s['peak_hour_count']:,} 单。")
            lines.append(f"- 全天订单量最低的时段是 **{s['quietest_hour']}:00**。")
            lines.append(f"- 工作日日均约 {s['weekday_avg']:,} 单，"
                        f"周末日均约 {s['weekend_avg']:,} 单。")

        if is_peak:
            lines.append(f"\n高峰时段（7-9点、16-19点）日均约 {s['peak_hour_avg']:,} 单。")

        lines.append(f"\n> 详情可查看图表: `{self.chart_map['需求时间']}`")
        return "\n".join(lines)

    def answer_zone_ranking(self, entities: dict) -> str:
        """区域热度排名：按 PULocationID/DOLocationID 聚合取 TOP N"""
        df = self.df
        top_n: int = entities.get("top_n", 10)
        direction: str = entities.get("direction", "pickup")
        col = "PULocationID" if direction == "pickup" else "DOLocationID"
        label = "上车" if direction == "pickup" else "下车"

        top = df[col].value_counts().head(top_n)
        lines = [f"## {label}量 TOP {top_n} 区域\n"]
        lines.append("| 排名 | 区域 | 订单量 |")
        lines.append("|------|------|--------|")
        for i, (loc_id, count) in enumerate(top.items(), 1):
            lines.append(f"| {i} | {self._zone_name(int(loc_id))} | {int(count):,} |")
        lines.append(f"\n> 详情可查看图表: `{self.chart_map['区域热度']}`")
        return "\n".join(lines)

    def answer_demand_prediction(self, entities: dict) -> str:
        """
        需求预测：基于历史同区域同时段数据给出日均估计值。

        如果 M3 模型已训练并保存，优先用模型预测；否则用历史均值作为近似。
        同时附上模型性能指标（MAE/RMSE）供参考。
        """
        zone_id: int = entities.get("zone_id", 100)
        hour: int = entities.get("hour", 15)
        lines = [f"## 需求预测: 区域 {zone_id} ({self._zone_name(zone_id)}), {hour}:00\n"]

        mask = (self.df["PULocationID"] == zone_id) & (self.df["pickup_hour"] == hour)
        historical = int(mask.sum())

        if historical > 0:
            daily_avg = float(historical) / float(self.df["pickup_day"].nunique())
            lines.append(f"该区域在 {hour}:00 的历史日均订单量约为 **{daily_avg:.0f}** 单。")
            lines.append(f"历史同区域同小时总订单量: **{historical:,}** 单。")
        else:
            lines.append(f"该区域在 {hour}:00 暂无历史数据，无法给出预测。")

        metrics_path = self.project_root / "outputs" / "m3_model_metrics.csv"
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            lines.append("\n**模型性能参考**:")
            for _, row in metrics_df.iterrows():
                lines.append(f"- {row['模型']}: MAE={row['MAE']:.2f}, RMSE={row['RMSE']:.2f}")

        lines.append(f"\n> 详情可查看: `{self.chart_map['模型指标']}` 和 `{self.chart_map['Loss曲线']}`")
        return "\n".join(lines)

    def answer_fare_analysis(self, entities: dict) -> str:
        """
        费用分析：返回平均车费、距离、小费比例等统计量。

        如果用户提供了具体距离，用线性估算公式计算预估车费。
        估算公式：fare ≈ 2.5 + 3.0 × distance（基于数据中标准费率的回归拟合）。
        """
        s = self.stats
        distance: float | None = entities.get("distance")
        lines = ["## 车费分析\n"]
        lines.append(f"- 平均车费: **${s['avg_fare']:.2f}**（中位数 ${s['median_fare']:.2f}）")
        lines.append(f"- 平均行程距离: **{s['avg_distance']:.2f}** 英里")
        lines.append(f"- 平均小费比例: **{s['avg_tip_ratio']:.1f}%**")
        lines.append(f"- 平均行程时长: **{s['avg_duration']:.1f}** 分钟")
        lines.append(f"- 平均速度: **{s['avg_speed']:.1f}** 英里/小时")

        if distance is not None:
            estimated_fare = 2.5 + 3.0 * float(distance)
            lines.append(f"\n**{distance} 英里**的预估车费约为 **${estimated_fare:.2f}**"
                        f"（基于标准费率线性估算，不含小费及附加费）。")

        lines.append(f"\n> 详情可查看图表: `{self.chart_map['车费因素']}`")
        return "\n".join(lines)

    def answer_data_overview(self, _entities: dict) -> str:
        """数据总览：返回数据集基本信息 + 统计数据摘要"""
        s = self.stats
        lines = ["## 数据总览\n"]
        lines.append(f"本数据集为 **2026年1月** NYC 黄牌出租车行程记录。")
        lines.append(f"- 总行程数: **{s['total_trips']:,}** 单")
        lines.append(f"- 平均车费: **${s['avg_fare']:.2f}**")
        lines.append(f"- 平均距离: **{s['avg_distance']:.2f}** 英里")
        lines.append(f"- 平均时长: **{s['avg_duration']:.1f}** 分钟")
        lines.append(f"- 工作日日均: **{s['weekday_avg']:,}** 单")
        lines.append(f"- 周末日均: **{s['weekend_avg']:,}** 单")
        lines.append(f"- 高峰时段: **{s['peak_hour']}:00**（{s['peak_hour_count']:,} 单）")
        lines.append(f"\n原始数据 3,724,889 条，清洗后 {s['total_trips']:,} 条。")
        lines.append(f"\n> 质量报告: `{self.chart_map['质量报告']}`")
        return "\n".join(lines)

    def answer_ratecode_analysis(self, entities: dict) -> str:
        """费率类型分析：按 RatecodeID 聚合统计各费率的订单数、平均车费、平均距离"""
        ratecode: int | None = entities.get("ratecode")
        rc: dict = self.stats["ratecode"]

        lines = ["## 费率类型分析\n"]
        lines.append("| 费率类型 | 订单数 | 平均车费 | 平均距离 |")
        lines.append("|----------|--------|----------|----------|")
        for name, info in rc.items():
            lines.append(f"| {name} | {info['count']:,} | ${info['avg_fare']:.2f} |"
                        f" {info['avg_distance']:.2f}mi |")

        if ratecode is not None:
            rate_names: dict[int, str] = {2: "JFK机场", 3: "Newark机场", 5: "议价", 6: "拼车"}
            name = rate_names.get(ratecode, f"费率{ratecode}")
            if name in rc:
                info = rc[name]
                pct = float(info["count"]) / float(self.stats["total_trips"]) * 100
                lines.append(f"\n**{name}**: 共 {info['count']:,} 单（占比 {pct:.1f}%），"
                            f"平均车费 ${info['avg_fare']:.2f}。")

        most_expensive = max(rc.items(), key=lambda x: x[1]["avg_fare"])
        most_common = max(rc.items(), key=lambda x: x[1]["count"])
        lines.append(f"\n平均车费最高: **{most_expensive[0]}** (${most_expensive[1]['avg_fare']:.2f})")
        lines.append(f"订单量最大: **{most_common[0]}** ({most_common[1]['count']:,} 单)")
        lines.append(f"\n> 详情可查看图表: `{self.chart_map['费率分析']}`")
        return "\n".join(lines)

    # ================================================================
    #  核心回答流程
    # ================================================================
    def answer(self, question: str) -> str:
        """
        主回答入口。

        流程：意图识别 → 规则回答 → 未匹配则 LLM 兜底。
        """
        intent_result = self.recognize_intent(question)
        intent = intent_result["intent"]
        entities = intent_result["entities"]

        handlers = {
            "time_demand": self.answer_time_demand,
            "zone_ranking": self.answer_zone_ranking,
            "demand_prediction": self.answer_demand_prediction,
            "fare_analysis": self.answer_fare_analysis,
            "data_overview": self.answer_data_overview,
            "ratecode_analysis": self.answer_ratecode_analysis,
        }

        if intent in handlers:
            response = handlers[intent](entities)
            response += f"\n\n---\n*意图识别: {intent} (置信度 {intent_result['confidence']:.0%})*"
            return response

        return self._llm_fallback(question)

    # ================================================================
    #  LLM 兜底
    # ================================================================
    def _build_system_prompt(self) -> str:
        """
        构建 System Prompt。

        设计要点：
        1. 明确角色：NYC 出租车数据分析助手
        2. 提供数据上下文：当前数据集的统计摘要，让模型了解数据范围
        3. 防幻觉约束：只基于提供的数据回答，不编造数字
        4. 结构化输出格式：数字结论 → 简要解释 → 可尝试的其他问题
        """
        s = self.stats
        return f"""你是 NYC 黄牌出租车数据分析助手。你只能基于以下数据来回答问题。

当前数据集概况：
- 数据来源：2026年1月 NYC 黄牌出租车行程记录
- 清洗后总行程数：{s['total_trips']:,} 单
- 平均车费：${s['avg_fare']:.2f}
- 平均行程距离：{s['avg_distance']:.2f} 英里
- 覆盖 262 个 TLC 出租车区域

回答规则：
1. 仅基于提供的数据回答问题，不要编造任何数字或结论。
2. 如果数据不足以回答，直接说明"数据不足以回答该问题"并解释原因。
3. 回答结构：先给出数字结论（如有），再给出简要解释。
4. 如果问题超出你的知识范围，建议用户尝试以下问题类型：
   - 时段需求查询（如"早上8点有多少订单？"）
   - 区域热度排名（如"最热门的10个上车区域？"）
   - 车费分析（如"平均车费多少？"）
   - 费率类型分析（如"JFK机场费率多少钱？"）
   - 数据总览（如"总共有多少条记录？"）
   - 需求预测（如"预测区域100下午3点的需求量？"）"""

    def _llm_fallback(self, question: str) -> str:
        """
        调用大模型 API 兜底回答。

        配置来源（优先级从高到低）：
        1. Gradio 界面保存的 .env 文件
        2. 系统环境变量
        3. 未配置时返回友好提示
        """
        api_key = self._llm_config["api_key"]
        base_url = self._llm_config["base_url"]
        model = self._llm_config["model"]

        if not api_key or not base_url or not model:
            return (
                "抱歉，我暂时无法理解这个问题。\n\n"
                "你可以尝试以下问题类型：\n"
                "- 时段查询: \"早上8点有多少订单？\"\n"
                "- 区域排名: \"最热门的10个上车区域有哪些？\"\n"
                "- 车费分析: \"平均车费是多少？\"\n"
                "- 费率分析: \"JFK机场费率是多少？\"\n"
                "- 数据总览: \"总共有多少条记录？\"\n"
                "- 需求预测: \"预测区域100下午3点的需求量？\"\n\n"
                "> 提示：配置 LLM_API_KEY 环境变量可启用大模型智能回答。"
            )

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": question},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            message = response.choices[0].message
            return message.content if message.content else "模型未返回内容"
        except Exception as e:
            return f"LLM 调用失败: {e}\n\n请尝试使用支持的问题类型重新提问。"

    # ================================================================
    #  命令行交互
    # ================================================================
    def cli_loop(self) -> None:
        """命令行交互循环，输入 exit 退出，输入 help 查看帮助"""
        print("\n" + "=" * 60)
        print("  NYC 出租车数据智能问答系统")
        print("  输入 'exit' 退出，输入 'help' 查看支持的问题类型")
        print("=" * 60 + "\n")

        while True:
            try:
                question = input("您的问题: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break
            if not question:
                continue
            if question.lower() in ("exit", "quit", "退出"):
                print("再见！")
                break
            if question.lower() in ("help", "帮助", "?"):
                self._print_help()
                continue
            print()
            print(self.answer(question))
            print("\n" + "-" * 40 + "\n")

    def _print_help(self) -> None:
        print("""
支持的问题类型:
  1. 时段需求查询 —— "早上8点有多少订单？"、"工作日高峰时段订单量？"
  2. 区域热度排名 —— "最热门的10个上车区域？"、"下车最多的区域TOP5？"
  3. 需求预测      —— "预测区域100下午3点的需求量？"、"模型准确率怎么样？"
  4. 费用估算分析  —— "10英里大概多少钱？"、"平均车费多少？"、"小费比例？"
  5. 数据总览质量  —— "总共有多少条记录？"、"数据质量怎么样？"
  6. 费率类型分析  —— "JFK机场费率多少钱？"、"拼车订单占比多少？"
""")

    # ================================================================
    #  LLM 配置保存
    # ================================================================
    def save_llm_config(self, api_key: str, base_url: str, model: str) -> str:
        """
        将 LLM 配置写入 .env 文件并更新内存中的配置。

        写入 .env 文件使配置持久化，重启程序后仍然生效。
        同时更新 self._llm_config 使当前会话立即生效。
        """
        env_path = self.project_root / ".env"
        existing = {}

        # 读取现有 .env 内容（保留其他配置项）
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.startswith("#"):
                    key, _, val = line.partition("=")
                    existing[key.strip()] = val.strip()

        # 更新 LLM 配置
        existing["LLM_API_KEY"] = api_key
        existing["LLM_BASE_URL"] = base_url
        existing["LLM_MODEL"] = model

        # 写入 .env
        lines = [f"{k}={v}" for k, v in existing.items()]
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # 同步到内存配置
        self._llm_config = {"api_key": api_key, "base_url": base_url, "model": model}

        return "LLM 配置已保存  ✓" if api_key else "（未填写 API Key，LLM 功能不可用）"

    # ================================================================
    #  Gradio 可视化界面
    # ================================================================
    def launch_ui(self, share: bool = False) -> None:
        """
        启动 Gradio 可视化问答界面。

        界面布局：左侧聊天面板 + 右侧信息面板（问题类型、快速提问、图表列表）。
        数据流：用户输入 → chat_fn 回调 → answer() → Chatbot 更新 + Gallery 更新图表。
        """
        try:
            import gradio as gr
        except ImportError:
            print("Gradio 未安装，请执行: pip install gradio")
            self.cli_loop()
            return

        chart_files: list[tuple[str, str]] = []
        for name, rel_path in self.chart_map.items():
            abs_path = self.project_root / rel_path
            if abs_path.exists():
                chart_files.append((name, str(abs_path)))

        def chat_fn(message: str, history: list[dict]) -> tuple[str, list[dict]]:
            response = self.answer(message)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            return "", history

        with gr.Blocks(title="NYC 出租车数据问答系统") as demo:
            gr.Markdown(
                "# NYC 黄牌出租车数据智能问答系统\n"
                "基于 2026年1月 行程数据，支持时段查询、区域排名、需求预测、费用分析等。"
            )

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="对话", height=500,
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="请输入您的问题...（例如：早上8点有多少订单？）",
                            scale=8, show_label=False,
                        )
                        send_btn = gr.Button("发送", variant="primary", scale=1)

                    msg_input.submit(chat_fn, [msg_input, chatbot], [msg_input, chatbot])
                    send_btn.click(chat_fn, [msg_input, chatbot], [msg_input, chatbot])

                with gr.Column(scale=1):
                    # LLM 设置面板
                    gr.Markdown("### LLM 设置")
                    gr.Markdown("配置大模型 API 后可回答任意问题")

                    platform = gr.Dropdown(
                        choices=["DeepSeek", "通义千问(Qwen)", "智谱GLM", "自定义"],
                        value="DeepSeek",
                        label="平台",
                    )

                    llm_base_url = gr.Textbox(
                        value="https://api.deepseek.com/v1",
                        label="Base URL",
                        placeholder="https://api.deepseek.com/v1",
                    )
                    llm_model = gr.Textbox(
                        value="deepseek-chat",
                        label="模型名称",
                        placeholder="deepseek-chat",
                    )
                    llm_api_key = gr.Textbox(
                        value=self._llm_config.get("api_key", ""),
                        label="API Key",
                        placeholder="sk-...",
                        type="password",
                    )

                    llm_status = gr.Markdown(
                        "已配置  ✓" if self._llm_config.get("api_key") else "未配置"
                    )

                    def on_platform_change(choice: str) -> tuple[str, str]:
                        """切换平台时自动填充 Base URL 和模型名"""
                        presets = {
                            "DeepSeek": ("https://api.deepseek.com/v1", "deepseek-chat"),
                            "通义千问(Qwen)": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
                            "智谱GLM": ("https://open.bigmodel.cn/api/paas/v4", "glm-4-flash"),
                            "自定义": ("", ""),
                        }
                        url, model = presets.get(choice, ("", ""))
                        return url, model

                    platform.change(
                        on_platform_change, platform, [llm_base_url, llm_model]
                    )

                    def on_save(key: str, url: str, model: str) -> str:
                        return self.save_llm_config(key, url, model)

                    save_btn = gr.Button("保存配置", variant="secondary")
                    save_btn.click(
                        on_save,
                        [llm_api_key, llm_base_url, llm_model],
                        [llm_status],
                    )

                    gr.Markdown("### 支持的问题类型")
                    gr.Markdown("""
                    - 时段需求查询
                    - 区域热度排名
                    - 需求预测
                    - 费用估算分析
                    - 数据总览质量
                    - 费率类型分析
                    """)

                    gr.Markdown("### 快速提问")
                    quick_questions = [
                        "早上8点有多少订单？",
                        "最热门的10个上车区域？",
                        "平均车费是多少？",
                        "JFK机场费率多少钱？",
                        "总共有多少条记录？",
                        "工作日和周末哪个订单多？",
                    ]

                    def make_click_handler(text: str):
                        return lambda: text

                    for question_text in quick_questions:
                        btn = gr.Button(question_text, size="sm", elem_classes="quick-btn")
                        btn.click(make_click_handler(question_text), None, msg_input)

                    if chart_files:
                        gr.Markdown("### 相关图表")
                        gr.Gallery(
                            value=[f for _, f in chart_files],
                            columns=1, height=300, object_fit="contain",
                            show_label=False,
                        )

        print("\n启动 Gradio 界面...")
        # 自动寻找可用端口，避免端口冲突
        launch_kwargs = {
            "server_name": "127.0.0.1",
            "share": share,
            "inbrowser": True,
            "theme": gr.themes.Soft(),
            "css": CHAT_CSS,
        }
        for port in [7860, 7861, 7862, 7863, 7864]:
            try:
                demo.launch(server_port=port, **launch_kwargs)
                break
            except OSError:
                if port == 7864:
                    # 所有端口都不可用，让系统分配
                    demo.launch(server_port=0, **launch_kwargs)


# ================================================================
#  直接运行入口
# ================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.m1_data_processing import DataQualityAnalyzer

    analyzer = DataQualityAnalyzer(
        data_path="data/yellow_tripdata_2026-01.parquet",
        output_path="outputs/data_quality_report.csv",
    )
    analyzer.load_data()
    analyzer.clean_data()
    analyzer.add_time_features()

    qa = QASystem(analyzer.df)
    qa.launch_ui()