<div align="center">

# From Context-Aware to Conflict-Aware: Generalizing Contrastive Decoding for Knowledge Conflict in LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2606.10298-b31b1b.svg)](https://arxiv.org/abs/2606.10298)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg)](https://www.python.org/)

</div>

**Runze Jiang**<sup>1,2</sup>, **Taiqiang Wu**<sup>3</sup>, **Yan Wang**<sup>2</sup>, **Bingyu Zhu**<sup>2†</sup>, **Longtao Huang**<sup>2</sup>

<sup>1</sup> Peking University &nbsp; | &nbsp; <sup>2</sup> Alibaba Group &nbsp; | &nbsp; <sup>3</sup> The University of Hong Kong &nbsp; | &nbsp; <sup>†</sup> Corresponding author

[English](./README.md) | [中文](./README_zh.md)

---

## 🔍 概述

大语言模型经常面临参数记忆（先验知识）与检索到的上下文证据之间的**知识冲突**。现有对比解码方法将这一问题建模为 *context-aware amplification*——增强上下文条件分布以压制仅依赖先验的分布——但这隐含地偏向上下文，当先验正确而上下文具有误导性时反而会覆盖模型原本正确的概率结构。

我们将这一问题重新定义为 **conflict-aware authority allocation**（冲突感知的权威分配）。在 logit 空间中，先验与上下文分布的仿射组合导出一个**幂乘族** $q_{\tau,t}(y) \propto p_{\text{pri},t}(y)^{1-\tau}p_{\text{ctx},t}(y)^{\tau}$，它将现有对比解码方法统一为固定指数的特例。该族暴露了一个**机制非对称性**：插值（$\tau \in (0,1)$）在上下文正确时纠正不足，而外推（$\tau > 1$）在先验正确时无界放大错误——这是一个结构性缺陷，任何静态 $\tau$ 都无法同时覆盖两个冲突方向。

为使两个冲突方向均可衡量，我们提出 **TriState-Bench**，通过 Greedy-Anchored Prior Screening (GAPS) 在每个模型上先探测其参数化先验答案，再为每个模型单独标注校正（correction）、抵抗（resistance）和一致（agreement）三种状态。为实现权威分配，我们提出 **Gated Reversal Decoding (GRD)**，在首次冲突步长对先验和上下文的信任进行门控，并通过成对逆转阈值解决冲突 token。

**核心贡献：**
- 🔬 **幂乘族 + 机制非对称性** —— 统一理论框架，揭示插值与外推在相反冲突方向上的结构性失败
- 📊 **TriState-Bench** —— 模型感知评测，使两个冲突方向均可衡量
- 🚀 **Gated Reversal Decoding (GRD)** —— 冲突感知解码，将抵抗 EM 从 $2.1$ 提升至 $20.8$，且不牺牲校正与一致性能
- 🔁 **可复现 baseline** —— 原始 CAD、AdaCAD、CoCoA、COIECD 实现完整保留

## 📰 最新动态

- **2026.06.09** — 论文发布于 arXiv：[2606.10298](https://arxiv.org/abs/2606.10298)
- 代码与评测数据已公开。

## 🔗 链接

| | |
|---|---|
| 📄 **论文** | [arXiv:2606.10298](https://arxiv.org/abs/2606.10298) |
| 📦 **代码** | 本仓库 |

---

## ⚙️ 环境配置

```bash
conda create -n conflict python==3.10 -y
conda activate conflict
pip install -r requirements.txt
```

主要依赖：`torch==2.5.0`、`transformers==4.51.3`、`accelerate`、`datasets`、`bitsandbytes`。

## 🧭 项目结构

```text
.
├── methods/                         # 统一单卡解码方法
│   ├── inference.py                 # 单卡推理入口
│   ├── base.py                      # 解码方法接口
│   ├── greedy.py                    # 带上下文的 greedy 解码
│   ├── greedy_no_ctx.py             # 无上下文（仅 prior）的 greedy 解码
│   ├── cad.py                       # CAD / Context-Aware Decoding
│   ├── adacad.py                    # AdaCAD / 自适应 CAD
│   ├── cocoa.py                     # CoCoA 解码
│   ├── coiecd.py                    # COIECD 熵约束解码
│   ├── grd.py                       # GRD: Gated Reversal Decoding
│   └── simple_interp.py             # 固定 α 插值
├── evaluation/                      # 统一评测框架
│   ├── unified_eval.py              # 评测入口
│   ├── utils.py                     # 数据读取与归一化
│   └── metrics/
│       └── qa_metrics.py            # EM, substring EM, F1
├── data/                            # 统一格式 benchmark 数据
│   ├── SQA/                         # 标准 QA 评测集
│   │   ├── hotpotqa.jsonl
│   │   ├── nq.jsonl
│   │   ├── tabmwp.jsonl
│   │   └── triviaqa.jsonl
│   └── TriState/                    # 按模型划分的三态评测集
│       └── {MODEL_SHORT}/
│           ├── C_right_P_wrong.jsonl
│           ├── C_right_P_right.jsonl
│           └── C_wrong_P_right.jsonl
├── CAD/                             # 原始 CAD baseline
├── AdaCAD/                          # 原始 AdaCAD baseline
├── CoCoA/                           # 原始 CoCoA baseline
├── COIECD/                          # 原始 COIECD 资源
├── inference_qa_self.sh             # 单卡 QA 批量推理
├── inference_qa_origin.sh           # 双卡 QA baseline 批量推理
├── inference_tristate_self.sh       # 单卡三态批量推理
├── inference_tristate_origin.sh     # 双卡三态 baseline 批量推理
└── requirements.txt
```

## 🧩 已实现方法

所有方法均读取统一的 dual-process JSONL 格式，同时使用 context prompt 与 prior prompt。

| 方法 | 文件 | 说明 |
|------|------|------|
| `greedy` | `methods/greedy.py` | 使用带上下文 prompt 的 greedy 解码 |
| `greedy_no_ctx` | `methods/greedy_no_ctx.py` | 仅使用 prior prompt 的 greedy 解码 |
| `coiecd` | `methods/coiecd.py` | 熵约束融合 context/prior logits |
| `grd` | `methods/grd.py` | **Gated Reversal Decoding** — 有状态冲突门控 + τ\* 成对逆转 |
| `simple_interp` | `methods/simple_interp.py` | 固定 α 的 context/prior logits 插值 |

原始 baseline 代码保留在 `CAD/`、`AdaCAD/`、`CoCoA/` 和 `COIECD/` 中，用于复现实验。

## 📦 数据

所有统一数据集使用 **dual-process JSONL** 格式。每个样本由两行组成，共享同一个 `input_index`：

```jsonl
{"input_index": 0, "assigned_process": 0, "context_string": "<context+question>", "assigned_weight": 2, "gold_answers": "answer", "benchmark": "nq", "task_type": "qa"}
{"input_index": 0, "assigned_process": 1, "context_string": "<question_only>", "assigned_weight": -1, "gold_answers": "answer", "benchmark": "nq", "task_type": "qa"}
```

- `assigned_process=0` — 包含证据和问题的上下文 prompt
- `assigned_process=1` — 仅包含问题的 prior prompt

### Standard QA Benchmarks

```text
data/SQA/hotpotqa.jsonl
data/SQA/nq.jsonl
data/SQA/tabmwp.jsonl
data/SQA/triviaqa.jsonl
```

### TriState-Bench

```text
data/TriState/{MODEL_SHORT}/C_right_P_wrong.jsonl
data/TriState/{MODEL_SHORT}/C_right_P_right.jsonl
data/TriState/{MODEL_SHORT}/C_wrong_P_right.jsonl
```

当前 `MODEL_SHORT` 目录覆盖 Llama-2、Llama-3、Mistral 和 Gemma2.5 的多个模型版本。

## 🚀 推理

两条执行路径：

| 路径 | 入口 | GPU |
|------|------|-----|
| **统一单卡方法** | `methods/inference.py` | 1 |
| **原始 baseline** | `{CAD,AdaCAD,CoCoA}/run_group_decode_fileio.sh` | 2 |

### 🖥️ 单卡推理

支持方法：`greedy`, `greedy_no_ctx`, `cad`, `adacad`, `cocoa`, `coiecd`, `grd`, `simple_interp`

```bash
# Greedy
python -m methods.inference --method greedy \
    --model meta-llama/Meta-Llama-3-8B \
    --input_path data/SQA/nq.jsonl \
    --output_path results/greedy/Meta-Llama-3-8B/nq.jsonl \
    --max_new_tokens 32

# COIECD
python -m methods.inference --method coiecd \
    --model meta-llama/Meta-Llama-3-8B \
    --input_path data/SQA/nq.jsonl \
    --output_path results/coiecd/Meta-Llama-3-8B/nq.jsonl \
    --max_new_tokens 32 --alpha 1.0 --threshold_ratio 4

# Simple interpolation
python -m methods.inference --method simple_interp \
    --model meta-llama/Meta-Llama-3-8B \
    --input_path data/SQA/nq.jsonl \
    --output_path results/simple_interp/Meta-Llama-3-8B/nq.jsonl \
    --max_new_tokens 32 --interp_alpha 0.75

# GRD
python -m methods.inference --method grd \
    --model meta-llama/Meta-Llama-3-8B \
    --input_path data/SQA/nq.jsonl \
    --output_path results/grd/Meta-Llama-3-8B/nq.jsonl \
    --max_new_tokens 32 --grd_lambda 0.75
```

### ⚙️ 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--method` | 解码方法名称 | *必填* |
| `--model` | HuggingFace 模型名或本地路径 | *必填* |
| `--input_path` | dual-process JSONL 输入文件 | *必填* |
| `--output_path` | 预测结果输出路径 | *必填* |
| `--max_new_tokens` | 最大生成 token 数 | `32` |
| `--max_ctx_len` | 输入截断长度 | `4064` |
| `--seed` | 随机种子 | `42` |
| `--dtype` | 模型精度 | `float16` |
| `--alpha` | CAD / COIECD 的 α | `1.0` |
| `--threshold_ratio` | COIECD 熵阈值比例 | `4` |
| `--warmup_beta` | AdaCAD 最小 α | `0.0` |
| `--interp_alpha` | SimpleInterp 的 context 权重 | `0.75` |
| `--cocoa_global_alpha` | CoCoA 全局混合 α | `0.5` |
| `--cocoa_gamma` | CoCoA γ 权重 | `1.0` |
| `--cocoa_lambda_pm` | CoCoA 扰动权重 | `100.0` |
| `--grd_lambda` | GRD 承诺深度 (τ = (1−λ)·τ\*) | `0.75` |
| `--grd_gate_mode` | GRD 首次冲突门控信号 | `full` |

> `methods/inference.py` 支持**断点续跑**：若 `output_path` 中已有某些 `input_index` 的结果，将自动跳过。

### 🧪 批量推理脚本

```bash
# 单卡 QA 评测
MODEL_NAME=meta-llama/Meta-Llama-3-8B \
METHODS="grd" \
BENCHMARKS="nq tabmwp triviaqa hotpotqa" \
GPU=0 \
OUTPUT_ROOT=results_new \
OVERWRITE=0 \
bash inference_qa_self.sh

# 双卡原始 baseline
MODEL_NAME=google/gemma-2-9b \
DEVICE="0,1" \
METHODS="cad adacad cocoa" \
bash inference_qa_origin.sh

# 三态 — 单卡
MODEL_SHORT=Gemma2.5-7B \
METHODS="greedy greedy_no_ctx simple_interp_0.25 simple_interp_0.5 simple_interp_0.75 coiecd grd" \
bash inference_tristate_self.sh

# 三态 — 双卡 baseline
MODEL_SHORT=Meta-Llama-3-8B \
DEVICE="0,1" \
METHODS="cad_0.25 cad_0.5 cad_0.75 cad adacad cocoa" \
bash inference_tristate_origin.sh
```

批量脚本包含过长样本过滤和已完成输出检测。

### 🧱 原始 Baseline 推理

原始 CAD / AdaCAD / CoCoA 实现使用双进程推理，需要两张 GPU：

```bash
export MODEL_NAME=meta-llama/Meta-Llama-3-8B

# CAD
cd CAD && bash run_group_decode_fileio.sh \
    42 "0,1" "fin|../data/SQA/nq.jsonl" \
    4096 4064 32 0.0 \
    ../results/cad/Meta-Llama-3-8B/nq.jsonl

# AdaCAD
cd AdaCAD && bash run_group_decode_fileio.sh \
    42 "0,1" "fin|../data/SQA/nq.jsonl" \
    4096 4064 32 0.0 \
    2 no 0 ../results/adacad/Meta-Llama-3-8B/nq.jsonl

# CoCoA
cd CoCoA && bash run_group_decode_fileio.sh \
    42 "0,1" "fin|../data/SQA/nq.jsonl" \
    4096 4064 32 0.0 \
    2 no 0 ../results/cocoa/Meta-Llama-3-8B/nq.jsonl
```

## 📊 评测

```bash
python -m evaluation.unified_eval \
    --task_type qa \
    --gold_path data/SQA/nq.jsonl \
    --pred_path results/greedy/Meta-Llama-3-8B/nq.jsonl \
    --metrics em,f1,substring_em
```

QA 指标：

| 指标 | 说明 |
|------|------|
| `em` | 使用 SQuAD 风格归一化的 Exact Match |
| `substring_em` | gold answer 是否出现在 prediction 中 |
| `f1` | token 级 F1 |

保存评测结果：

```bash
python -m evaluation.unified_eval \
    --task_type qa \
    --gold_path data/SQA/nq.jsonl \
    --pred_path results/greedy/Meta-Llama-3-8B/nq.jsonl \
    --metrics em,f1,substring_em \
    --output_path results/greedy/Meta-Llama-3-8B/nq_eval.json
```

输出示例：

```json
{
  "em": 0.4523,
  "f1": 0.5812,
  "substring_em": 0.6134,
  "total": 4000,
  "missing": 0
}
```

## 🙏 致谢

本工作由阿里巴巴集团通过 Alibaba Research Intern Program 支持。

## 📝 引用

```bibtex
@misc{jiang2026contextawareconflictawaregeneralizingcontrastive,
      title={From Context-Aware to Conflict-Aware: Generalizing Contrastive Decoding for Knowledge Conflict in LLMs},
      author={Runze Jiang and Taiqiang Wu and Yan Wang and Bingyu Zhu and Longtao Huang},
      year={2026},
      eprint={2606.10298},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.10298},
}
```
