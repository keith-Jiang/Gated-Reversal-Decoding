# From Context-Aware to Conflict-Aware: Generalizing Contrastive Decoding for Knowledge Conflict in LLMs

[English](./README.md) | [中文](./README_zh.md)

本仓库是论文 **From Context-Aware to Conflict-Aware: Generalizing Contrastive Decoding for Knowledge Conflict in LLMs** 的代码仓库，包含知识冲突场景下的 conflict-aware decoding 实现、统一单卡推理入口、原始双卡 baseline 封装、QA/tristate 数据与评测脚本。

## 📚 目录

- [📰 最新动态](#-最新动态)
- [⚙️ 环境配置](#️-环境配置)
- [🧭 项目结构](#-项目结构)
- [🧩 已实现模块](#-已实现模块)
- [📦 数据](#-数据)
- [🚀 推理](#-推理)
- [📊 评测](#-评测)
- [🙏 致谢](#-致谢)
- [📝 引用](#-引用)

## 📰 最新动态

- **2026.06.09**：论文已上传至 arXiv：[From Context-Aware to Conflict-Aware: Generalizing Contrastive Decoding for Knowledge Conflict in LLMs](https://arxiv.org/abs/2606.10298)。

## ⚙️ 环境配置

```bash
conda create -n conflict python==3.10 -y
conda activate conflict
pip install -r requirements.txt
```

主要依赖包括 `torch==2.5.0`、`transformers==4.51.3`、`accelerate`、`datasets`、`bitsandbytes` 以及常用评测包。

## 🧭 项目结构

```text
.
├── methods/                         # 统一单卡解码方法
│   ├── inference.py                 # 单卡推理入口
│   ├── base.py                      # 解码方法接口
│   ├── greedy.py                    # 带上下文的 greedy 解码
│   ├── cad.py                       # CAD / Context-Aware Decoding
│   ├── adacad.py                    # AdaCAD / 自适应 CAD
│   ├── cocoa.py                     # CoCoA 解码
│   ├── coiecd.py                    # COIECD 熵约束解码
│   ├── simple_interp.py             # 固定 alpha 插值
│   └── arr.py                       # ARR: Adaptive Regime Routing
├── evaluation/                      # 统一评测框架
│   ├── unified_eval.py              # 评测入口
│   ├── utils.py                     # 数据读取与归一化
│   └── metrics/
│       └── qa_metrics.py            # EM, substring EM, F1
├── data/                            # 统一格式 benchmark 数据
│   ├── Trad_QA/                     # 传统 QA benchmarks
│   │   ├── hotpotqa.jsonl
│   │   ├── nq.jsonl
│   │   ├── tabmwp.jsonl
│   │   └── triviaqa.jsonl
│   └── TriState/                    # 按模型划分的 tristate benchmarks
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
├── inference_tristate_self.sh       # 单卡 tristate 批量推理
├── inference_tristate_origin.sh     # 双卡 tristate baseline 批量推理
└── requirements.txt
```

## 🧩 已实现模块

统一单卡实现位于 `methods/`。所有方法都读取同一种 dual-process JSONL 格式，并同时使用 context prompt 与 prior prompt。

| 方法 | 文件 | 说明 |
|------|------|------|
| `greedy` | `methods/greedy.py`, `methods/inference.py` | 使用带上下文 prompt 的 greedy 解码。 |
| `greedy_no_ctx` | `methods/inference.py` | 使用无上下文 prompt 的 greedy 解码。 |
| `cad` | `methods/cad.py` | 固定 alpha 的对比式上下文感知解码。 |
| `adacad` | `methods/adacad.py` | 使用 JSD 自适应调整 token-level alpha。 |
| `cocoa` | `methods/cocoa.py` | CoCoA 风格的 logit 融合与扰动项。 |
| `coiecd` | `methods/coiecd.py` | 基于熵约束融合 context/prior logits。 |
| `simple_interp` | `methods/simple_interp.py` | 固定 alpha 的 context/prior logits 插值。 |
| `arr` | `methods/arr.py` | 基于置信度门控和 JSD 冲突强度的自适应机制路由。 |

原始 baseline 代码保留在 `CAD/`、`AdaCAD/`、`CoCoA/` 和 `COIECD/` 中，用于复现实验。

## 📦 数据

所有统一数据集都使用 **dual-process JSONL** 格式。每个样本由两行组成，并共享同一个 `input_index`：

```jsonl
{"input_index": 0, "assigned_process": 0, "context_string": "<context+question>", "assigned_weight": 2, "gold_answers": "answer", "benchmark": "nq", "task_type": "qa"}
{"input_index": 0, "assigned_process": 1, "context_string": "<question_only>", "assigned_weight": -1, "gold_answers": "answer", "benchmark": "nq", "task_type": "qa"}
```

- `assigned_process=0`：包含证据和问题的上下文 prompt。
- `assigned_process=1`：只包含问题的 prior prompt。

已包含的 QA 数据：

```text
data/Trad_QA/hotpotqa.jsonl
data/Trad_QA/nq.jsonl
data/Trad_QA/tabmwp.jsonl
data/Trad_QA/triviaqa.jsonl
```

已包含的 tristate 数据：

```text
data/TriState/{MODEL_SHORT}/C_right_P_wrong.jsonl
data/TriState/{MODEL_SHORT}/C_right_P_right.jsonl
data/TriState/{MODEL_SHORT}/C_wrong_P_right.jsonl
```

当前 `MODEL_SHORT` 目录覆盖 Llama-2、Llama-3、Mistral 和 Qwen2.5 的多个模型版本。

## 🚀 推理

推理分为两条路径：

- **统一单卡方法**：通过 `methods/inference.py` 运行。
- **原始双卡 baseline**：通过各 baseline 子目录中的 `run_group_decode_fileio.sh` 运行。

### 🖥️ 单卡推理

支持的方法：

```text
greedy, greedy_no_ctx, cad, adacad, cocoa, coiecd, simple_interp, arr
```

```bash
# Greedy baseline
python -m methods.inference --method greedy \
    --model /data/models/Meta-Llama-3-8B \
    --input_path data/Trad_QA/nq.jsonl \
    --output_path results/greedy/Meta-Llama-3-8B/nq.jsonl \
    --max_new_tokens 32

# COIECD
python -m methods.inference --method coiecd \
    --model /data/models/Meta-Llama-3-8B \
    --input_path data/Trad_QA/nq.jsonl \
    --output_path results/coiecd/Meta-Llama-3-8B/nq.jsonl \
    --max_new_tokens 32 \
    --alpha 1.0 \
    --threshold_ratio 4

# Simple interpolation
python -m methods.inference --method simple_interp \
    --model /data/models/Meta-Llama-3-8B \
    --input_path data/Trad_QA/nq.jsonl \
    --output_path results/simple_interp/Meta-Llama-3-8B/nq.jsonl \
    --max_new_tokens 32 \
    --interp_alpha 0.75

# ARR
python -m methods.inference --method arr \
    --model /data/models/Meta-Llama-3-8B \
    --input_path data/Trad_QA/nq.jsonl \
    --output_path results/arr/Meta-Llama-3-8B/nq.jsonl \
    --max_new_tokens 32
```

关键参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--method` | 解码方法名称。 | required |
| `--model` | HuggingFace 模型名或本地模型路径。 | required |
| `--input_path` | dual-process JSONL 输入文件。 | required |
| `--output_path` | 预测结果输出路径。 | required |
| `--max_new_tokens` | 最大生成 token 数。 | `32` |
| `--max_ctx_len` | 输入截断长度。 | `4064` |
| `--seed` | 随机种子。 | `42` |
| `--dtype` | 模型精度。 | `float16` |
| `--alpha` | CAD/COIECD 的 alpha 参数。 | `1.0` |
| `--threshold_ratio` | COIECD 熵阈值比例。 | `4` |
| `--warmup_beta` | AdaCAD 最小 alpha。 | `0.0` |
| `--interp_alpha` | SimpleInterp 的 context 权重。 | `0.75` |
| `--cocoa_global_alpha` | CoCoA 全局混合 alpha。 | `0.5` |
| `--cocoa_gamma` | CoCoA gamma 权重。 | `1.0` |
| `--cocoa_lambda_pm` | CoCoA 扰动权重。 | `100.0` |

`methods/inference.py` 支持断点续跑：如果 `output_path` 中已经包含某些 `input_index` 的结果，这些样本会被自动跳过。

### 🧪 批量推理脚本

使用统一单卡方法运行 QA benchmarks：

```bash
MODEL_NAME=/data/models/Meta-Llama-3-8B \
METHODS="arr" \
BENCHMARKS="nq tabmwp triviaqa hotpotqa" \
GPU=0 \
OUTPUT_ROOT=results_new \
OVERWRITE=0 \
bash inference_qa_self.sh
```

使用原始双卡 baseline 运行 QA benchmarks：

```bash
MODEL_NAME=/data/models/Qwen2.5-7B \
DEVICE="0,1" \
METHODS="cad adacad cocoa" \
bash inference_qa_origin.sh
```

运行 tristate benchmarks：

```bash
# 直接使用仓库内置数据
python -m methods.inference --method arr \
    --model /data/models/Meta-Llama-3-8B \
    --input_path data/TriState/Meta-Llama-3-8B/C_right_P_wrong.jsonl \
    --output_path results/arr/Meta-Llama-3-8B/C_right_P_wrong.jsonl \
    --max_new_tokens 32

# 单卡批量脚本
MODEL_SHORT=Qwen2.5-7B \
METHODS="greedy greedy_no_ctx simple_interp_0.25 simple_interp_0.5 simple_interp_0.75 coiecd arr" \
bash inference_tristate_self.sh

# 双卡原始 baseline
MODEL_SHORT=Meta-Llama-3-8B \
DEVICE="0,1" \
METHODS="cad_0.25 cad_0.5 cad_0.75 cad adacad cocoa" \
bash inference_tristate_origin.sh
```

批量脚本包含过长样本过滤和已完成输出检测。

### 🧱 原始 Baseline 推理

原始 CAD/AdaCAD/CoCoA 实现使用双进程推理，需要两张 GPU：

```bash
export MODEL_NAME=/data/models/Meta-Llama-3-8B

# CAD
cd CAD && bash run_group_decode_fileio.sh \
    42 "0,1" "fin|../data/Trad_QA/nq.jsonl" \
    4096 4064 32 0.0 \
    ../results/cad/Meta-Llama-3-8B/nq.jsonl

# AdaCAD
cd AdaCAD && bash run_group_decode_fileio.sh \
    42 "0,1" "fin|../data/Trad_QA/nq.jsonl" \
    4096 4064 32 0.0 \
    2 no 0 ../results/adacad/Meta-Llama-3-8B/nq.jsonl

# CoCoA
cd CoCoA && bash run_group_decode_fileio.sh \
    42 "0,1" "fin|../data/Trad_QA/nq.jsonl" \
    4096 4064 32 0.0 \
    2 no 0 ../results/cocoa/Meta-Llama-3-8B/nq.jsonl
```

## 📊 评测

使用 `evaluation/unified_eval.py` 进行 QA 评测：

```bash
python -m evaluation.unified_eval \
    --task_type qa \
    --gold_path data/Trad_QA/nq.jsonl \
    --pred_path results/greedy/Meta-Llama-3-8B/nq.jsonl \
    --metrics em,f1,substring_em
```

QA 指标：

| 指标 | 说明 |
|------|------|
| `em` | 使用 SQuAD 风格归一化的 Exact Match。 |
| `em_test` | 保留标点的 Exact Match。 |
| `substring_em` | gold answer 是否出现在 prediction 中。 |
| `f1` | token 级 F1。 |

使用 `--output_path` 保存评测结果：

```bash
python -m evaluation.unified_eval \
    --task_type qa \
    --gold_path data/Trad_QA/nq.jsonl \
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

如果本仓库对你的研究有帮助，请引用我们的论文：

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
