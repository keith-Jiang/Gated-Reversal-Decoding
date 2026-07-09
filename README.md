From Context-Aware to Conflict-Aware: Generalizing Contrastive Decoding for Knowledge Conflict in LLMs

[English](./README.md) | [中文](./README_zh.md)

This repository contains the code and data utilities for **From Context-Aware to Conflict-Aware: Generalizing Contrastive Decoding for Knowledge Conflict in LLMs**. It provides unified single-GPU decoding implementations, wrappers for original dual-GPU baselines, QA/tristate benchmark data, and evaluation scripts.

## 📚 Table of Contents

- [📰 News](#-news)
- [⚙️ Environment](#️-environment)
- [🧭 Project Structure](#-project-structure)
- [🧩 Implemented Modules](#-implemented-modules)
- [📦 Data](#-data)
- [🚀 Inference](#-inference)
- [📊 Evaluation](#-evaluation)
- [🙏 Acknowledgements](#-acknowledgements)
- [📝 Citation](#-citation)

## 📰 News

- **2026.06.09**: Our paper is available on arXiv: [From Context-Aware to Conflict-Aware: Generalizing Contrastive Decoding for Knowledge Conflict in LLMs](https://arxiv.org/abs/2606.10298).

## ⚙️ Environment

```bash
conda create -n conflict python==3.10 -y
conda activate conflict
pip install -r requirements.txt
```

## 🧭 Project Structure

```text
.
├── methods/                         # Unified single-GPU decoding methods
│   ├── inference.py                 # Single-GPU inference entry point
│   ├── base.py                      # Decoding method interface
│   ├── greedy.py                    # Greedy decoding with context
│   ├── cad.py                       # CAD / Context-Aware Decoding
│   ├── adacad.py                    # AdaCAD / adaptive CAD
│   ├── cocoa.py                     # CoCoA decoding
│   ├── coiecd.py                    # COIECD entropy-constrained decoding
│   ├── simple_interp.py             # Fixed-alpha interpolation
│   └── arr.py                       # ARR: Adaptive Regime Routing
├── evaluation/                      # Unified evaluation framework
│   ├── unified_eval.py              # Evaluation entry point
│   ├── utils.py                     # Data loading and normalization
│   └── metrics/
│       └── qa_metrics.py            # EM, substring EM, F1
├── data/                            # Unified benchmark data
│   ├── Trad_QA/                     # Traditional QA benchmarks
│   │   ├── hotpotqa.jsonl
│   │   ├── nq.jsonl
│   │   ├── tabmwp.jsonl
│   │   └── triviaqa.jsonl
│   └── TriState/                    # Model-specific tristate benchmarks
│       └── {MODEL_SHORT}/
│           ├── C_right_P_wrong.jsonl
│           ├── C_right_P_right.jsonl
│           └── C_wrong_P_right.jsonl
├── CAD/                             # Original CAD baseline
├── AdaCAD/                          # Original AdaCAD baseline
├── CoCoA/                           # Original CoCoA baseline
├── COIECD/                          # Original COIECD resources
├── inference_qa_self.sh             # Batch QA inference, single-GPU
├── inference_qa_origin.sh           # Batch QA inference, dual-GPU baselines
├── inference_tristate_self.sh       # Batch tristate inference, single-GPU
├── inference_tristate_origin.sh     # Batch tristate inference, dual-GPU baselines
└── requirements.txt
```

## 🧩 Implemented Modules

The unified single-GPU path is implemented in `methods/`. All methods consume paired context/prior prompts from the same dual-process JSONL format.

| Method | File | Description |
|--------|------|-------------|
| `greedy` | `methods/greedy.py`, `methods/inference.py` | Greedy decoding with the context prompt. |
| `greedy_no_ctx` | `methods/inference.py` | Greedy decoding with the prior-only prompt. |
| `cad` | `methods/cad.py` | Fixed-alpha contrastive context-aware decoding. |
| `adacad` | `methods/adacad.py` | Adaptive CAD with JSD-based token-level alpha. |
| `cocoa` | `methods/cocoa.py` | CoCoA-style logit mixing with perturbation terms. |
| `coiecd` | `methods/coiecd.py` | Entropy-constrained decoding for context/prior logits. |
| `simple_interp` | `methods/simple_interp.py` | Fixed-alpha interpolation between context and prior logits. |
| `arr` | `methods/arr.py` | Adaptive Regime Routing with a confidence gate and JSD conflict strength. |

Original baseline implementations are preserved under `CAD/`, `AdaCAD/`, `CoCoA/`, and `COIECD/` for reproducibility.

## 📦 Data

All unified benchmarks use a **dual-process JSONL** format. Each example has two lines with the same `input_index`:

```jsonl
{"input_index": 0, "assigned_process": 0, "context_string": "<context+question>", "assigned_weight": 2, "gold_answers": "answer", "benchmark": "nq", "task_type": "qa"}
{"input_index": 0, "assigned_process": 1, "context_string": "<question_only>", "assigned_weight": -1, "gold_answers": "answer", "benchmark": "nq", "task_type": "qa"}
```

- `assigned_process=0`: context prompt, containing evidence plus question.
- `assigned_process=1`: prior prompt, containing question only.

Included QA datasets:

```text
data/Trad_QA/hotpotqa.jsonl
data/Trad_QA/nq.jsonl
data/Trad_QA/tabmwp.jsonl
data/Trad_QA/triviaqa.jsonl
```

Included tristate datasets:

```text
data/TriState/{MODEL_SHORT}/C_right_P_wrong.jsonl
data/TriState/{MODEL_SHORT}/C_right_P_right.jsonl
data/TriState/{MODEL_SHORT}/C_wrong_P_right.jsonl
```

The available `MODEL_SHORT` directories currently cover Llama-2, Llama-3, Mistral, and Qwen2.5 variants.

## 🚀 Inference

Inference is split into two paths:

- **Unified single-GPU methods** via `methods/inference.py`.
- **Original dual-GPU baselines** via each baseline repository's `run_group_decode_fileio.sh`.

### 🖥️ Single-GPU Inference

Supported methods:

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

Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--method` | Decoding method name. | required |
| `--model` | HuggingFace model ID or local model path. | required |
| `--input_path` | Dual-process JSONL input file. | required |
| `--output_path` | Prediction output path. | required |
| `--max_new_tokens` | Maximum number of generated tokens. | `32` |
| `--max_ctx_len` | Maximum input length after truncation. | `4064` |
| `--seed` | Random seed. | `42` |
| `--dtype` | Model precision. | `float16` |
| `--alpha` | CAD/COIECD alpha. | `1.0` |
| `--threshold_ratio` | COIECD entropy threshold ratio. | `4` |
| `--warmup_beta` | AdaCAD minimum alpha. | `0.0` |
| `--interp_alpha` | SimpleInterp context weight. | `0.75` |
| `--cocoa_global_alpha` | CoCoA global mixing alpha. | `0.5` |
| `--cocoa_gamma` | CoCoA gamma weight. | `1.0` |
| `--cocoa_lambda_pm` | CoCoA perturbation weight. | `100.0` |

`methods/inference.py` supports resumption: if `output_path` already contains completed `input_index` entries, they will be skipped.

### 🧪 Batch Inference Scripts

Run QA benchmarks with unified single-GPU methods:

```bash
MODEL_NAME=/data/models/Meta-Llama-3-8B \
METHODS="arr" \
BENCHMARKS="nq tabmwp triviaqa hotpotqa" \
GPU=0 \
OUTPUT_ROOT=results_new \
OVERWRITE=0 \
bash inference_qa_self.sh
```

Run QA benchmarks with original dual-GPU baselines:

```bash
MODEL_NAME=/data/models/Qwen2.5-7B \
DEVICE="0,1" \
METHODS="cad adacad cocoa" \
bash inference_qa_origin.sh
```

Run tristate benchmarks:

```bash
# Directly use the included repository data
python -m methods.inference --method arr \
    --model /data/models/Meta-Llama-3-8B \
    --input_path data/TriState/Meta-Llama-3-8B/C_right_P_wrong.jsonl \
    --output_path results/arr/Meta-Llama-3-8B/C_right_P_wrong.jsonl \
    --max_new_tokens 32

# Single-GPU batch script
MODEL_SHORT=Qwen2.5-7B \
METHODS="greedy greedy_no_ctx simple_interp_0.25 simple_interp_0.5 simple_interp_0.75 coiecd arr" \
bash inference_tristate_self.sh

# Dual-GPU original baselines
MODEL_SHORT=Meta-Llama-3-8B \
DEVICE="0,1" \
METHODS="cad_0.25 cad_0.5 cad_0.75 cad adacad cocoa" \
bash inference_tristate_origin.sh
```

The batch scripts include input filtering for over-length examples and checkpoint detection for completed outputs.

### 🧱 Original Baseline Inference

The original CAD/AdaCAD/CoCoA implementations use dual-process inference and require two GPUs:

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

## 📊 Evaluation

Use `evaluation/unified_eval.py` for QA evaluation:

```bash
python -m evaluation.unified_eval \
    --task_type qa \
    --gold_path data/Trad_QA/nq.jsonl \
    --pred_path results/greedy/Meta-Llama-3-8B/nq.jsonl \
    --metrics em,f1,substring_em
```

QA metrics:

| Metric | Description |
|--------|-------------|
| `em` | Exact Match with SQuAD-style normalization. |
| `substring_em` | Whether a gold answer appears in the prediction. |
| `f1` | Token-level F1 score. |

Save evaluation output with `--output_path`:

```bash
python -m evaluation.unified_eval \
    --task_type qa \
    --gold_path data/Trad_QA/nq.jsonl \
    --pred_path results/greedy/Meta-Llama-3-8B/nq.jsonl \
    --metrics em,f1,substring_em \
    --output_path results/greedy/Meta-Llama-3-8B/nq_eval.json
```

Example output:

```json
{
  "em": 0.4523,
  "f1": 0.5812,
  "substring_em": 0.6134,
  "total": 4000,
  "missing": 0
}
```

## 🙏 Acknowledgements

This work was supported by Alibaba Group through Alibaba Research Intern Program.

## 📝 Citation

If you find this repository useful, please cite our paper:

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
