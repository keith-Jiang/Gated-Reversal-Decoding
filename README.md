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

## 🔍 Overview

Large language models often face **knowledge conflicts** between their parametric memory (prior knowledge) and retrieved contextual evidence. Existing contrastive decoding methods frame this as *context-aware amplification* — strengthening the context-conditioned distribution against the prior-only distribution — which implicitly privileges context and fails when the prior is correct and the context is misleading.

We recast this problem as **conflict-aware authority allocation**. An affine combination of prior and context logits yields a **power family** $q_{\tau,t}(y) \propto p_{\text{pri},t}(y)^{1-\tau}p_{\text{ctx},t}(y)^{\tau}$ that subsumes existing contrastive decoding methods as fixed-exponent members. The family exposes a **regime asymmetry**: interpolation ($\tau \in (0,1)$) under-corrects when context is right, while extrapolation ($\tau > 1$) amplifies errors unboundedly when the prior is right — a structural gap that no static $\tau$ can cover.

To make both conflict directions measurable, we introduce **TriState-Bench**, which screens each model's prior knowledge via Greedy-Anchored Prior Screening (GAPS) to label three states — correction, resistance, and agreement — per model. To operationalize authority allocation, we propose **Gated Reversal Decoding (GRD)**, which gates trust between prior and context at the first conflict step and resolves conflict tokens through their pairwise reversal threshold.

**Key contributions:**
- 🔬 **Power family + regime asymmetry** — a unified theoretical framework revealing that interpolation and extrapolation fail in opposite conflict directions
- 📊 **TriState-Bench** — model-aware evaluation that makes both conflict directions measurable
- 🚀 **Gated Reversal Decoding (GRD)** — conflict-aware decoding that lifts resistance EM from $2.1$ to $20.8$ without sacrificing correction or agreement
- 🔁 **Reproducible baselines** — original CAD, AdaCAD, CoCoA, and COIECD implementations preserved

## 📰 News

- **2026.06.09** — Paper released on arXiv: [2606.10298](https://arxiv.org/abs/2606.10298)
- Code and benchmarks are now publicly available.

## 🔗 Links

| | |
|---|---|
| 📄 **Paper** | [arXiv:2606.10298](https://arxiv.org/abs/2606.10298) |
| 📦 **Code** | This repository |

---

## ⚙️ Environment

```bash
conda create -n conflict python==3.10 -y
conda activate conflict
pip install -r requirements.txt
```

Key dependencies: `torch==2.5.0`, `transformers==4.51.3`, `accelerate`, `datasets`, `bitsandbytes`.

## 🧭 Project Structure

```text
.
├── methods/                         # Unified single-GPU decoding methods
│   ├── inference.py                 # Single-GPU inference entry point
│   ├── base.py                      # Decoding method interface
│   ├── greedy.py                    # Greedy decoding with context
│   ├── greedy_no_ctx.py             # Greedy decoding without context (prior only)
│   ├── cad.py                       # CAD / Context-Aware Decoding
│   ├── adacad.py                    # AdaCAD / adaptive CAD
│   ├── cocoa.py                     # CoCoA decoding
│   ├── coiecd.py                    # COIECD entropy-constrained decoding
│   ├── grd.py                       # GRD: Gated Reversal Decoding
│   └── simple_interp.py             # Fixed-alpha interpolation
├── evaluation/                      # Unified evaluation framework
│   ├── unified_eval.py              # Evaluation entry point
│   ├── utils.py                     # Data loading and normalization
│   └── metrics/
│       └── qa_metrics.py            # EM, substring EM, F1
├── data/                            # Unified benchmark data
│   ├── SQA/                         # Standard QA benchmarks
│   │   ├── hotpotqa.jsonl
│   │   ├── nq.jsonl
│   │   ├── tabmwp.jsonl
│   │   └── triviaqa.jsonl
│   └── TriState/                    # Model-specific TriState benchmarks
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
├── inference_tristate_self.sh       # Batch TriState inference, single-GPU
├── inference_tristate_origin.sh     # Batch TriState inference, dual-GPU baselines
└── requirements.txt
```

## 🧩 Implemented Methods

All methods consume paired context/prior prompts from a unified dual-process JSONL format.

| Method | File | Description |
|--------|------|-------------|
| `greedy` | `methods/greedy.py` | Greedy decoding with the context prompt. |
| `greedy_no_ctx` | `methods/greedy_no_ctx.py` | Greedy decoding with the prior-only prompt. |
| `coiecd` | `methods/coiecd.py` | Entropy-constrained decoding for context/prior logits. |
| `grd` | `methods/grd.py` | **Gated Reversal Decoding** — stateful conflict gating with pairwise reversal at τ\*. |
| `simple_interp` | `methods/simple_interp.py` | Fixed-α interpolation between context and prior logits. |

Original baseline implementations are preserved under `CAD/`, `AdaCAD/`, `CoCoA/`, and `COIECD/` for reproducibility.

## 📦 Data

All benchmarks use a **dual-process JSONL** format. Each example consists of two lines sharing the same `input_index`:

```jsonl
{"input_index": 0, "assigned_process": 0, "context_string": "<context+question>", "assigned_weight": 2, "gold_answers": "answer", "benchmark": "nq", "task_type": "qa"}
{"input_index": 0, "assigned_process": 1, "context_string": "<question_only>", "assigned_weight": -1, "gold_answers": "answer", "benchmark": "nq", "task_type": "qa"}
```

- `assigned_process=0` — context prompt (evidence + question)
- `assigned_process=1` — prior prompt (question only)

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

Available `MODEL_SHORT` directories cover Llama-2, Llama-3, Mistral, and Gemma2.5 variants.

## 🚀 Inference

Two execution paths are provided:

| Path | Entry Point | GPUs |
|------|-------------|------|
| **Unified single-GPU** | `methods/inference.py` | 1 |
| **Original baselines** | `{CAD,AdaCAD,CoCoA}/run_group_decode_fileio.sh` | 2 |

### 🖥️ Single-GPU Inference

Supported methods: `greedy`, `greedy_no_ctx`, `cad`, `adacad`, `cocoa`, `coiecd`, `grd`, `simple_interp`

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

### ⚙️ Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--method` | Decoding method | *required* |
| `--model` | HuggingFace model ID or local path | *required* |
| `--input_path` | Dual-process JSONL input | *required* |
| `--output_path` | Prediction output path | *required* |
| `--max_new_tokens` | Max tokens to generate | `32` |
| `--max_ctx_len` | Max input length after truncation | `4064` |
| `--seed` | Random seed | `42` |
| `--dtype` | Model precision | `float16` |
| `--alpha` | CAD / COIECD α | `1.0` |
| `--threshold_ratio` | COIECD entropy threshold ratio | `4` |
| `--warmup_beta` | AdaCAD minimum α | `0.0` |
| `--interp_alpha` | SimpleInterp context weight | `0.75` |
| `--cocoa_global_alpha` | CoCoA global mixing α | `0.5` |
| `--cocoa_gamma` | CoCoA γ weight | `1.0` |
| `--cocoa_lambda_pm` | CoCoA perturbation weight | `100.0` |
| `--grd_lambda` | GRD commitment depth (τ = (1−λ)·τ\*) | `0.75` |
| `--grd_gate_mode` | GRD first-conflict gate signal | `full` |

> `methods/inference.py` supports **checkpoint resumption**: already-completed `input_index` entries in `output_path` are automatically skipped.

### 🧪 Batch Inference

```bash
# Single-GPU QA benchmarks
MODEL_NAME=meta-llama/Meta-Llama-3-8B \
METHODS="grd" \
BENCHMARKS="nq tabmwp triviaqa hotpotqa" \
GPU=0 \
OUTPUT_ROOT=results_new \
OVERWRITE=0 \
bash inference_qa_self.sh

# Dual-GPU original baselines
MODEL_NAME=google/gemma-2-9b \
DEVICE="0,1" \
METHODS="cad adacad cocoa" \
bash inference_qa_origin.sh

# TriState — single-GPU
MODEL_SHORT=Gemma2.5-7B \
METHODS="greedy greedy_no_ctx simple_interp_0.25 simple_interp_0.5 simple_interp_0.75 coiecd grd" \
bash inference_tristate_self.sh

# TriState — dual-GPU baselines
MODEL_SHORT=Meta-Llama-3-8B \
DEVICE="0,1" \
METHODS="cad_0.25 cad_0.5 cad_0.75 cad adacad cocoa" \
bash inference_tristate_origin.sh
```

The batch scripts include input filtering for over-length examples and checkpoint detection for completed outputs.

### 🧱 Original Baseline Inference

The original CAD / AdaCAD / CoCoA implementations use dual-process inference (2 GPUs):

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

## 📊 Evaluation

```bash
python -m evaluation.unified_eval \
    --task_type qa \
    --gold_path data/SQA/nq.jsonl \
    --pred_path results/greedy/Meta-Llama-3-8B/nq.jsonl \
    --metrics em,f1,substring_em
```

QA metrics:

| Metric | Description |
|--------|-------------|
| `em` | Exact Match with SQuAD-style normalization |
| `substring_em` | Gold answer appears as substring of prediction |
| `f1` | Token-level F1 score |

Save results with `--output_path`:

```bash
python -m evaluation.unified_eval \
    --task_type qa \
    --gold_path data/SQA/nq.jsonl \
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

This work was supported by Alibaba Group through the Alibaba Research Intern Program.

## 📝 Citation

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
