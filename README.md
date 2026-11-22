# Mitigating Role Confusion: Lightweight Middleware Firewall for Detecting Chain-of-Thought Forgery

A three-layer inference middleware pipeline designed to detect and block **Chain-of-Thought (CoT) Forgery** and role confusion inside `<think>...</think>` reasoning blocks without modifying target model weights or deploying a second large judge model.

## Overview

Large Reasoning Models (LRMs) such as [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) generate intermediate deliberation inside `<think>...</think>` before emitting a final response. An adversary can submit a benign-looking surface prompt while embedding spoofed authority cues, persona pivots, or policy-violating reasoning inside `<think>`. Input-only guardrails miss hidden reasoning shifts, while auxiliary 7B–70B safety judges introduce heavy latency and VRAM overhead.

This project implements a lightweight inference-time firewall around reasoning models:
- **Layer 1 — Canonicalization & Hybrid Input Guard:** Applies Unicode `NFKC` normalization, strips invisible/directional characters, decodes high-entropy Base64 payloads (`H >= 3.5`), and evaluates prompts using [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) + [FAISS](https://github.com/facebookresearch/faiss) (`IndexFlatIP`) alongside a fine-tuned [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) (`distilbert-base-uncased`) classifier.
- **Layer 2 — Boundary CoT Alignment Verifier (`</think>`):** Intercepts the reasoning trace `r` at the `</think>` boundary, canonicalizes it (`r' = C(r)`), and scores it with a fine-tuned [DeBERTa-v3-small](https://huggingface.co/microsoft/deberta-v3-small) (`microsoft/deberta-v3-small`) model before answer decoding conditions on forged reasoning.
- **Layer 3 — Post-Execution Output Guard:** Scans final responses `y` for leaked `<think>`/`</think>` tags, spoofed override tokens, and high-risk instructional signatures prior to client delivery.

---

## Architecture & Decision Thresholds

- **Layer 1 Early Exit:** Rejects requests prior to reasoning generation when semantic similarity `s_sim >= 0.85` or input classifier probability `p_in >= 0.50`.
- **Layer 2 Boundary Abort:** Aborts at `</think>` when CoT forgery probability `p_cot >= 0.50`, preventing answer generation from building on a compromised trace.
- **Layer 3 Output Scan:** Performs deterministic post-generation checks on `y` before returning a safe response.

---

## Datasets & Evaluation Splits

- **Layer 1 Input Split:** Built from prompt-injection and jailbreak datasets alongside benign and over-refusal edge cases (`1,419` train / `304` validation / `304` test; 710 adversarial and 709 benign in training).
- **Layer 2 CoT Trace Corpus:** `1,560` reasoning traces (`780` adversarial / `780` benign) covering four synthetic CoT-forgery subtypes (*Authority Spoofing*, *Context-Diluted Hijack*, *Destyled Neutral Forgery*, and *Stylistic CoT Imitation*) plus `6x`-oversampled live benign `DeepSeek-R1-Distill-Qwen-7B` prefixes paired with forged endings to reduce false positives on natural reasoning fillers.
- **End-to-End CoT Forgery Suite:** `200` samples (`100` adversarial across the four 25-sample subtypes / `100` benign hard negatives).
- **Live LRM Benchmark:** `80` live prompts (`40` adversarial / `40` benign) evaluated on 4-bit NF4 `DeepSeek-R1-Distill-Qwen-7B`.

---

## Experimental Results (NVIDIA Tesla T4)

Evaluated on a single **NVIDIA Tesla T4 (16 GB VRAM)** using [PyTorch](https://pytorch.org/) (`2.14.1+cu130`), [Transformers](https://huggingface.co/docs/transformers) (`5.22.1`), and `FAISS` (`1.16.0`):

### 1. Component-Wise Ablation & Latency
- **Layer 1 Hybrid (Input Test Split, $N=304$):** `94.10%` accuracy, `92.41%` precision, `96.05%` recall, `94.19%` F1, `7.84%` FPR, `18.30 ms` mean latency (`23.86 ms` P95).
- **Layer 1 Only (E2E Suite, $N=200$):** `87.00%` accuracy, `95.12%` precision, `78.00%` recall, `85.71%` F1, `4.00%` FPR, `18.30 ms` mean latency (`23.86 ms` P95).
- **Full Firewall — Layer 1 + Layer 2 (E2E Suite, $N=200$):** **`98.00%` accuracy**, **`96.15%` precision**, **`100.00%` recall**, **`98.04%` F1**, **`4.00%` FPR**, **`25.85 ms` mean latency** (`38.22 ms` P95).

### 2. Subtype Detection Recovery (25 Samples Each)
- **Authority Spoofing:** `60.00%` (Layer 1 only) -> **`100.00%`** (Full Firewall, `+40.00%`).
- **Destyled Neutral Forgery:** `52.00%` (Layer 1 only) -> **`100.00%`** (Full Firewall, `+48.00%`).
- **Context-Diluted Hijack & Stylistic CoT Imitation:** `100.00%` (Layer 1 only) -> **`100.00%`** (Full Firewall).

### 3. Live 4-Bit `DeepSeek-R1-Distill-Qwen-7B` ($N=80$)
- **Unprotected Baseline:** `67.50%` Attack Success Rate (27/40), `0.00%` benign false blocks.
- **With CoT Firewall:** **`0.00%` Attack Success Rate** (0/40), **`5.00%` benign false blocks** (2/40), **`19.69 ms` mean live overhead**.

---

## Limitations & Threats to Validity

- **Synthetic template separability:** High standalone CoT classification metrics reflect structured forgery templates; LLM-paraphrased or adaptive reasoning hijacks require further evaluation.
- **Model-specific calibration:** `6x` oversampling on live benign `DeepSeek-R1-Distill-Qwen-7B` traces improves alignment on its reasoning style, but cross-model transfer to other LRMs remains future work.
- **Benign keyword sensitivity:** Benign prompts containing security-adjacent terms account for remaining false blocks (`4.00%` E2E / `5.00%` live).
- **Live sample size:** Live testing covers $N=80$ prompts (40 benign, 40 adversarial), where each false block equals `2.50%`.

---
## Quick Setup & Reproduction

1. **Open the Notebook in Colab:** Open [FYP_initial.ipynb](https://github.com/Zohaib3810/lrm-cot-firewall/blob/main/FYP_initial.ipynb) in Google Colab and attach a GPU runtime (evaluated on an NVIDIA Tesla T4).
2. **Mount Drive & Set Workspace Root:** Mount Google Drive and point `BASE_DIR` to `/content/drive/MyDrive/CoT_Firewall_Research` (which stores `data/raw`, `data/processed`, `data/vector_db`, `models`, `results`, and exported assets).
3. **Run Cells in Order:** Execute the environment check, canonicalization and helper utilities, dataset/training/evaluation cells, and latency/metric exporters—avoiding unnecessary full re-training if pretrained checkpoints and result CSVs/JSONs are already saved in Drive.

## Project & Citation

This repository accompanies the Final Year Project and research manuscript (`conference_101719.tex`, maintained in Overleaf) on detecting and mitigating **Chain-of-Thought (CoT) Forgery** in Large Reasoning Models via a three-layer middleware firewall.

## Repository & Workspace Layout

```text
.
├── FYP_initial.ipynb       # Dataset curation, training, evaluation, and export pipeline
├── conference_101719.tex   # IEEEtran research paper source
├── .gitignore              # Python / artifact ignore rules
├── LICENSE                 # MIT License
└── README.md
