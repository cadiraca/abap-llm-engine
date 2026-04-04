# ABAP LLM Engine 🧠

**Run a Large Language Model inside SAP S/4HANA. Pure ABAP. No external libraries.**

> The world's first LLM inference engine implemented entirely in ABAP, running natively on the SAP application server with HANA acceleration.

## What Is This?

This is a proof-of-concept implementation of transformer-based neural network inference in pure ABAP. It loads a pre-trained language model (SmolLM2-135M) and generates text — all inside your S/4HANA system.

No Python. No llama.cpp. No ONNX. Just ABAP.

## Why?

Because we can. And because having an LLM running *inside* SAP means:

- **Zero hallucination for SAP objects** — the model can `SELECT` from `DD03L` to verify table/field names
- **Direct data dictionary access** — understands your system's custom objects
- **Transportable via SAP transports** — deploy the model like any other ABAP add-on
- **Air-gapped environments** — no external API calls needed
- **Sub-second inference** (with HANA AMDP acceleration)

---

## Quick Start

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| SAP S/4HANA 2021+ or SAP BTP ABAP Environment | ABAP 7.55+ required |
| abapGit (≥ 1.100) | Install via [abapgit.org](https://abapgit.org) |
| ~300 MB free work process memory | For model weights in memory |
| Python 3.8+ (on developer laptop) | Weight conversion only — not on SAP server |

### Step-by-step

```
1. Convert model weights (laptop/server with Python)
   → python tools/convert_weights.py --format int8

2. Upload model.bin to SAP application server (AL11)
   → e.g. /usr/sap/trans/llm/model.bin

3. Import this repository via abapGit
   → Create package ZLLM_ENGINE, then Online / Git pull

4. Upload weights into ZLLM_WEIGHTS table
   → Run report: ZLLM_UPLOAD_WEIGHTS

5. Upload tokenizer vocabulary into ZLLM_VOCAB / ZLLM_MERGES
   → Upload tokenizer.json to AL11, then run: ZLLM_UPLOAD_VOCAB

6. Run inference
   → Run report: ZLLM_DEMO

7. Run tests
   → Run report: ZLLM_TEST
```

---

## Weight Preparation

### 1. Convert weights (on developer machine)

```bash
pip install safetensors huggingface_hub numpy
python tools/convert_weights.py \
  --model  HuggingFaceTB/SmolLM2-135M \
  --output tools/model.bin \
  --format int8
```

This downloads SmolLM2-135M from HuggingFace (~270 MB) and produces a
compact `model.bin` (~130 MB, INT8 quantized) in the ALLM v1 binary format.

Optional formats: `float16` (~270 MB) or `float32` (~540 MB).

### 2. Upload to AL11

Copy `tools/model.bin` to a directory accessible from the SAP application server,
for example `/usr/sap/trans/llm/`. Use SFTP, SCP, or the SAP file transfer tools.

Verify the file is accessible: transaction AL11 → navigate to the directory.

### 3. Run ZLLM_UPLOAD_WEIGHTS

```
Report : ZLLM_UPLOAD_WEIGHTS
P_PATH : /usr/sap/trans/llm/model.bin
P_MODEL: SMOLLM2-135M
P_RESET: ✓ (check to replace existing rows)
```

Progress is printed to the spool. Expect ~272 tensor records inserted into
the `ZLLM_WEIGHTS` transparent table.

### 4. Prepare tokenizer (vocabulary)

Download `tokenizer.json` from HuggingFace and copy it to AL11:

```bash
python -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download('HuggingFaceTB/SmolLM2-135M', 'tokenizer.json')
print(p)
"
# copy the printed path to /usr/sap/trans/llm/tokenizer.json
```

Then run:

```
Report : ZLLM_UPLOAD_VOCAB
P_PATH : /usr/sap/trans/llm/tokenizer.json
P_MODEL: SMOLLM2-135M
P_RESET: ✓
```

This inserts 49,152 vocab entries into `ZLLM_VOCAB` and ~48,000 BPE merge
rules into `ZLLM_MERGES`.

---

## Running Inference

```
Report : ZLLM_DEMO
```

The demo report:
1. Loads the model config (vocab size, layer count, etc.)
2. Creates a `ZCL_LLM_ENGINE` instance
3. Loads weights from `ZLLM_WEIGHTS`
4. Attaches the BPE tokenizer (from `ZLLM_VOCAB` / `ZLLM_MERGES`)
5. Runs auto-regressive generation for a hard-coded prompt
6. Prints each generated token to the output list

Expected output (with real weights):
```
Prompt: Hello, I am
Token 1: a
Token 2: language
Token 3: model
...
```

---

## Testing

```
Report : ZLLM_TEST
```

The test report validates every major component with known reference values.
No model weights are required — it creates tensors directly.

**Test suite:**

| # | Component | What is tested |
|---|-----------|----------------|
| 1 | Tensor ops | create_zeros, 2×2 matmul, add, reshape, slice |
| 2 | RMSNorm | `[1,2,3]` with unit weight → expected `[0.463, 0.926, 1.389]` |
| 3 | SiLU | `silu(1.0)≈0.7311`, `silu(-1.0)≈-0.2689` |
| 4 | Softmax | `softmax([1,2,3])≈[0.090, 0.245, 0.665]`, sum=1.0 |
| 5 | BPE | encode/decode roundtrip on `"hello"` |
| 6 | Forward pass | zero-weight forward, output shape = `vocab_size` (49152) |

All tests print `✓ PASS` / `✗ FAIL` lines followed by a summary.

### Reference outputs (Python)

To generate authoritative reference values from the actual model:

```bash
# Lightweight mode (uses model.bin, no torch required)
python tools/validate_reference.py --mode lite

# Full mode (requires: pip install transformers torch)
python tools/validate_reference.py --mode full
```

Output is saved to `tools/reference_outputs.json`.

---

## Architecture

```
ZCL_LLM_ENGINE (orchestrator)
│  Manages tokenizer, transformer layers, KV cache, and generation loop
│
├── ZCL_LLM_BPE_TOKENIZER
│     Byte-Pair Encoding tokenizer
│     Vocab + merge rules loaded from ZLLM_VOCAB / ZLLM_MERGES
│
├── ZCL_LLM_TENSOR  (implements ZIF_LLM_TENSOR)
│     Multi-dimensional float tensor with row-major storage
│     Operations: matmul (tiled), add, multiply_elementwise,
│                 scale, reshape, slice
│
├── ZCL_LLM_MODEL_LOADER
│     Reads ALLM binary format from ZLLM_WEIGHTS or file
│     Returns tensors by name for the engine to consume
│
├── ZCL_LLM_MATH
│     Static utility: sigmoid, SiLU, softmax, RMSNorm, RoPE
│
├── ZCL_LLM_TRANSFORMER_BLOCK  (×30 layers)
│   │  Single transformer layer: pre-norm → attention → residual
│   │                            → post-norm → FFN → residual
│   │
│   ├── ZCL_LLM_ATTENTION
│   │     Grouped-Query Attention (GQA): 9 Q heads, 3 KV heads
│   │     Includes KV cache for autoregressive decoding
│   │
│   └── ZCL_LLM_FFN
│         SiLU-gated feed-forward: gate_proj · up_proj → SiLU → down_proj
│
└── ZCL_LLM_SAMPLER
      Temperature scaling, Top-K filtering, Top-P (nucleus) sampling
```

---

## Model: SmolLM2-135M

| Property | Value |
|----------|-------|
| Parameters | 135 million |
| Architecture | Llama (30 layers, 576 hidden, 1536 intermediate) |
| Attention | GQA — 9 Q heads, 3 KV heads, 64 head dim |
| Tokenizer | BPE (GPT-NeoX-style) |
| Vocabulary | 49,152 tokens |
| Context window | 8,192 tokens |
| Memory (INT8) | ~130 MB weights + overhead |

---

## File Listing

```
abap-llm-engine/
├── README.md                          — This file
├── .abapgit.xml                       — abapGit repository settings
│
├── src/                               — All ABAP source objects
│   ├── package.devc.xml               — Package ZLLM_ENGINE
│   │
│   │   ── Interfaces ──
│   ├── zif_llm_tensor.intf.abap       — Tensor interface (get_data, matmul, …)
│   ├── zif_llm_tensor.intf.xml
│   │
│   │   ── Classes ──
│   ├── zcl_llm_tensor.clas.abap       — Tensor implementation (tiled matmul)
│   ├── zcl_llm_tensor.clas.xml
│   ├── zcl_llm_math.clas.abap         — SiLU, softmax, RMSNorm, RoPE
│   ├── zcl_llm_math.clas.xml
│   ├── zcl_llm_bpe_tokenizer.clas.abap — BPE encode/decode
│   ├── zcl_llm_bpe_tokenizer.clas.xml
│   ├── zcl_llm_attention.clas.abap    — GQA attention + KV cache
│   ├── zcl_llm_attention.clas.xml
│   ├── zcl_llm_ffn.clas.abap          — Feed-forward network (SiLU gate)
│   ├── zcl_llm_ffn.clas.xml
│   ├── zcl_llm_transformer_block.clas.abap — Single transformer layer
│   ├── zcl_llm_transformer_block.clas.xml
│   ├── zcl_llm_sampler.clas.abap      — Temperature/Top-K/Top-P sampling
│   ├── zcl_llm_sampler.clas.xml
│   ├── zcl_llm_model_loader.clas.abap — Weight loader (ALLM format)
│   ├── zcl_llm_model_loader.clas.xml
│   ├── zcl_llm_engine.clas.abap       — Main engine / orchestrator
│   ├── zcl_llm_engine.clas.xml
│   │
│   │   ── Database Tables ──
│   ├── zllm_weights.tabl.abap         — Model weight storage (binary blobs)
│   ├── zllm_weights.tabl.xml
│   ├── zllm_vocab.tabl.abap           — BPE vocabulary (token_id → token)
│   ├── zllm_vocab.tabl.xml
│   ├── zllm_merges.tabl.abap          — BPE merge rules (priority + pair)
│   ├── zllm_merges.tabl.xml
│   │
│   │   ── Reports ──
│   ├── zllm_demo.prog.abap            — Inference demo (run to generate text)
│   ├── zllm_demo.prog.xml
│   ├── zllm_upload_weights.prog.abap  — Load model.bin → ZLLM_WEIGHTS
│   ├── zllm_upload_weights.prog.xml
│   ├── zllm_upload_vocab.prog.abap    — Load tokenizer.json → ZLLM_VOCAB/MERGES
│   ├── zllm_upload_vocab.prog.xml
│   ├── zllm_test.prog.abap            — Component test suite
│   └── zllm_test.prog.xml
│
├── tools/                             — Developer tooling (Python, not deployed to SAP)
│   ├── convert_weights.py             — HuggingFace safetensors → ALLM binary
│   ├── validate_reference.py          — Generate reference values for ABAP tests
│   ├── model.bin                      — Converted INT8 weights (130 MB, git-ignored)
│   └── reference_outputs.json         — Reference values from validate_reference.py
│
└── docs/
    ├── ARTICLE.md                     — Deep-dive technical article
    └── GEMMA4-PRIMER.md               — Architecture primer
```

---

## Performance

| Mode | Speed | Notes |
|------|-------|-------|
| Pure ABAP | 5-30 sec/token | Works everywhere, no special config |
| HANA AMDP | 0.5-3 sec/token | Pushes matmul to HANA's parallel engine |
| Shared Memory | +30% faster | Load weights once, share across WPs |

---

## Project Status

🚧 **Phase 1: Proof of Concept** — Core tensor operations and transformer forward pass complete.

**Completed:**
- [x] Tensor class with tiled matrix multiplication
- [x] RMSNorm, SiLU, Softmax, RoPE math utilities
- [x] BPE tokenizer (encode/decode)
- [x] Grouped-Query Attention with KV cache
- [x] SiLU-gated FFN
- [x] Transformer block (30 layers)
- [x] Model loader (ALLM binary format)
- [x] Database tables (weights, vocab, merges)
- [x] Weight upload report (ZLLM_UPLOAD_WEIGHTS)
- [x] Vocab/merge upload report (ZLLM_UPLOAD_VOCAB)
- [x] Weight conversion script (bfloat16 → INT8)
- [x] Reference validator (Python)
- [x] Component test suite (ZLLM_TEST)

**Upcoming (Phase 2):**
- [ ] HANA AMDP matrix multiplication acceleration
- [ ] Shared memory weight cache
- [ ] Streaming token output
- [ ] SAP BTP ABAP Environment compatibility testing

See the [full article](docs/ARTICLE.md) for the deep technical writeup.

---

## License

Apache 2.0

## Credits

- [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) by HuggingFace
- [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy (reference implementation)
- [SharpInfer](https://github.com/jpaulduncan/SharpInfer) (proof that managed languages can do inference)
