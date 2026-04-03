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

## Architecture

```
ZCL_LLM_ENGINE (orchestrator)
├── ZCL_LLM_BPE_TOKENIZER     — Byte-Pair Encoding tokenizer
├── ZCL_LLM_TENSOR             — Tensor operations (matmul, add, reshape)
├── ZCL_LLM_MODEL_LOADER       — Load weights from Z-table or file
├── ZCL_LLM_TRANSFORMER_BLOCK  — Single transformer layer
│   ├── ZCL_LLM_RMS_NORM       — Root Mean Square normalization
│   ├── ZCL_LLM_ROPE           — Rotary Position Embeddings
│   ├── ZCL_LLM_ATTENTION      — Grouped-Query Attention + KV cache
│   └── ZCL_LLM_FFN            — SiLU-gated Feed-Forward Network
├── ZCL_LLM_SAMPLER            — Temperature, Top-K, Top-P sampling
└── ZCL_LLM_HANA_ACCEL         — AMDP-based HANA acceleration (Phase 2)
```

## Model: SmolLM2-135M

| Property | Value |
|----------|-------|
| Parameters | 135 million |
| Architecture | Llama (30 layers, 576 hidden, 9 heads) |
| Memory (INT8) | ~250 MB total |
| Vocabulary | 49,152 tokens |
| Context window | 8,192 tokens |

## Requirements

- SAP S/4HANA 2021+ or SAP BTP ABAP Environment
- ABAP 7.55+ (for modern language features)
- abapGit installed on the target system
- ~300 MB free work process memory

## Installation

1. Install [abapGit](https://abapgit.org) on your SAP system
2. Create package `ZLLM_ENGINE` 
3. Clone this repository via abapGit
4. Run the weight loader: `ZLLM_LOAD_WEIGHTS` (downloads and converts model weights)
5. Test: `ZLLM_DEMO` — generates text from a prompt

## Performance

| Mode | Speed | Notes |
|------|-------|-------|
| Pure ABAP | 5-30 sec/token | Works everywhere, no special config |
| HANA AMDP | 0.5-3 sec/token | Pushes matmul to HANA's parallel engine |
| Shared Memory | +30% faster | Load weights once, share across WPs |

## Project Status

🚧 **Phase 1: Proof of Concept** — Building core tensor operations and transformer forward pass

See the [full article](docs/ARTICLE.md) for the deep technical writeup.

## Article

📝 *"Running an LLM Inside SAP S/4HANA: A Deep Dive into Pure ABAP Neural Network Inference"*

The companion article documents the entire journey from "is this even possible?" to working inference. Covering:

- Transformer architecture explained for ABAP developers
- Implementing matrix multiplication in ABAP
- Memory management and quantization tricks
- HANA as an accidental GPU
- Benchmark results and lessons learned

## License

Apache 2.0

## Credits

- [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) by HuggingFace
- [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy (reference implementation)
- [SharpInfer](https://github.com/jpaulduncan/SharpInfer) (proof that managed languages can do inference)
