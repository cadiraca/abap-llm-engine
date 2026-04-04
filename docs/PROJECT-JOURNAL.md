# ABAP LLM Engine — Project Journal

*Documenting every step: the What, Why, How, and When.*

---

## Project Overview

| Field | Value |
|-------|-------|
| **Name** | ABAP LLM Engine |
| **Goal** | Run LLM inference inside SAP S/4HANA using pure ABAP — no external libraries |
| **Target Model** | SmolLM2-135M (HuggingFace) |
| **Target System** | SAP S/4HANA 2025 (SHC, client 100, AWS) |
| **Author** | Carlos Diego Ramírez |
| **Started** | 2026-04-03 |
| **Status** | Phase 1 Complete — ready for deployment |
| **Repository** | github.com/cadiraca/abap-llm-engine |
| **CrownTrack** | cmnj43mm2002ro801txv9x04q |

---

## The Idea — What & Why

### What
Build a transformer neural network inference engine entirely in ABAP. Load a pre-trained 135-million parameter language model (SmolLM2-135M) into SAP S/4HANA and generate text from prompts — all inside the application server.

### Why
1. **Zero hallucination for SAP objects** — The model runs inside SAP, meaning it can `SELECT` from `DD03L` to verify table/field names exist before suggesting them. No guessing.
2. **Air-gapped environments** — Many SAP systems have no internet access. An embedded model needs no external API calls.
3. **Transportable** — Deploy the model like any other ABAP development object. Transport it between systems.
4. **Proof of concept** — If inference works in ABAP, it opens the door to HANA-accelerated ML directly in the application layer.
5. **Because we can** — Someone ran LLMs in SQL and Excel. ABAP is more capable than both.

### The Insight That Started It All
Carlos's observation: the real problem with ABAP + LLMs isn't model quality — it's **hallucinated object names**. LLMs suggest non-existent BAPIs, CDS views, and classes because they don't have ground truth. An LLM running inside SAP has direct access to the data dictionary.

---

## Architecture Decisions

### Decision 1: SmolLM2-135M as Target Model (2026-04-03)

**Why SmolLM2-135M?**
- Smallest model that produces coherent text
- 135M parameters → ~135 MB at INT8 quantization
- Fits in ABAP work process memory (2-4 GB limit)
- Llama architecture — well-documented, many reference implementations
- Weight tying (embedding = LM head) saves memory

**Why not Gemma 4?**
- Gemma 4 26B MoE: 9.6 GB — way too big for ABAP memory
- Gemma 4 E2B (2.3B): ~2-3 GB quantized — pushes limits, minutes per token
- For a PoC, the value is proving inference works, not model intelligence

**Architecture specs:**
| Property | Value |
|----------|-------|
| Parameters | 135M |
| Hidden size | 576 |
| FFN intermediate | 1,536 |
| Attention heads | 9 |
| KV heads | 3 (grouped-query attention) |
| Layers | 30 |
| Vocab size | 49,152 |
| Max context | 8,192 tokens |
| Activation | SiLU |
| Norm | RMSNorm |
| Position encoding | RoPE (θ=100,000) |

### Decision 2: Pure ABAP First, HANA Acceleration Second (2026-04-03)

The PoC value is in proving inference can run at all. Speed optimization via AMDP comes in Phase 2. Estimated performance:
- Pure ABAP (optimized): 5-30 seconds per token
- AMDP + HANA acceleration: 0.5-3 seconds per token
- Reference (Ollama/llama.cpp on CPU): ~60 tokens/sec

### Decision 3: Z-Table Weight Storage (2026-04-03)

Store model weights in a Z-table (ZLLM_WEIGHTS) with RAWSTRING field. Benefits:
- Transportable between SAP systems via TMS
- Queryable (check what's loaded)
- Alternative: AL11 file for faster initial loading

### Decision 4: INT8 Quantization (2026-04-03)

Quantize weights from bfloat16 → INT8 with per-channel scale factors. Reduces memory from 270 MB to ~135 MB while preserving quality for a PoC.

---

## Build Log

### Phase 1: Research (2026-04-03, ~16:00-17:00 UTC)

**What:** Full feasibility analysis of running LLM inference in ABAP.

**How:**
- Analyzed ABAP memory constraints (dialog WP: 2GB, background: 2-4GB)
- Estimated matrix multiplication performance in ABAP
- Researched precedents (llama2.c, SharpInfer C#, llama2.sql, llama-in-excel)
- Analyzed SmolLM2-135M architecture from HuggingFace config
- Explored HANA acceleration potential (PAL, AFL, vector ops, AMDP)

**Output:** `memory/research-abap-llm-inference.md` (comprehensive research doc)

**Finding:** 350 MB total memory fits comfortably in ABAP. The bottleneck is matmul speed, but HANA AMDP could serve as a "GPU replacement" for Phase 2.

### Phase 1: Core ABAP Classes (2026-04-03, ~22:00-23:00 UTC)

**What:** Built all 10 core ABAP classes implementing the transformer inference pipeline.

**How:** Translated the Llama architecture into an ABAP class hierarchy. Each mathematical operation maps to an ABAP method. Used internal tables of floats as tensor storage.

**Classes built (1,980 lines total):**

| Class | Lines | Purpose |
|-------|-------|---------|
| `ZIF_LLM_TENSOR` | 79 | Interface: tensor operations contract |
| `ZCL_LLM_TENSOR` | 299 | Tensor implementation: create, matmul, add, reshape, slice |
| `ZCL_LLM_MATH` | 224 | Math utilities: RMSNorm, softmax, SiLU, RoPE |
| `ZCL_LLM_BPE_TOKENIZER` | 212 | Byte-Pair Encoding tokenizer: encode/decode |
| `ZCL_LLM_ATTENTION` | 210 | Grouped-Query Attention with KV cache |
| `ZCL_LLM_FFN` | 102 | SiLU-gated Feed-Forward Network |
| `ZCL_LLM_TRANSFORMER_BLOCK` | 152 | Single transformer layer (norm → attn → norm → FFN) |
| `ZCL_LLM_SAMPLER` | 195 | Token sampling: temperature, top-K, top-P |
| `ZCL_LLM_ENGINE` | 374 | Main orchestrator: tokenize → prefill → generate |
| `ZLLM_DEMO` | 133 | Demo program: end-to-end text generation |

**Commit:** `3c95995` — pushed to GitHub

### Phase 1: Weight Converter (2026-04-03, ~16:30 UTC)

**What:** Python script to convert SmolLM2-135M weights from HuggingFace safetensors to a flat binary format loadable into ABAP Z-tables.

**How:** 
- Downloads model from HuggingFace Hub
- Reads safetensors file with the `safetensors` library
- Per-channel INT8 quantization: `abs_max / 127`, clip to [-127, 127]
- Writes custom binary format (ALLM magic, header, per-tensor: name + shape + data + scales)

**Output:** `tools/convert_weights.py`

### Phase 1: Model Loader + Z-Tables + Upload Reports (2026-04-04, ~12:00-12:22 UTC)

**What:** Everything needed to get real weights into the ABAP system.

**How:** Sub-agent builds, two rounds (first timed out, second completed).

**Files created:**

| File | Purpose |
|------|---------|
| `ZCL_LLM_MODEL_LOADER` | Loads weights from Z-table or AL11 file into engine tensors |
| `ZLLM_WEIGHTS` (table def) | Model weights: MODEL_ID, LAYER_NAME, WEIGHT_DATA (RAWSTRING), SHAPE, DTYPE, SCALE_DATA |
| `ZLLM_VOCAB` (table def) | BPE vocabulary: MODEL_ID, TOKEN_ID, TOKEN, SCORE |
| `ZLLM_MERGES` (table def) | BPE merge rules: MODEL_ID, PRIORITY, PAIR_LEFT, PAIR_RIGHT |
| `ZLLM_UPLOAD_WEIGHTS` | Report: reads model.bin from AL11, parses ALLM format, inserts into ZLLM_WEIGHTS |
| `ZLLM_UPLOAD_VOCAB` | Report: reads tokenizer.json, parses BPE vocab + merges, inserts into Z-tables |
| `ZLLM_TEST` | Test harness: validates matmul, RMSNorm, SiLU, softmax, BPE, forward pass |

**Weight conversion result:**
- SmolLM2-135M → INT8: 130 MB (`tools/model.bin`)
- 272 tensors, 134.5M parameters
- Handles bfloat16 source format (SmolLM2 uses BF16 in safetensors)

**Reference validator:** `tools/validate_reference.py` — captures intermediate tensor values from Python for verifying ABAP correctness.

**Commit:** `14975a7` — pushed to GitHub

### SmolLM2-135M Available in Ollama (2026-04-04, ~11:52 UTC)

**What:** Pulled SmolLM2-135M into local Ollama for reference benchmarking.

**How:** `ollama pull smollm2:135m` — 270 MB download.

**Result:** Running at ~60 tokens/sec on carlab (12-core CPU). This is our ground truth for validating ABAP output.

---

## What's Next

### Phase 2: End-to-End PoC (pending — needs S/4HANA access)

**Blockers:**
- S/4HANA system (SHC on AWS) not reachable from carlab — all ports closed
- Need: VPN access, IP whitelist, or work through abapGit on Carlos's laptop

**Steps:**
1. Install abapGit on SHC system
2. Pull repo from GitHub
3. Activate all objects (classes, tables, programs)
4. Upload model.bin to AL11 directory on app server
5. Run ZLLM_UPLOAD_WEIGHTS to populate ZLLM_WEIGHTS table
6. Upload tokenizer.json, run ZLLM_UPLOAD_VOCAB
7. Run ZLLM_TEST to validate all components
8. Run ZLLM_DEMO — first text generation in ABAP! 🎉

### Phase 3: HANA Acceleration
- Push matmul to AMDP (HANA SQLScript)
- Explore HANA PAL neural network primitives
- Shared memory weights (load once, share across work processes)
- Target: <3 seconds per token

### Phase 4: Fine-tune + Productize
- Fine-tune SmolLM2 on ABAP code (using CrownSAP MCP data)
- ADT integration via BAdI
- Fiori UI for the assistant
- Package as transportable ABAP add-on

---

## Reference Materials

- SmolLM2-135M: https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- Karpathy's llama2.c: https://github.com/karpathy/llama2.c
- SharpInfer (pure C#): https://jpaulduncan.github.io/SharpInfer/
- Research notes: `memory/research-abap-llm-inference.md`
- Gemma 4 primer: `docs/GEMMA4-PRIMER.md`
- Article draft: `docs/ARTICLE.md`
