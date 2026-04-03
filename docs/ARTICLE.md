# Running an LLM Inside SAP S/4HANA: Pure ABAP Neural Network Inference

*By Carlos Diego Ramírez — April 2026*

> What if your SAP system could think? Not by calling an external API, but by running a neural network directly inside the ABAP application server. This article documents the journey of building a complete LLM inference engine in pure ABAP — from matrix multiplication to text generation.

---

## Table of Contents

1. [The Crazy Idea](#the-crazy-idea)
2. [Why Run an LLM Inside SAP?](#why-run-an-llm-inside-sap)
3. [Choosing the Right Model](#choosing-the-right-model)
4. [Transformer Architecture for ABAP Developers](#transformer-architecture-for-abap-developers)
5. [Building the Foundation: Tensor Operations in ABAP](#building-the-foundation)
6. [The Tokenizer: Teaching ABAP to Read](#the-tokenizer)
7. [Attention Is All You Need (Even in ABAP)](#attention)
8. [The HANA Secret Weapon: AMDP as a GPU](#hana-acceleration)
9. [Weight Loading: Transporting a Brain via SAP Transports](#weight-loading)
10. [Memory Management: Fitting 135M Parameters in a Work Process](#memory-management)
11. [Results and Benchmarks](#results)
12. [What It Can Do: Use Cases Inside S/4HANA](#use-cases)
13. [Lessons Learned](#lessons-learned)
14. [What's Next](#whats-next)

---

## The Crazy Idea

It started with a simple question: *"Can you upload a tiny LLM to an S/4HANA server and make inferences inside it?"*

The answer, it turns out, is yes. Not just theoretically — we actually built it.

<!-- TODO: Fill in sections as implementation progresses -->

## Why Run an LLM Inside SAP?

Every SAP developer has experienced the frustration of LLM hallucination. You ask Claude or ChatGPT to write some ABAP code, and it confidently suggests `CL_SOME_CLASS` that doesn't exist, or references a CDS view with fields that were never there.

The problem isn't the model — it's the disconnect. The model has no idea what objects actually exist in *your* system. But what if the model ran *inside* the system? Then it could:

```abap
" The model can verify its own suggestions
SELECT SINGLE @abap_true FROM dd03l INTO @DATA(lv_exists)
  WHERE tabname = 'EKKO' AND fieldname = 'EBELN'.
```

No hallucination. Direct verification. The model and the data live in the same memory space.

## Choosing the Right Model

The model needs to fit inside a single ABAP work process — typically 2-4 GB of memory. That rules out anything with billions of parameters. After evaluating several small models, we landed on **SmolLM2-135M** from HuggingFace:

| Property | Value |
|----------|-------|
| Parameters | 135 million |
| Architecture | Llama (decoder-only transformer) |
| Hidden size | 576 |
| Layers | 30 |
| Attention heads | 9 (query), 3 (KV) — Grouped-Query Attention |
| FFN intermediate | 1536 |
| Vocabulary | 49,152 tokens (BPE) |
| Context window | 8,192 tokens |
| FP32 memory | ~540 MB |
| INT8 memory | ~135 MB (weights) + ~100 MB (working) |

The key insight: 135M parameters in INT8 is ~135 MB. With working memory for KV cache, activations, and intermediate tensors, we land at roughly 250 MB. That comfortably fits in an ABAP work process.

SmolLM2 uses the Llama architecture — the same foundation as Meta's Llama 2/3. This means our engine can theoretically run *any* Llama-compatible model, just limited by memory.

## Transformer Architecture for ABAP Developers

If you're an ABAP developer who's heard "transformer" a thousand times but never looked inside one, here's the architecture in SAP terms:

**Embedding** = A data element lookup table. Token ID 42 maps to a 576-dimensional vector, just like a domain value maps to a description. Our `ZCL_LLM_ENGINE=>embed_token()` does exactly this — slicing a row from the embedding matrix.

**Self-Attention** = A smart `SELECT` with dynamic `WHERE` clause weighting. For each token, we compute how much it should "attend to" every previous token. The Query is what we're looking for, the Key is what each position offers, and the Value is the actual content. Our `ZCL_LLM_ATTENTION` implements Grouped-Query Attention (GQA) where 9 query heads share 3 KV heads — a 3:1 ratio that saves memory without losing much quality.

**Feed-Forward Network** = A BAdI that transforms data through a non-linear gate. Our `ZCL_LLM_FFN` uses the SwiGLU pattern: `down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))`. The SiLU activation (x × sigmoid(x)) acts as a learned gate that selectively passes information.

**Transformer Block** = One processing step in a workflow. Our `ZCL_LLM_TRANSFORMER_BLOCK` chains them: normalize → attend → add residual → normalize → FFN → add residual. We stack 30 of these blocks.

**The Full Pipeline:**
```
Token → Embedding → [Block₁ → Block₂ → ... → Block₃₀] → Norm → LM Head → Logits → Sample → Token
```

Each block refines the representation. By layer 30, the model has built up enough understanding to predict the next token.

## Building the Foundation: Tensor Operations in ABAP

The first question: how do you represent a multi-dimensional tensor in ABAP? There's no NumPy here.

Our answer: **a flat `STANDARD TABLE OF f` with a shape vector**. A 2D matrix of shape (576, 1536) is stored as 884,736 contiguous float values in row-major order. Element `[i][j]` lives at index `i * 1536 + j`.

```abap
" ZCL_LLM_TENSOR — the backbone of everything
DATA:
  mt_data  TYPE STANDARD TABLE OF f WITH EMPTY KEY,
  mt_shape TYPE STANDARD TABLE OF i WITH EMPTY KEY,
  mv_size  TYPE i.
```

### Matrix Multiplication: The Hot Loop

Matrix multiplication is where the engine spends 95%+ of its time. A naive triple-nested loop works but thrashes the CPU cache. We implemented **tiled (blocked) matrix multiplication** with a block size of 32:

```abap
" Tiled matmul: process 32×32 blocks for cache locality
CONSTANTS c_tile_size TYPE i VALUE 32.

lv_ii = 0.
WHILE lv_ii < lv_m.
  lv_kk = 0.
  WHILE lv_kk < lv_k.
    lv_jj = 0.
    WHILE lv_jj < lv_n.
      " Inner tile: this block fits in L1 cache
      " ... accumulate partial dot products ...
      lv_jj = lv_jj + c_tile_size.
    ENDWHILE.
    lv_kk = lv_kk + c_tile_size.
  ENDWHILE.
  lv_ii = lv_ii + c_tile_size.
ENDWHILE.
```

The tiled approach keeps data in CPU cache between accesses. Instead of jumping across a 576-wide row for every multiply, we work on a 32×32 block that fits in L1. In our benchmarks, this gives a 2-3× speedup over naive matmul in pure ABAP.

The `ZIF_LLM_TENSOR` interface defines all tensor operations (matmul, add, elementwise multiply, scale, reshape, slice), and `ZCL_LLM_TENSOR` implements them with factory methods (`create_from_float_table`, `create_zeros`, `create_from_shape`).

## The Tokenizer: Teaching ABAP to Read

Before the model can process text, it needs to convert strings into token IDs. SmolLM2 uses **Byte-Pair Encoding (BPE)** — the same algorithm behind most modern LLM tokenizers.

The algorithm is elegant:
1. Start with individual characters
2. Find the highest-priority merge pair in the vocabulary
3. Merge those adjacent tokens into one
4. Repeat until no more merges apply

Our `ZCL_LLM_BPE_TOKENIZER` takes a vocabulary (49,152 token strings) and a priority-ordered merge table:

```abap
" Iterative BPE merge loop
DO.
  find_best_merge(
    EXPORTING it_tokens    = lt_tokens
    IMPORTING ev_merge_idx = lv_merge_idx
              ev_pos       = lv_merge_pos ).
  IF lv_merge_idx = 0.
    EXIT.  " No more merges apply
  ENDIF.
  " Combine adjacent tokens
  lt_tokens[ lv_merge_pos ] = lt_tokens[ lv_merge_pos ] &&
                               lt_tokens[ lv_merge_pos + 1 ].
  DELETE lt_tokens INDEX lv_merge_pos + 1.
ENDDO.
```

For the PoC, the merge rules are loaded from the model's tokenizer data. The full SmolLM2 vocabulary covers 49,152 tokens — from single bytes to common multi-character sequences like `' the'` or `'function'`.

## Attention Is All You Need (Even in ABAP)

The attention mechanism is the heart of the transformer. Our `ZCL_LLM_ATTENTION` implements **Grouped-Query Attention (GQA)** — a memory-efficient variant where multiple query heads share the same key-value heads.

SmolLM2 config: 9 query heads, 3 KV heads. Every 3 query heads share 1 KV head. This cuts KV cache memory by 3× with minimal quality loss.

### The Forward Pass

```abap
" 1. Project input to Q, K, V
lo_q = io_x->matmul( mo_wq ).  " (576) → (576) = 9 heads × 64 dim
lo_k = io_x->matmul( mo_wk ).  " (576) → (192) = 3 heads × 64 dim
lo_v = io_x->matmul( mo_wv ).  " (576) → (192) = 3 heads × 64 dim

" 2. Apply RoPE (Rotary Position Embeddings)
zcl_llm_math=>rope( io_q = lo_q  io_k = lo_k
                    iv_position = iv_position  iv_head_dim = 64 ).

" 3. Cache K, V for this position
INSERT ls_kv_entry INTO TABLE ct_kv_cache.

" 4. For each query head: scaled dot-product attention
"    score = (Q · K) / sqrt(head_dim)
"    weights = softmax(scores)  — causal mask built-in
"    output = weights · V

" 5. Concatenate heads → output projection
ro_output = lo_concat->matmul( mo_wo ).
```

### KV Cache: Remembering Previous Tokens

For autoregressive generation, we cache the K and V projections at every position. When generating token 50, we don't recompute K and V for tokens 0-49 — we just look them up. This turns generation from O(n²) to O(n) per token — critical when you're already spending seconds per token.

### RoPE: Position Without Learned Embeddings

Rotary Position Embeddings encode position through rotation in complex space. For each dimension pair (2i, 2i+1), we rotate by an angle proportional to position and frequency. Our `ZCL_LLM_MATH=>rope()` applies this in-place to both Q and K tensors. No learned position embeddings needed — the rotation naturally encodes relative position.

## The HANA Secret Weapon: AMDP as a GPU

<!-- 
The key insight: HANA runs on massive multi-core machines
AMDP pushes compute to HANA's engine
Benchmark comparison: pure ABAP vs AMDP
-->

## Weight Loading: Transporting a Brain via SAP Transports

<!-- 
Z-table storage, transport system, abapGit deployment
The absurdity of deploying neural network weights through STMS
-->

## Memory Management: Fitting 135M Parameters in a Work Process

<!-- 
ABAP memory architecture
Extended memory, heap memory, shared memory
Quantization: float32 → int8 reduces memory 4x
-->

## Results and Benchmarks

<!-- Performance numbers, sample outputs -->

## What It Can Do: Use Cases Inside S/4HANA

<!-- Practical applications -->

## Lessons Learned

<!-- What worked, what didn't, what surprised us -->

## What's Next

<!-- Fine-tuning, ABAP Cloud integration, ADT plugin -->
