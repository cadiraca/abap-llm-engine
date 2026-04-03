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

<!-- 
SmolLM2-135M analysis:
- 135M params, Llama arch
- 576 hidden, 30 layers, 9 heads, 3 KV heads (GQA)
- INT8: ~135MB weights, ~100MB working memory
- Fits in ABAP WP (2-4GB limit)
-->

## Transformer Architecture for ABAP Developers

<!-- 
Explain transformers using SAP analogies:
- Embedding = data element → value mapping
- Attention = a smart SELECT with WHERE clause weighting
- FFN = a BAdI that transforms data
- Layer = a processing step in a workflow
-->

## Building the Foundation: Tensor Operations in ABAP

<!-- 
ZCL_LLM_TENSOR implementation details:
- Internal table representation
- Matrix multiplication (naive → tiled → HANA)
- Benchmarks per approach
-->

## The Tokenizer: Teaching ABAP to Read

<!-- BPE tokenizer implementation -->

## Attention Is All You Need (Even in ABAP)

<!-- GQA implementation, KV cache -->

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
