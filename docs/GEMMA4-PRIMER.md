# Gemma 4: Architecture Primer for the ABAP LLM Engine

*A technical deep dive into Google DeepMind's Gemma 4 — understanding what makes it tick, what Mixture of Experts really means, and how these concepts inform our approach to running LLM inference inside SAP S/4HANA.*

**Author:** Carlos Diego Ramírez  
**Date:** April 2026  
**Context:** This primer accompanies the [ABAP LLM Engine](https://github.com/cadiraca/abap-llm-engine) project.

---

## 1. The Gemma 4 Family at a Glance

Released April 2, 2026, under **Apache 2.0** (fully open — no restrictions on commercial use, redistribution, or fine-tuning). Built from the same technology as Gemini 3.

| Model | Effective Params | Total Params | Active Params | Context | Modalities |
|-------|-----------------|-------------|---------------|---------|------------|
| **E2B** | 2.3B | 5.1B | 2.3B | 128K | Text, Image, Audio |
| **E4B** | 4.5B | 8B | 4.5B | 128K | Text, Image, Audio |
| **26B MoE** | 3.8B active | 25.2B total | 3.8B | 256K | Text, Image |
| **31B Dense** | 31B | 31B | 31B | 256K | Text, Image |

The naming tells the story:
- **"E" = Effective** — the model *behaves* like 2.3B/4.5B at runtime, but carries more parameters on disk (the extra are Per-Layer Embeddings, cheap at inference time).
- **"A4B" = Active 4B** — only 3.8B parameters fire per token. The other 21.4B are dormant experts waiting their turn.

### Where They Run

- **E2B/E4B:** Smartphones. Raspberry Pi. NVIDIA Jetson. Fully offline.
- **26B MoE:** Consumer GPU (quantized) or workstation. This is what we pulled to Ollama.
- **31B Dense:** Single NVIDIA H100 80GB (unquantized).

---

## 2. What Changed From Gemma 3?

The leap is dramatic. On AIME 2026 (a hard math reasoning benchmark):

- Gemma 3 27B: **20.8%**
- Gemma 4 31B: **89.2%**
- Gemma 4 26B MoE: **88.3%** (with only 3.8B active!)

This isn't incremental. It's a qualitative shift in reasoning capability. The MoE hitting 88.3% with 4B active parameters is, in HuggingFace's own words, "mind-blowing."

**Key architectural innovations:**
1. Per-Layer Embeddings (PLE)
2. Shared KV Cache
3. Hybrid Attention (Sliding + Global)
4. MoE with 128 small experts
5. Native function calling (built into training, not prompted)

---

## 3. Understanding Mixture of Experts (MoE)

This is the concept that matters most for our ABAP LLM Engine project, so let's break it down properly.

### The Standard Dense Model

In a normal transformer (like Gemma 4's 31B Dense or our target SmolLM2-135M), every token passes through **every** parameter in the feed-forward network (FFN). If the FFN has 1,536 hidden units, every single input activates all 1,536 units. It's like a restaurant where every chef cooks every dish.

```
Input → [All 31B parameters] → Output
         ↑ everything fires
```

### The MoE Model

In a Mixture of Experts model, the FFN layer is replaced by **multiple smaller FFNs** (the "experts") plus a tiny **router network** that decides which experts to activate for each token.

Gemma 4's 26B MoE has:
- **128 small experts** per MoE layer
- **1 shared expert** that always runs (processes every token)
- **8 experts activated per token** (chosen by the router)
- Router: a small linear layer that produces a probability distribution over the 128 experts

```
Input → Router → selects 8 of 128 experts
                  ↓
         Expert_17: [small FFN] ─┐
         Expert_42: [small FFN] ─┤
         Expert_03: [small FFN] ─┤
         Expert_91: [small FFN] ─┼→ Weighted sum → Output
         Expert_55: [small FFN] ─┤
         Expert_78: [small FFN] ─┤
         Expert_12: [small FFN] ─┤
         Expert_64: [small FFN] ─┤
         Shared_Expert: [FFN] ───┘  (always active)
```

### Why 128 Small Experts Instead of 8 Large Ones?

Most MoE models (like Mixtral) use 8 large experts and activate 2. Gemma 4 goes the opposite direction: 128 tiny experts, activate 8. Why?

**Finer-grained specialization.** With 128 experts, each one can specialize in a very narrow domain — maybe Expert_42 is great at arithmetic patterns, Expert_17 handles code syntax, Expert_91 knows about spatial reasoning. The router learns to assemble exactly the right "team" for each token.

**Better routing.** With more experts, the router has more options. It's like having 128 specialized consultants on speed dial vs. 8 generalists — you can build more precise combinations.

**Same compute, more knowledge.** Total parameters: 25.2B (lots of knowledge stored). Active parameters: 3.8B (fast inference). You get the intelligence of a large model with the speed of a small one.

### The Router: How Tokens Choose Their Experts

The router is just a linear projection:

```
router_logits = input × W_router    (shape: hidden_size × num_experts)
expert_weights = softmax(top_k(router_logits, k=8))
```

For each token:
1. Compute a score for all 128 experts
2. Keep the top 8 scores
3. Softmax to get weights (how much each expert contributes)
4. Run the token through those 8 experts
5. Combine outputs using the weights

The router is trained end-to-end with the rest of the model. During training, auxiliary losses ensure tokens are distributed somewhat evenly across experts (to avoid "expert collapse" where the router sends everything to the same 3 experts).

### The Compute Math

- **31B Dense:** Every token → 31B computations
- **26B MoE:** Every token → ~3.8B computations (shared expert + 8 small experts)
- **Speedup:** ~8x less compute per token, but the same quality because the *total knowledge* across all experts is 25.2B

This is why the MoE runs so fast on our carlab — even though it's "26B," inference costs are closer to a 4B model.

---

## 4. Per-Layer Embeddings (PLE): The Other Innovation

Standard transformers give each token ONE embedding vector at input. This single vector must contain everything every layer might need — a compression bottleneck. Imagine writing one sticky note that has to serve as instructions for 30 different people.

PLE adds a **second, smaller embedding table** that generates a dedicated signal for every layer:

```
Standard:
  Token → [one embedding] → Layer 1 → Layer 2 → ... → Layer 30

PLE:
  Token → [main embedding] → Layer 1 ← [PLE signal for Layer 1]
                            → Layer 2 ← [PLE signal for Layer 2]
                            → ...
                            → Layer 30 ← [PLE signal for Layer 30]
```

Each layer gets its own little "hint" about the token. The PLE dimension is much smaller than the main hidden size — it's cheap in compute but adds meaningful specialization per layer.

This is why E2B has 5.1B parameters on disk but only 2.3B "effective" — the extra 2.8B are PLE tables that add storage but minimal compute at inference time.

### Relevance to Our Project

We won't implement PLE in the ABAP engine (SmolLM2-135M doesn't use it). But understanding PLE explains Google's strategy: **decouple storage from compute.** You can pack more knowledge into a model without making inference proportionally more expensive. MoE is the extreme version of this same principle.

---

## 5. Shared KV Cache

During autoregressive generation, each layer computes Key and Value tensors that get cached for future tokens (so you don't recompute them). With 30+ layers, this KV cache gets expensive — especially at 256K context.

Gemma 4's trick: the **last N layers don't compute their own K/V**. They reuse K/V from an earlier layer of the same attention type. Quality impact: minimal. Memory savings: significant, especially at long context.

### Relevance to Our Project

We **will** implement KV caching in the ABAP engine — it's essential for autoregressive generation. Without it, generating 20 tokens would require 20 full forward passes through all layers. With KV cache, only the new token goes through the model while previous tokens' K/V tensors are reused.

---

## 6. Hybrid Attention (Sliding + Global)

Gemma 4 alternates between two types of attention:

- **Sliding window attention:** Each token only attends to the nearest 512-1024 tokens. Fast, O(n × window) instead of O(n²). Most layers use this.
- **Global attention:** Each token attends to ALL previous tokens. Expensive but necessary for long-range dependencies. Every 6th layer (roughly).

The final layer is always global. RoPE (rotary position embeddings) is configured differently for each type — standard RoPE for sliding, proportional RoPE for global (to handle 256K positions).

### Relevance to Our Project

SmolLM2-135M uses standard full attention (not hybrid). But understanding this pattern matters for future work — if we ever build a larger ABAP model or adapt a Gemma variant, hybrid attention would be critical for fitting in ABAP's memory constraints.

---

## 7. What MoE Means for LLMs Inside SAP

Here's where it gets interesting for our ABAP LLM Engine:

### Could We Run an MoE Inside ABAP?

**Technically yes**, and it might actually be a better fit than a dense model:

1. **Memory:** An MoE stores all experts in memory (25.2B params) but only activates a fraction. In ABAP, we'd need to load all experts into shared memory, but the actual *compute* per token is small.

2. **HANA as Expert Storage:** Imagine storing expert weights in HANA in-memory tables. The router selects which 8 experts to load. HANA fetches them in microseconds. The FFN computation happens in AMDP. This maps surprisingly well to MoE architecture.

3. **Custom Expert Training:** You could train domain-specific experts — one for Finance posting logic, one for MM procurement patterns, one for SD pricing. The router learns when each expert is needed. An SAP-specialized MoE.

### Why We Start With Dense (SmolLM2-135M)

For the PoC, a dense model is simpler:
- No router to implement
- No expert selection logic  
- Every forward pass is deterministic
- 135M params fit in one work process without tricks

Once the dense PoC works, MoE is a natural next step — especially with HANA as the expert storage backend.

### The Dream Architecture: SAP-Native MoE

```
ABAP Application Server
├── ZCL_LLM_ROUTER → Selects experts based on token context
├── HANA In-Memory Tables → Store expert weights (each expert ~50MB)
│   ├── Expert: FI (Finance posting patterns)
│   ├── Expert: MM (Procurement/MRP patterns)
│   ├── Expert: SD (Sales/pricing patterns)
│   ├── Expert: ABAP_SYNTAX (code generation)
│   ├── Expert: CONFIG (customizing patterns)
│   └── ... (128 specialized experts)
├── AMDP → Parallel FFN computation in HANA
└── ZCL_LLM_ENGINE → Orchestrator, tokenizer, attention
```

Each expert is a specialist in one SAP domain. The router learns which experts to activate based on the prompt. Ask about pricing conditions? The SD expert lights up. Ask about FI posting keys? The FI expert activates. Code question? ABAP_SYNTAX expert.

This is the long-term vision. The ABAP LLM Engine PoC proves the foundation is possible.

---

## 8. Native Function Calling: The Agentic Angle

Gemma 4 was trained with **native function calling** — not prompted into it, but trained from the ground up to invoke tools. This enables:

- Multi-turn agentic workflows
- Structured JSON output
- Tool invocation in production

For our ABAP context, imagine:
```
User: "What tables store purchase order data?"
Model: <calls tool: search_data_dictionary(query="purchase order")>
System returns: EKKO, EKPO, EKKN, EKET...
Model: "The main purchase order tables are EKKO (header), EKPO (items)..."
```

The CrownSAP MCP Server we're building today provides exactly this kind of tool interface. A future ABAP-native model + MCP = zero-hallucination SAP assistant.

---

## 9. Summary: What We Take Into the ABAP Engine

| Gemma 4 Concept | Status in ABAP Engine | Notes |
|-----------------|----------------------|-------|
| Dense Transformer | **Phase 1** (SmolLM2-135M) | Core PoC |
| MoE / Expert Routing | **Future** | HANA as expert store |
| PLE | Skip | SmolLM2 doesn't use it |
| Shared KV Cache | **Phase 2** | Memory optimization |
| Hybrid Attention | **Future** | For larger models |
| RoPE | **Phase 1** | Standard RoPE in SmolLM2 |
| GQA (Grouped-Query Attention) | **Phase 1** | SmolLM2 uses 9 heads, 3 KV heads |
| Native Function Calling | **Future** | Combine with CrownSAP MCP |
| Quantization (INT8) | **Phase 1** | Essential for ABAP memory limits |

---

## 10. Further Reading

- [Google's Gemma 4 announcement](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [HuggingFace Gemma 4 deep dive](https://huggingface.co/blog/gemma4)
- [Pebblous Gemma 4 deep analysis](https://blog.pebblous.ai/story/google-gemma-4-report-pb/en/)
- [Karpathy's llama2.c](https://github.com/karpathy/llama2.c) — Reference for minimal inference implementation
- [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) — Our target model

---

*This primer is Chapter 0 of the ABAP LLM Engine series. Next: "Building Tensor Operations in ABAP" — implementing matrix multiplication, the heart of every neural network, in a language designed for enterprise resource planning.*
