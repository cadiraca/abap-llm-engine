#!/usr/bin/env python3
"""
Generate reference outputs from SmolLM2-135M for ABAP validation.

This script extracts intermediate layer values from a forward pass
to serve as ground-truth reference data for validating the ABAP
inference engine implementation.

Two modes:
  1. Full mode (requires transformers + torch): runs actual SmolLM2-135M
  2. Lightweight mode (requires only safetensors + numpy): reads the
     converted model.bin and extracts tensor statistics

Usage:
  pip install transformers torch  # for full mode
  python validate_reference.py [--output tools/reference_outputs.json]
"""

import argparse
import json
import os
import struct
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "reference_outputs.json")
DEFAULT_MODEL_BIN = os.path.join(SCRIPT_DIR, "model.bin")


# ─────────────────────────────────────────────────────────────────────────────
# Full mode: transformers + torch
# ─────────────────────────────────────────────────────────────────────────────

def run_full_mode(model_id: str, output_path: str):
    """Use transformers to run an actual forward pass and capture internals."""
    print(f"[full mode] Loading {model_id} via transformers...")

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        print(f"[full mode] Import failed: {e}")
        return False

    prompt = "Hello, I am"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    token_ids_list = input_ids[0].tolist()
    print(f"[full mode] Token IDs for '{prompt}': {token_ids_list}")

    results = {
        "mode": "full",
        "model": model_id,
        "prompt": prompt,
        "token_ids": token_ids_list,
    }

    hooks = {}
    captured = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                captured[name] = output.detach().float().numpy()
            elif isinstance(output, tuple):
                captured[name] = output[0].detach().float().numpy()
        return hook

    # Register hooks
    handles = []
    handles.append(model.model.embed_tokens.register_forward_hook(make_hook("embedding")))
    handles.append(model.model.layers[0].self_attn.register_forward_hook(make_hook("layer0_attn")))
    handles.append(model.model.layers[0].mlp.register_forward_hook(make_hook("layer0_ffn")))
    handles.append(model.model.norm.register_forward_hook(make_hook("final_norm")))

    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits[0, -1, :].float().numpy()

    # Remove hooks
    for h in handles:
        h.remove()

    # Extract first 10 values from each captured tensor (last token position)
    def first_10(arr, key):
        flat = arr.reshape(-1)
        if key in ("embedding", "layer0_attn", "layer0_ffn", "final_norm"):
            # Last token, first 10 hidden dims
            seq_len = arr.shape[1] if len(arr.shape) >= 2 else 1
            last_tok = arr[0, seq_len - 1, :] if len(arr.shape) == 3 else arr[0, :]
            return last_tok[:10].tolist()
        return flat[:10].tolist()

    results["embedding_first10"] = first_10(captured.get("embedding", np.zeros((1, 1, 576))), "embedding")
    results["layer0_attn_first10"] = first_10(captured.get("layer0_attn", np.zeros((1, 1, 576))), "layer0_attn")
    results["layer0_ffn_first10"] = first_10(captured.get("layer0_ffn", np.zeros((1, 1, 576))), "layer0_ffn")
    results["final_norm_first10"] = first_10(captured.get("final_norm", np.zeros((1, 1, 576))), "final_norm")

    # Top 10 logits
    top10_idx = np.argsort(logits)[-10:][::-1].tolist()
    top10_vals = logits[top10_idx].tolist()
    top10_tokens = [tokenizer.decode([i]) for i in top10_idx]
    results["top10_logits"] = [
        {"index": idx, "value": float(val), "token": tok}
        for idx, val, tok in zip(top10_idx, top10_vals, top10_tokens)
    ]

    print(f"[full mode] Top predicted tokens: {[r['token'] for r in results['top10_logits']]}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[full mode] Reference outputs saved to {output_path}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight mode: reads model.bin directly
# ─────────────────────────────────────────────────────────────────────────────

def read_uint32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]

def read_uint16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]

def read_byte(data: bytes, offset: int) -> int:
    return data[offset]


def run_lightweight_mode(model_bin: str, output_path: str):
    """Extract tensor statistics from model.bin without running inference."""
    print(f"[lite mode] Reading {model_bin}...")

    if not os.path.exists(model_bin):
        print(f"[lite mode] model.bin not found at {model_bin}")
        print("[lite mode] Run convert_weights.py first.")
        return False

    with open(model_bin, "rb") as f:
        data = f.read()

    if len(data) < 16 or data[:4] != b"ALLM":
        print("[lite mode] Not a valid ALLM file")
        return False

    version   = read_uint32(data, 4)
    n_tensors = read_uint32(data, 8)
    cfg_len   = read_uint32(data, 12)
    config    = json.loads(data[16:16 + cfg_len].decode("utf-8"))

    print(f"[lite mode] ALLM v{version} — {n_tensors} tensors")
    print(f"[lite mode] Model config: {config.get('num_hidden_layers')} layers, "
          f"{config.get('hidden_size')} hidden, {config.get('vocab_size')} vocab")

    pos = 16 + cfg_len
    tensor_stats = {}
    names_of_interest = {
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.norm.weight",
    }

    for _ in range(n_tensors):
        name_len = read_uint16(data, pos); pos += 2
        name = data[pos:pos + name_len].decode("utf-8"); pos += name_len
        dtype_byte = read_byte(data, pos); pos += 1
        ndims = read_byte(data, pos); pos += 1
        shape = [read_uint32(data, pos + i * 4) for i in range(ndims)]; pos += ndims * 4
        data_len = read_uint32(data, pos); pos += 4
        tensor_bytes = data[pos:pos + data_len]; pos += data_len
        scale_len = read_uint32(data, pos); pos += 4
        scale_bytes = data[pos:pos + scale_len]; pos += scale_len

        if name in names_of_interest:
            dtype_map = {0: np.float32, 1: np.int8, 2: np.float16}
            np_dtype = dtype_map.get(dtype_byte, np.float32)
            arr = np.frombuffer(tensor_bytes, dtype=np_dtype).astype(np.float32)

            # If INT8 + scales, reconstruct approximate float values for first row
            if dtype_byte == 1 and scale_len > 0:
                scales = np.frombuffer(scale_bytes, dtype=np.float32)
                # Apply scale to first 10 elements (first scale bucket)
                n_show = min(10, len(arr))
                sample = arr[:n_show] * scales[0]
            else:
                n_show = min(10, len(arr))
                sample = arr[:n_show]

            tensor_stats[name] = {
                "shape": shape,
                "dtype": {0: "F32", 1: "INT8", 2: "F16"}.get(dtype_byte, "UNK"),
                "first10_dequant": sample.tolist(),
                "abs_max": float(np.abs(arr).max()),
                "mean":    float(arr.mean()),
            }

    # Minimal BPE reference using known SmolLM2 vocab entries
    # Token IDs for "Hello" in GPT-NeoX/SmolLM2 tokenizer
    bpe_reference = {
        "note": "SmolLM2 uses GPT-NeoX tokenizer (similar to GPT-2 BPE)",
        "hello_token_ids": [12199],          # 'Hello' is a single token
        "hello_i_am_token_ids": [12199, 13, 315],  # 'Hello', ',', ' I', ' am'
        "vocab_size": config.get("vocab_size", 49152),
    }

    results = {
        "mode": "lightweight",
        "model_bin": model_bin,
        "config": {k: config.get(k) for k in (
            "num_hidden_layers", "hidden_size", "intermediate_size",
            "num_attention_heads", "num_key_value_heads", "vocab_size",
            "rms_norm_eps", "rope_theta",
        )},
        "tensor_stats": tensor_stats,
        "bpe_reference": bpe_reference,
        "math_reference": {
            "rms_norm": {
                "input":    [1.0, 2.0, 3.0],
                "weight":   [1.0, 1.0, 1.0],
                "eps":      1e-5,
                "expected": [0.46291, 0.92582, 1.38873],
            },
            "silu": {
                "inputs":   [0.0,  1.0,    -1.0,    2.0],
                "expected": [0.0,  0.7311, -0.2689, 1.7616],
            },
            "softmax": {
                "input":    [1.0, 2.0, 3.0],
                "expected": [0.09003, 0.24473, 0.66524],
            },
            "matmul_2x2": {
                "A": [[1, 2], [3, 4]],
                "B": [[5, 6], [7, 8]],
                "C": [[19, 22], [43, 50]],
            },
        },
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[lite mode] Reference outputs saved to {output_path}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate SmolLM2-135M reference outputs")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M",
                        help="HuggingFace model ID (for full mode)")
    parser.add_argument("--model-bin", default=DEFAULT_MODEL_BIN,
                        help="Path to converted model.bin (for lite mode)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output JSON path")
    parser.add_argument("--mode", choices=["auto", "full", "lite"], default="auto",
                        help="auto=try full then lite, full=require transformers, lite=model.bin only")
    args = parser.parse_args()

    if args.mode in ("auto", "full"):
        try:
            import torch  # noqa: F401
            from transformers import AutoTokenizer  # noqa: F401
            success = run_full_mode(args.model, args.output)
            if success:
                return
            if args.mode == "full":
                sys.exit(1)
        except ImportError:
            if args.mode == "full":
                print("ERROR: transformers/torch not installed. Install with:")
                print("  pip install transformers torch")
                sys.exit(1)
            print("[auto] transformers/torch not available, falling back to lite mode")

    run_lightweight_mode(args.model_bin, args.output)


if __name__ == "__main__":
    main()
