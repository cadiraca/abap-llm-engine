#!/usr/bin/env python3
"""
Convert SmolLM2-135M weights from HuggingFace safetensors format
to a flat binary format that can be loaded into ABAP Z-tables.

Output: model.bin — concatenated INT8 quantized weights with header.

Usage:
  pip install transformers safetensors torch numpy
  python convert_weights.py --model HuggingFaceTB/SmolLM2-135M --output model.bin
"""

import argparse
import struct
import numpy as np

def quantize_int8(tensor: np.ndarray):
    """Quantize float32/float16 tensor to int8 with per-channel scale factor."""
    # Per-channel quantization (along last axis)
    abs_max = np.abs(tensor).max(axis=-1, keepdims=True)
    abs_max = np.where(abs_max == 0, 1.0, abs_max)  # avoid division by zero
    scale = abs_max / 127.0
    quantized = np.clip(np.round(tensor / scale), -127, 127).astype(np.int8)
    return quantized, scale.astype(np.float32).flatten()

def main():
    parser = argparse.ArgumentParser(description="Convert SmolLM2-135M to ABAP-loadable format")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M", help="HuggingFace model ID")
    parser.add_argument("--output", default="model.bin", help="Output binary file")
    parser.add_argument("--format", choices=["int8", "float16", "float32"], default="int8")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    
    try:
        from safetensors import safe_open
        from huggingface_hub import hf_hub_download
        import json
        
        # Download config
        config_path = hf_hub_download(args.model, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        
        # Download safetensors
        model_path = hf_hub_download(args.model, "model.safetensors")
        
        print(f"Config: {config['num_hidden_layers']} layers, {config['hidden_size']} hidden, {config['vocab_size']} vocab")
        
        tensors = {}
        with safe_open(model_path, framework="numpy") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        
        print(f"Loaded {len(tensors)} tensors")
        
        # Write binary format
        # Header: magic(4) + version(4) + num_tensors(4) + config_json_len(4) + config_json
        # Per tensor: name_len(2) + name + dtype(1) + ndims(1) + shape(ndims×4) + data_len(4) + data + [scale_len(4) + scale]
        
        with open(args.output, "wb") as out:
            config_bytes = json.dumps(config).encode("utf-8")
            
            # Header
            out.write(b"ALLM")  # ABAP LLM magic
            out.write(struct.pack("<I", 1))  # version
            out.write(struct.pack("<I", len(tensors)))
            out.write(struct.pack("<I", len(config_bytes)))
            out.write(config_bytes)
            
            total_params = 0
            for name, tensor in sorted(tensors.items()):
                data = tensor.astype(np.float32)
                total_params += data.size
                
                # Tensor header
                name_bytes = name.encode("utf-8")
                out.write(struct.pack("<H", len(name_bytes)))
                out.write(name_bytes)
                
                if args.format == "int8":
                    quantized, scales = quantize_int8(data)
                    out.write(struct.pack("B", 1))  # dtype: 1=int8
                    out.write(struct.pack("B", len(data.shape)))
                    for dim in data.shape:
                        out.write(struct.pack("<I", dim))
                    
                    q_bytes = quantized.tobytes()
                    out.write(struct.pack("<I", len(q_bytes)))
                    out.write(q_bytes)
                    
                    s_bytes = scales.tobytes()
                    out.write(struct.pack("<I", len(s_bytes)))
                    out.write(s_bytes)
                    
                elif args.format == "float16":
                    f16_data = data.astype(np.float16)
                    out.write(struct.pack("B", 2))  # dtype: 2=float16
                    out.write(struct.pack("B", len(data.shape)))
                    for dim in data.shape:
                        out.write(struct.pack("<I", dim))
                    d_bytes = f16_data.tobytes()
                    out.write(struct.pack("<I", len(d_bytes)))
                    out.write(d_bytes)
                    out.write(struct.pack("<I", 0))  # no scales
                    
                else:  # float32
                    out.write(struct.pack("B", 0))  # dtype: 0=float32
                    out.write(struct.pack("B", len(data.shape)))
                    for dim in data.shape:
                        out.write(struct.pack("<I", dim))
                    d_bytes = data.tobytes()
                    out.write(struct.pack("<I", len(d_bytes)))
                    out.write(d_bytes)
                    out.write(struct.pack("<I", 0))  # no scales
                
                print(f"  {name}: {data.shape} → {args.format}")
            
            print(f"\nTotal parameters: {total_params:,}")
            print(f"Output: {args.output}")
    
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install transformers safetensors torch numpy huggingface_hub")

if __name__ == "__main__":
    main()
