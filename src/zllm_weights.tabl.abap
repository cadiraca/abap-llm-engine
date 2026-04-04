"! <p class="shorttext synchronized" lang="en">LLM Engine - Model Weight Storage</p>
"! Transparent table to store quantized model weights for the ABAP LLM Engine.
"! Each row represents one named tensor (e.g. a weight matrix or bias vector)
"! from the HuggingFace model.  Weight data is stored as raw bytes (RAWSTRING)
"! so that large tensors do not need to be converted to ABAP float tables before
"! being inserted.  The SCALE_DATA field holds per-channel float32 scale factors
"! required to dequantize INT8 weights back to float32 during inference.
"!
"! Naming convention for LAYER_NAME matches HuggingFace safetensors keys:
"!   model.embed_tokens.weight
"!   model.layers.0.self_attn.q_proj.weight
"!   model.layers.0.mlp.gate_proj.weight
"!   ...
"!   model.norm.weight
"!   lm_head.weight
@EndUserText.label: 'LLM Engine - Model Weight Storage'
@AbapCatalog.enhancement.category: #NOT_EXTENSIBLE
@AbapCatalog.tableCategory: #TRANSPARENT
@AbapCatalog.deliveryClass: #APPLICATION_DATA
@AbapCatalog.dataMaintenance: #RESTRICTED
define table zllm_weights {
  key mandt      : mandt not null;
  key model_id   : abap.char(40) not null;
  key layer_name : abap.char(200) not null;
  weight_data    : abap.rawstring(0);
  shape          : abap.string(0);
  dtype          : abap.char(10);
  scale_data     : abap.rawstring(0);
}
