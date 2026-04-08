"! <p class="shorttext synchronized" lang="en">LLM Engine - Single Transformer Layer</p>
"! Implements one transformer decoder block with the pre-norm
"! architecture used in Llama/SmolLM2:
"!   x = x + attention(input_norm(x))
"!   x = x + ffn(post_norm(x))
CLASS zcl_llm_transformer_block DEFINITION
  PUBLIC
  CREATE PUBLIC.

  PUBLIC SECTION.
    TYPES:
      ty_float_tab TYPE zif_llm_tensor=>ty_float_tab,
      ty_shape     TYPE zif_llm_tensor=>ty_shape,
      ty_kv_cache  TYPE zcl_llm_attention=>ty_kv_cache.

    " Model configuration structure
    TYPES:
      BEGIN OF ty_config,
        hidden_size       TYPE i,
        intermediate_size TYPE i,
        num_heads         TYPE i,
        num_kv_heads      TYPE i,
        head_dim          TYPE i,
        rms_norm_eps      TYPE f,
      END OF ty_config.

    "! <p class="shorttext synchronized">Constructor</p>
    "! @parameter iv_layer_index | Layer index (0-based)
    "! @parameter is_config | Model configuration
    METHODS constructor
      IMPORTING iv_layer_index TYPE i
                is_config      TYPE ty_config.

    "! <p class="shorttext synchronized">Set all layer weights</p>
    "! @parameter io_input_norm | Input RMSNorm weights
    "! @parameter io_post_norm | Post-attention RMSNorm weights
    "! @parameter io_wq | Attention query projection
    "! @parameter io_wk | Attention key projection
    "! @parameter io_wv | Attention value projection
    "! @parameter io_wo | Attention output projection
    "! @parameter io_gate_proj | FFN gate projection
    "! @parameter io_up_proj | FFN up projection
    "! @parameter io_down_proj | FFN down projection
    METHODS set_weights
      IMPORTING io_input_norm TYPE REF TO zif_llm_tensor
                io_post_norm  TYPE REF TO zif_llm_tensor
                io_wq         TYPE REF TO zif_llm_tensor
                io_wk         TYPE REF TO zif_llm_tensor
                io_wv         TYPE REF TO zif_llm_tensor
                io_wo         TYPE REF TO zif_llm_tensor
                io_gate_proj  TYPE REF TO zif_llm_tensor
                io_up_proj    TYPE REF TO zif_llm_tensor
                io_down_proj  TYPE REF TO zif_llm_tensor.

    "! <p class="shorttext synchronized">Forward pass through this transformer block</p>
    "! @parameter io_x | Input hidden state tensor
    "! @parameter iv_position | Token position
    "! @parameter ct_kv_cache | KV cache for this layer (modified in-place)
    "! @parameter ro_output | Output hidden state tensor
    METHODS forward
      IMPORTING io_x             TYPE REF TO zif_llm_tensor
                iv_position      TYPE i
      CHANGING  ct_kv_cache      TYPE ty_kv_cache
      RETURNING VALUE(ro_output) TYPE REF TO zif_llm_tensor.

  PRIVATE SECTION.
    DATA:
      mv_layer_index TYPE i,
      ms_config      TYPE ty_config,
      mo_attention   TYPE REF TO zcl_llm_attention,
      mo_ffn         TYPE REF TO zcl_llm_ffn,
      mo_input_norm  TYPE REF TO zif_llm_tensor,
      mo_post_norm   TYPE REF TO zif_llm_tensor.

ENDCLASS.


CLASS zcl_llm_transformer_block IMPLEMENTATION.

  METHOD constructor.
    mv_layer_index = iv_layer_index.
    ms_config      = is_config.

    " Initialize attention module
    mo_attention = NEW zcl_llm_attention(
      iv_num_heads    = is_config-num_heads
      iv_num_kv_heads = is_config-num_kv_heads
      iv_head_dim     = is_config-head_dim
      iv_hidden_size  = is_config-hidden_size ).

    " Initialize FFN module
    mo_ffn = NEW zcl_llm_ffn(
      iv_hidden_size       = is_config-hidden_size
      iv_intermediate_size = is_config-intermediate_size ).
  ENDMETHOD.

  METHOD set_weights.
    mo_input_norm = io_input_norm.
    mo_post_norm  = io_post_norm.

    mo_attention->set_weights(
      io_wq = io_wq
      io_wk = io_wk
      io_wv = io_wv
      io_wo = io_wo ).

    mo_ffn->set_weights(
      io_gate_proj = io_gate_proj
      io_up_proj   = io_up_proj
      io_down_proj = io_down_proj ).
  ENDMETHOD.

  METHOD forward.
    "--------------------------------------------------------------------
    " Pre-norm transformer block (Llama architecture):
    " 1. normalized = RMSNorm(x, input_norm_weight)
    " 2. attn_out   = Attention(normalized, position, kv_cache)
    " 3. x          = x + attn_out           (residual connection)
    " 4. normalized = RMSNorm(x, post_norm_weight)
    " 5. ffn_out    = FFN(normalized)
    " 6. x          = x + ffn_out            (residual connection)
    "--------------------------------------------------------------------

    " Step 1: Input normalization
    DATA(lo_normed) = zcl_llm_math=>rms_norm(
      io_tensor = io_x
      io_weight = mo_input_norm
      iv_eps    = ms_config-rms_norm_eps ).

    " Step 2: Attention
    DATA lo_attn_out TYPE REF TO zif_llm_tensor.
    mo_attention->forward(
      EXPORTING io_x        = lo_normed
                iv_position = iv_position
      CHANGING  ct_kv_cache = ct_kv_cache
      RECEIVING ro_output   = lo_attn_out ).

    " Step 3: Residual connection after attention
    DATA(lo_residual) = io_x->add( lo_attn_out ).

    " Step 4: Post-attention normalization
    lo_normed = zcl_llm_math=>rms_norm(
      io_tensor = lo_residual
      io_weight = mo_post_norm
      iv_eps    = ms_config-rms_norm_eps ).

    " Step 5: Feed-forward network
    DATA(lo_ffn_out) = mo_ffn->forward( lo_normed ).

    " Step 6: Residual connection after FFN
    ro_output = lo_residual->add( lo_ffn_out ).
  ENDMETHOD.

ENDCLASS.
