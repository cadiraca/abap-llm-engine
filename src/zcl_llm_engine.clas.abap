"! <p class="shorttext synchronized" lang="en">LLM Engine - Main Orchestrator</p>
"! Top-level class orchestrating the full transformer inference pipeline.
"! Manages the tokenizer, embedding, transformer layers, final norm,
"! language model head, and autoregressive text generation.
"!
"! Target model: SmolLM2-135M (Llama architecture)
"!   - 30 transformer layers
"!   - 576 hidden size, 1536 intermediate
"!   - 9 attention heads, 3 KV heads (GQA)
"!   - 49152 vocab size
CLASS zcl_llm_engine DEFINITION
  PUBLIC
  CREATE PUBLIC.

  PUBLIC SECTION.
    TYPES:
      ty_float_tab TYPE zif_llm_tensor=>ty_float_tab,
      ty_shape     TYPE zif_llm_tensor=>ty_shape,

      "! Weight source specification
      BEGIN OF ty_weight_source,
        type   TYPE string,  " 'ZTAB' or 'FILE'
        path   TYPE string,  " Table name or file path
      END OF ty_weight_source,

      "! Model configuration
      BEGIN OF ty_model_config,
        vocab_size        TYPE i,
        hidden_size       TYPE i,
        intermediate_size TYPE i,
        num_layers        TYPE i,
        num_heads         TYPE i,
        num_kv_heads      TYPE i,
        head_dim          TYPE i,
        max_position      TYPE i,
        rms_norm_eps      TYPE f,
        rope_theta        TYPE f,
      END OF ty_model_config,

      "! KV cache: one entry per layer
      BEGIN OF ty_layer_kv_cache,
        layer_index TYPE i,
        kv_cache    TYPE zcl_llm_attention=>ty_kv_cache,
      END OF ty_layer_kv_cache,
      ty_kv_caches TYPE STANDARD TABLE OF ty_layer_kv_cache WITH KEY layer_index.

    "! <p class="shorttext synchronized">Constructor</p>
    "! Initializes engine with SmolLM2-135M default configuration.
    "! @parameter is_config | Optional config override
    METHODS constructor
      IMPORTING is_config TYPE ty_model_config OPTIONAL.

    "! <p class="shorttext synchronized">Load model weights</p>
    "! @parameter iv_source | Weight source type ('ZTAB' or 'FILE')
    "! @parameter iv_path | Source path (table name or file path)
    METHODS load_weights
      IMPORTING iv_source TYPE string DEFAULT 'ZTAB'
                iv_path   TYPE string DEFAULT 'ZLLM_WEIGHTS'.

    "! <p class="shorttext synchronized">Set tokenizer</p>
    "! @parameter io_tokenizer | Initialized tokenizer instance
    METHODS set_tokenizer
      IMPORTING io_tokenizer TYPE REF TO zcl_llm_bpe_tokenizer.

    "! <p class="shorttext synchronized">Generate text from prompt</p>
    "! @parameter iv_prompt | Input prompt text
    "! @parameter iv_max_tokens | Maximum tokens to generate (default 20)
    "! @parameter iv_temperature | Sampling temperature (default 0.7)
    "! @parameter iv_top_k | Top-K sampling (default 40)
    "! @parameter iv_top_p | Top-P nucleus sampling (default 0.9)
    "! @parameter rv_result | Generated text (prompt + completion)
    METHODS generate
      IMPORTING iv_prompt        TYPE string
                iv_max_tokens    TYPE i DEFAULT 20
                iv_temperature   TYPE f DEFAULT '0.7'
                iv_top_k         TYPE i DEFAULT 40
                iv_top_p         TYPE f DEFAULT '0.9'
      RETURNING VALUE(rv_result) TYPE string.

    "! <p class="shorttext synchronized">Single forward pass</p>
    "! Runs one forward pass through the full model, returning logits.
    "! @parameter it_token_ids | Input token IDs (currently processes last token)
    "! @parameter iv_position | Position index for current token
    "! @parameter ct_kv_caches | KV caches per layer (modified in-place)
    "! @parameter ro_logits | Output logits tensor (vocab_size)
    METHODS forward_pass
      IMPORTING it_token_ids     TYPE zcl_llm_bpe_tokenizer=>ty_token_ids
                iv_position      TYPE i
      CHANGING  ct_kv_caches     TYPE ty_kv_caches
      RETURNING VALUE(ro_logits) TYPE REF TO zif_llm_tensor.

    "! <p class="shorttext synchronized">Get model config</p>
    "! @parameter rs_config | Current model configuration
    METHODS get_config
      RETURNING VALUE(rs_config) TYPE ty_model_config.

  PRIVATE SECTION.
    DATA:
      ms_config          TYPE ty_model_config,
      mo_tokenizer       TYPE REF TO zcl_llm_bpe_tokenizer,
      mo_sampler         TYPE REF TO zcl_llm_sampler,
      mt_blocks          TYPE STANDARD TABLE OF REF TO zcl_llm_transformer_block
                           WITH EMPTY KEY,
      mo_embed_tokens    TYPE REF TO zif_llm_tensor,  " (vocab, hidden)
      mo_final_norm      TYPE REF TO zif_llm_tensor,  " RMSNorm weight
      mo_lm_head         TYPE REF TO zif_llm_tensor,  " (hidden, vocab)
      mv_weights_loaded  TYPE abap_bool.

    "! Initialize transformer blocks
    METHODS init_blocks.

    "! Look up embedding for a token
    METHODS embed_token
      IMPORTING iv_token_id        TYPE i
      RETURNING VALUE(ro_embedding) TYPE REF TO zif_llm_tensor.

    "! Initialize KV caches for all layers
    METHODS init_kv_caches
      RETURNING VALUE(rt_kv_caches) TYPE ty_kv_caches.

ENDCLASS.


CLASS zcl_llm_engine IMPLEMENTATION.

  METHOD constructor.
    " SmolLM2-135M default configuration
    IF is_config IS SUPPLIED AND is_config-hidden_size > 0.
      ms_config = is_config.
    ELSE.
      ms_config = VALUE ty_model_config(
        vocab_size        = 49152
        hidden_size       = 576
        intermediate_size = 1536
        num_layers        = 30
        num_heads         = 9
        num_kv_heads      = 3
        head_dim          = 64
        max_position      = 8192
        rms_norm_eps      = '1E-05'
        rope_theta        = '10000.0' ).
    ENDIF.

    " Initialize sampler
    mo_sampler = NEW zcl_llm_sampler( ).

    " Initialize transformer blocks
    init_blocks( ).

    mv_weights_loaded = abap_false.
  ENDMETHOD.

  METHOD init_blocks.
    DATA(ls_block_config) = VALUE zcl_llm_transformer_block=>ty_config(
      hidden_size       = ms_config-hidden_size
      intermediate_size = ms_config-intermediate_size
      num_heads         = ms_config-num_heads
      num_kv_heads      = ms_config-num_kv_heads
      head_dim          = ms_config-head_dim
      rms_norm_eps      = ms_config-rms_norm_eps ).

    DO ms_config-num_layers TIMES.
      DATA(lv_layer_idx) = sy-index - 1.
      DATA(lo_block) = NEW zcl_llm_transformer_block(
        iv_layer_index = lv_layer_idx
        is_config      = ls_block_config ).
      APPEND lo_block TO mt_blocks.
    ENDDO.
  ENDMETHOD.

  METHOD load_weights.
    "--------------------------------------------------------------------
    " Weight loading is source-dependent:
    " - 'ZTAB': Read from Z-table ZLLM_WEIGHTS (requires DB table setup)
    " - 'FILE': Read from application server file
    "
    " In this PoC, we initialize with random/zero weights for structure
    " validation. Real weight loading will populate from converted
    " SmolLM2-135M weights stored in the chosen format.
    "--------------------------------------------------------------------
    DATA(lv_hidden) = ms_config-hidden_size.
    DATA(lv_inter)  = ms_config-intermediate_size.
    DATA(lv_vocab)  = ms_config-vocab_size.
    DATA(lv_q_dim)  = ms_config-num_heads * ms_config-head_dim.
    DATA(lv_kv_dim) = ms_config-num_kv_heads * ms_config-head_dim.

    " Initialize embedding matrix (vocab_size × hidden_size)
    " In production this loads real weights; here zeros for structure
    mo_embed_tokens = zcl_llm_tensor=>create_zeros(
      VALUE ty_shape( ( lv_vocab ) ( lv_hidden ) ) ).

    " Initialize final RMSNorm weight
    mo_final_norm = zcl_llm_tensor=>create_zeros(
      VALUE ty_shape( ( lv_hidden ) ) ).

    " Initialize LM head (hidden_size × vocab_size)
    " Note: Many models tie embed_tokens and lm_head weights
    mo_lm_head = zcl_llm_tensor=>create_zeros(
      VALUE ty_shape( ( lv_hidden ) ( lv_vocab ) ) ).

    " Initialize weights for each transformer block
    DATA(lv_layer) = 0.
    LOOP AT mt_blocks INTO DATA(lo_block).
      " Per-layer weights — zeros for PoC structure
      lo_block->set_weights(
        io_input_norm = zcl_llm_tensor=>create_zeros(
                          VALUE ty_shape( ( lv_hidden ) ) )
        io_post_norm  = zcl_llm_tensor=>create_zeros(
                          VALUE ty_shape( ( lv_hidden ) ) )
        io_wq         = zcl_llm_tensor=>create_zeros(
                          VALUE ty_shape( ( lv_hidden ) ( lv_q_dim ) ) )
        io_wk         = zcl_llm_tensor=>create_zeros(
                          VALUE ty_shape( ( lv_hidden ) ( lv_kv_dim ) ) )
        io_wv         = zcl_llm_tensor=>create_zeros(
                          VALUE ty_shape( ( lv_hidden ) ( lv_kv_dim ) ) )
        io_wo         = zcl_llm_tensor=>create_zeros(
                          VALUE ty_shape( ( lv_q_dim ) ( lv_hidden ) ) )
        io_gate_proj  = zcl_llm_tensor=>create_zeros(
                          VALUE ty_shape( ( lv_hidden ) ( lv_inter ) ) )
        io_up_proj    = zcl_llm_tensor=>create_zeros(
                          VALUE ty_shape( ( lv_hidden ) ( lv_inter ) ) )
        io_down_proj  = zcl_llm_tensor=>create_zeros(
                          VALUE ty_shape( ( lv_inter ) ( lv_hidden ) ) ) ).
      lv_layer = lv_layer + 1.
    ENDLOOP.

    mv_weights_loaded = abap_true.

    " Log weight loading
    MESSAGE |LLM Engine: weights initialized ({ iv_source }/{ iv_path }), | &&
            |{ ms_config-num_layers } layers, { ms_config-hidden_size } hidden| TYPE 'I'.
  ENDMETHOD.

  METHOD set_tokenizer.
    mo_tokenizer = io_tokenizer.
  ENDMETHOD.

  METHOD embed_token.
    " Look up embedding vector for token_id from embedding matrix
    " embed_tokens is (vocab_size, hidden_size), we want row[token_id]
    DATA(lv_offset) = iv_token_id * ms_config-hidden_size.
    ro_embedding = mo_embed_tokens->slice(
      iv_start  = lv_offset
      iv_length = ms_config-hidden_size ).
  ENDMETHOD.

  METHOD init_kv_caches.
    DO ms_config-num_layers TIMES.
      APPEND VALUE ty_layer_kv_cache(
        layer_index = sy-index - 1 ) TO rt_kv_caches.
    ENDDO.
  ENDMETHOD.

  METHOD forward_pass.
    "--------------------------------------------------------------------
    " Full forward pass for one token:
    " 1. Embedding lookup
    " 2. Pass through all 30 transformer blocks
    " 3. Final RMSNorm
    " 4. LM head projection → logits
    "--------------------------------------------------------------------

    " Get the token to process (last in sequence for autoregressive)
    DATA(lv_token_id) = it_token_ids[ lines( it_token_ids ) ].

    " Step 1: Embedding lookup
    DATA(lo_hidden) = embed_token( lv_token_id ).

    " Step 2: Pass through transformer blocks
    DATA(lv_layer) = 0.
    LOOP AT mt_blocks INTO DATA(lo_block).
      " Get this layer's KV cache
      READ TABLE ct_kv_caches ASSIGNING FIELD-SYMBOL(<ls_kv_cache>)
        WITH KEY layer_index = lv_layer.
      IF sy-subrc <> 0.
        " Should not happen if caches are initialized
        ASSERT 1 = 0.
      ENDIF.

      lo_hidden = lo_block->forward(
        io_x        = lo_hidden
        iv_position = iv_position
        CHANGING ct_kv_cache = <ls_kv_cache>-kv_cache ).

      lv_layer = lv_layer + 1.
    ENDLOOP.

    " Step 3: Final RMSNorm
    lo_hidden = zcl_llm_math=>rms_norm(
      io_tensor = lo_hidden
      io_weight = mo_final_norm
      iv_eps    = ms_config-rms_norm_eps ).

    " Step 4: LM head — project hidden state to vocabulary logits
    " hidden (1, hidden_size) @ lm_head (hidden_size, vocab_size) → (1, vocab_size)
    DATA(lo_hidden_2d) = lo_hidden->reshape(
      VALUE ty_shape( ( 1 ) ( ms_config-hidden_size ) ) ).
    DATA(lo_logits_2d) = lo_hidden_2d->matmul( mo_lm_head ).

    " Reshape to 1D (vocab_size)
    ro_logits = lo_logits_2d->reshape(
      VALUE ty_shape( ( ms_config-vocab_size ) ) ).
  ENDMETHOD.

  METHOD generate.
    "--------------------------------------------------------------------
    " Autoregressive text generation:
    " 1. Tokenize the prompt
    " 2. Process prompt tokens (prefill)
    " 3. Generate new tokens one at a time
    " 4. Decode final token sequence to text
    "--------------------------------------------------------------------
    ASSERT mv_weights_loaded = abap_true.
    ASSERT mo_tokenizer IS BOUND.

    " Step 1: Tokenize prompt
    DATA(lt_token_ids) = mo_tokenizer->encode( iv_prompt ).

    " Step 2: Initialize KV caches
    DATA(lt_kv_caches) = init_kv_caches( ).

    " Step 3: Prefill — process all prompt tokens
    DATA(lv_pos) = 0.
    LOOP AT lt_token_ids INTO DATA(lv_token_id).
      " We need to run forward pass for each prompt token
      " to populate the KV cache
      DATA(lt_current) = VALUE zcl_llm_bpe_tokenizer=>ty_token_ids(
        ( lv_token_id ) ).
      DATA(lo_logits) = forward_pass(
        it_token_ids = lt_current
        iv_position  = lv_pos
        CHANGING ct_kv_caches = lt_kv_caches ).
      lv_pos = lv_pos + 1.
    ENDLOOP.

    " Step 4: Generate new tokens autoregressively
    DATA(lv_generated) = 0.
    WHILE lv_generated < iv_max_tokens.
      " Sample next token from logits
      DATA(lv_next_token) = mo_sampler->sample(
        io_logits      = lo_logits
        iv_temperature = iv_temperature
        iv_top_k       = iv_top_k
        iv_top_p       = iv_top_p ).

      " Append to sequence
      APPEND lv_next_token TO lt_token_ids.

      " Check for EOS token (token ID 0 or 2 typically)
      IF lv_next_token = 0 OR lv_next_token = 2.
        EXIT.
      ENDIF.

      " Forward pass for the new token
      lt_current = VALUE zcl_llm_bpe_tokenizer=>ty_token_ids(
        ( lv_next_token ) ).
      lo_logits = forward_pass(
        it_token_ids = lt_current
        iv_position  = lv_pos
        CHANGING ct_kv_caches = lt_kv_caches ).

      lv_pos = lv_pos + 1.
      lv_generated = lv_generated + 1.
    ENDWHILE.

    " Step 5: Decode all tokens back to text
    rv_result = mo_tokenizer->decode( lt_token_ids ).
  ENDMETHOD.

  METHOD get_config.
    rs_config = ms_config.
  ENDMETHOD.

ENDCLASS.
