"! <p class="shorttext synchronized" lang="en">LLM Engine - Grouped-Query Attention with KV Cache</p>
"! Implements Grouped-Query Attention (GQA) as used in SmolLM2/Llama.
"! Supports KV caching for efficient autoregressive inference.
"! Default config: 9 query heads, 3 KV heads (3:1 grouping), head_dim=64.
CLASS zcl_llm_attention DEFINITION
  PUBLIC
  CREATE PUBLIC.

  PUBLIC SECTION.
    TYPES:
      ty_float_tab TYPE zif_llm_tensor=>ty_float_tab,
      ty_shape     TYPE zif_llm_tensor=>ty_shape,

      "! KV cache entry for one position
      BEGIN OF ty_kv_entry,
        position TYPE i,
        k_data   TYPE ty_float_tab,  " (num_kv_heads * head_dim)
        v_data   TYPE ty_float_tab,  " (num_kv_heads * head_dim)
      END OF ty_kv_entry,
      ty_kv_cache TYPE STANDARD TABLE OF ty_kv_entry WITH KEY position.

    "! <p class="shorttext synchronized">Constructor</p>
    "! @parameter iv_num_heads | Number of query heads (default 9)
    "! @parameter iv_num_kv_heads | Number of KV heads for GQA (default 3)
    "! @parameter iv_head_dim | Dimension per head (default 64)
    "! @parameter iv_hidden_size | Total hidden size (default 576)
    METHODS constructor
      IMPORTING iv_num_heads    TYPE i DEFAULT 9
                iv_num_kv_heads TYPE i DEFAULT 3
                iv_head_dim     TYPE i DEFAULT 64
                iv_hidden_size  TYPE i DEFAULT 576.

    "! <p class="shorttext synchronized">Set weight tensors</p>
    "! @parameter io_wq | Query projection weights (hidden_size, num_heads*head_dim)
    "! @parameter io_wk | Key projection weights (hidden_size, num_kv_heads*head_dim)
    "! @parameter io_wv | Value projection weights (hidden_size, num_kv_heads*head_dim)
    "! @parameter io_wo | Output projection weights (num_heads*head_dim, hidden_size)
    METHODS set_weights
      IMPORTING io_wq TYPE REF TO zif_llm_tensor
                io_wk TYPE REF TO zif_llm_tensor
                io_wv TYPE REF TO zif_llm_tensor
                io_wo TYPE REF TO zif_llm_tensor.

    "! <p class="shorttext synchronized">Forward pass — compute attention output</p>
    "! @parameter io_x | Input tensor (hidden_size)
    "! @parameter iv_position | Current token position
    "! @parameter ct_kv_cache | KV cache (modified in-place)
    "! @parameter ro_output | Output tensor (hidden_size)
    METHODS forward
      IMPORTING io_x             TYPE REF TO zif_llm_tensor
                iv_position      TYPE i
      CHANGING  ct_kv_cache      TYPE ty_kv_cache
      RETURNING VALUE(ro_output) TYPE REF TO zif_llm_tensor.

  PRIVATE SECTION.
    DATA:
      mv_num_heads    TYPE i,
      mv_num_kv_heads TYPE i,
      mv_head_dim     TYPE i,
      mv_hidden_size  TYPE i,
      mv_heads_per_kv TYPE i,  " num_heads / num_kv_heads (GQA group size)
      mo_wq           TYPE REF TO zif_llm_tensor,
      mo_wk           TYPE REF TO zif_llm_tensor,
      mo_wv           TYPE REF TO zif_llm_tensor,
      mo_wo           TYPE REF TO zif_llm_tensor.

    "! Compute scaled dot-product attention for one query head
    METHODS compute_head_attention
      IMPORTING iv_head_idx      TYPE i
                io_q_head        TYPE REF TO zif_llm_tensor
                iv_position      TYPE i
                it_kv_cache      TYPE ty_kv_cache
      RETURNING VALUE(ro_output) TYPE REF TO zif_llm_tensor.

ENDCLASS.


CLASS zcl_llm_attention IMPLEMENTATION.

  METHOD constructor.
    mv_num_heads    = iv_num_heads.
    mv_num_kv_heads = iv_num_kv_heads.
    mv_head_dim     = iv_head_dim.
    mv_hidden_size  = iv_hidden_size.
    mv_heads_per_kv = iv_num_heads / iv_num_kv_heads.
  ENDMETHOD.

  METHOD set_weights.
    mo_wq = io_wq.
    mo_wk = io_wk.
    mo_wv = io_wv.
    mo_wo = io_wo.
  ENDMETHOD.

  METHOD forward.
    "--------------------------------------------------------------------
    " GQA Forward Pass:
    " 1. Project input to Q, K, V
    " 2. Apply RoPE to Q and K
    " 3. Store K, V in cache
    " 4. For each query head, compute attention using its KV group
    " 5. Concatenate heads and project output
    "--------------------------------------------------------------------

    " Step 1: Linear projections
    " x (hidden_size) @ Wq (hidden_size, q_dim) → Q (q_dim)
    " where q_dim = num_heads * head_dim
    DATA(lo_q) = io_x->matmul( mo_wq ).
    DATA(lo_k) = io_x->matmul( mo_wk ).
    DATA(lo_v) = io_x->matmul( mo_wv ).

    " Step 2: Apply RoPE to Q and K
    zcl_llm_math=>rope(
      io_q        = lo_q
      io_k        = lo_k
      iv_position = iv_position
      iv_head_dim = mv_head_dim ).

    " Step 3: Store K, V in cache for this position
    DATA(ls_kv_entry) = VALUE ty_kv_entry(
      position = iv_position
      k_data   = lo_k->get_data( )
      v_data   = lo_v->get_data( ) ).
    INSERT ls_kv_entry INTO TABLE ct_kv_cache.

    " Step 4: Compute attention for each query head
    DATA(lt_head_outputs) = VALUE ty_float_tab( ).
    DATA(lv_scale) = 1 / sqrt( CONV f( mv_head_dim ) ).

    DATA(lv_h) = 0.
    WHILE lv_h < mv_num_heads.
      " Extract this head's query: Q[h*head_dim : (h+1)*head_dim]
      DATA(lo_q_head) = lo_q->slice(
        iv_start  = lv_h * mv_head_dim
        iv_length = mv_head_dim ).

      " Determine which KV head group this query head uses
      DATA(lv_kv_head) = lv_h / mv_heads_per_kv.

      " Compute attention scores against all cached positions
      DATA(lt_scores) = VALUE ty_float_tab( ).
      DATA(lv_seq_len) = lines( ct_kv_cache ).

      " Collect scores for all cached positions
      DATA(lt_q_data) = lo_q_head->get_data( ).

      LOOP AT ct_kv_cache INTO DATA(ls_cached) WHERE position <= iv_position.
        " Extract K for this KV head from cached data
        DATA(lv_k_offset) = lv_kv_head * mv_head_dim.
        DATA(lv_score) = CONV f( 0 ).
        DATA(lv_d) = 0.
        WHILE lv_d < mv_head_dim.
          lv_score = lv_score + lt_q_data[ lv_d + 1 ] *
                     ls_cached-k_data[ lv_k_offset + lv_d + 1 ].
          lv_d = lv_d + 1.
        ENDWHILE.
        lv_score = lv_score * lv_scale.

        " Causal mask: only attend to positions <= current
        APPEND lv_score TO lt_scores.
      ENDLOOP.

      " Softmax over attention scores
      DATA(lo_scores_tensor) = zcl_llm_tensor=>create_from_float_table(
        it_data  = lt_scores
        it_shape = VALUE ty_shape( ( lines( lt_scores ) ) ) ).
      DATA(lo_attn_weights) = zcl_llm_math=>softmax( lo_scores_tensor ).
      DATA(lt_weights) = lo_attn_weights->get_data( ).

      " Weighted sum of V vectors
      DATA(lt_head_out) = VALUE ty_float_tab( ).
      DO mv_head_dim TIMES.
        APPEND CONV f( 0 ) TO lt_head_out.
      ENDDO.

      DATA(lv_w_idx) = 1.
      LOOP AT ct_kv_cache INTO ls_cached WHERE position <= iv_position.
        DATA(lv_v_offset) = lv_kv_head * mv_head_dim.
        DATA(lv_weight) = lt_weights[ lv_w_idx ].
        lv_d = 0.
        WHILE lv_d < mv_head_dim.
          lt_head_out[ lv_d + 1 ] = lt_head_out[ lv_d + 1 ] +
            lv_weight * ls_cached-v_data[ lv_v_offset + lv_d + 1 ].
          lv_d = lv_d + 1.
        ENDWHILE.
        lv_w_idx = lv_w_idx + 1.
      ENDLOOP.

      " Append this head's output to concatenated result
      APPEND LINES OF lt_head_out TO lt_head_outputs.
      lv_h = lv_h + 1.
    ENDWHILE.

    " Step 5: Output projection — concat(heads) @ Wo
    DATA(lo_concat) = zcl_llm_tensor=>create_from_float_table(
      it_data  = lt_head_outputs
      it_shape = VALUE ty_shape( ( 1 ) ( mv_num_heads * mv_head_dim ) ) ).
    ro_output = lo_concat->matmul( mo_wo ).

    " Reshape to 1D (hidden_size)
    ro_output = ro_output->reshape( VALUE ty_shape( ( mv_hidden_size ) ) ).
  ENDMETHOD.

  METHOD compute_head_attention.
    " Kept as placeholder — actual computation is inlined in forward()
    " for performance (avoids repeated method call overhead per head).
    ro_output = zcl_llm_tensor=>create_zeros( VALUE ty_shape( ( mv_head_dim ) ) ).
  ENDMETHOD.

ENDCLASS.
