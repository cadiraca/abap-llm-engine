"! <p class="shorttext synchronized" lang="en">LLM Engine - Math Utility Functions</p>
"! Static utility class providing activation functions, normalization,
"! and positional encoding operations for transformer inference.
CLASS zcl_llm_math DEFINITION
  PUBLIC
  FINAL
  CREATE PRIVATE.

  PUBLIC SECTION.
    TYPES:
      ty_shape     TYPE zif_llm_tensor=>ty_shape,
      ty_float_tab TYPE zif_llm_tensor=>ty_float_tab.

    "! <p class="shorttext synchronized">SiLU activation function</p>
    "! SiLU(x) = x * sigmoid(x). Used in Llama-style FFN gating.
    "! @parameter iv_x | Input value
    "! @parameter rv_result | SiLU(x)
    CLASS-METHODS silu
      IMPORTING iv_x             TYPE f
      RETURNING VALUE(rv_result) TYPE f.

    "! <p class="shorttext synchronized">Sigmoid function</p>
    "! sigmoid(x) = 1 / (1 + exp(-x))
    "! @parameter iv_x | Input value
    "! @parameter rv_result | sigmoid(x)
    CLASS-METHODS sigmoid
      IMPORTING iv_x             TYPE f
      RETURNING VALUE(rv_result) TYPE f.

    "! <p class="shorttext synchronized">Softmax over a 1D tensor</p>
    "! Numerically stable: subtracts max before exp.
    "! @parameter io_tensor | Input tensor (1D)
    "! @parameter ro_result | Softmax output tensor (1D, same shape)
    CLASS-METHODS softmax
      IMPORTING io_tensor        TYPE REF TO zif_llm_tensor
      RETURNING VALUE(ro_result) TYPE REF TO zif_llm_tensor.

    "! <p class="shorttext synchronized">RMS Normalization</p>
    "! RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * weight
    "! @parameter io_tensor | Input tensor
    "! @parameter io_weight | Normalization weight tensor
    "! @parameter iv_eps | Epsilon for numerical stability (default 1e-5)
    "! @parameter ro_result | Normalized tensor
    CLASS-METHODS rms_norm
      IMPORTING io_tensor        TYPE REF TO zif_llm_tensor
                io_weight        TYPE REF TO zif_llm_tensor
                iv_eps           TYPE f DEFAULT '1E-05'
      RETURNING VALUE(ro_result) TYPE REF TO zif_llm_tensor.

    "! <p class="shorttext synchronized">Apply Rotary Position Embeddings (RoPE)</p>
    "! Modifies Q and K tensors in-place to encode position information.
    "! Uses the standard Llama RoPE formulation with configurable theta.
    "! @parameter io_q | Query tensor (modified in-place)
    "! @parameter io_k | Key tensor (modified in-place)
    "! @parameter iv_position | Token position index
    "! @parameter iv_head_dim | Dimension per head
    "! @parameter iv_theta | RoPE base frequency (default 10000.0)
    CLASS-METHODS rope
      IMPORTING io_q        TYPE REF TO zif_llm_tensor
                io_k        TYPE REF TO zif_llm_tensor
                iv_position TYPE i
                iv_head_dim TYPE i
                iv_theta    TYPE f DEFAULT '10000.0'.

  PRIVATE SECTION.
    CONSTANTS:
      "! Clamp range for exp() to prevent overflow
      c_exp_clamp TYPE f VALUE '88.0'.

ENDCLASS.


CLASS zcl_llm_math IMPLEMENTATION.

  METHOD sigmoid.
    DATA(lv_x) = iv_x.
    " Clamp to prevent overflow in exp()
    IF lv_x < - c_exp_clamp.
      lv_x = - c_exp_clamp.
    ELSEIF lv_x > c_exp_clamp.
      lv_x = c_exp_clamp.
    ENDIF.

    rv_result = 1 / ( 1 + exp( - lv_x ) ).
  ENDMETHOD.

  METHOD silu.
    " SiLU(x) = x * sigmoid(x)
    rv_result = iv_x * sigmoid( iv_x ).
  ENDMETHOD.

  METHOD softmax.
    DATA(lt_data) = io_tensor->get_data( ).
    DATA(lv_size) = io_tensor->get_size( ).

    " Find max for numerical stability
    DATA(lv_max) = CONV f( '-1E38' ).
    LOOP AT lt_data INTO DATA(lv_val).
      IF lv_val > lv_max.
        lv_max = lv_val.
      ENDIF.
    ENDLOOP.

    " Compute exp(x - max) and sum
    DATA(lv_sum) = CONV f( 0 ).
    DATA(lt_result) = VALUE ty_float_tab( ).

    LOOP AT lt_data INTO lv_val.
      DATA(lv_exp) = exp( lv_val - lv_max ).
      APPEND lv_exp TO lt_result.
      lv_sum = lv_sum + lv_exp.
    ENDLOOP.

    " Normalize
    LOOP AT lt_result ASSIGNING FIELD-SYMBOL(<lv_r>).
      <lv_r> = <lv_r> / lv_sum.
    ENDLOOP.

    ro_result = zcl_llm_tensor=>create_from_float_table(
      it_data  = lt_result
      it_shape = io_tensor->get_shape( ) ).
  ENDMETHOD.

  METHOD rms_norm.
    DATA(lt_data)   = io_tensor->get_data( ).
    DATA(lt_weight) = io_weight->get_data( ).
    DATA(lv_size)   = io_tensor->get_size( ).

    " Compute mean of squares
    DATA(lv_sum_sq) = CONV f( 0 ).
    LOOP AT lt_data INTO DATA(lv_val).
      lv_sum_sq = lv_sum_sq + lv_val * lv_val.
    ENDLOOP.

    DATA(lv_mean_sq) = lv_sum_sq / lv_size.
    DATA(lv_rms_inv) = CONV f( 1 / sqrt( lv_mean_sq + iv_eps ) ).

    " Normalize and scale by weight
    DATA(lt_result) = VALUE ty_float_tab( ).
    DATA(lv_idx) = 1.

    LOOP AT lt_data INTO lv_val.
      DATA(lv_normalized) = lv_val * lv_rms_inv * lt_weight[ lv_idx ].
      APPEND lv_normalized TO lt_result.
      lv_idx = lv_idx + 1.
    ENDLOOP.

    ro_result = zcl_llm_tensor=>create_from_float_table(
      it_data  = lt_result
      it_shape = io_tensor->get_shape( ) ).
  ENDMETHOD.

  METHOD rope.
    "--------------------------------------------------------------------
    " Apply Rotary Position Embeddings to Q and K tensors.
    " For each pair of dimensions (2i, 2i+1):
    "   freq = 1 / (theta ^ (2i / head_dim))
    "   angle = position * freq
    "   (x0, x1) → (x0*cos - x1*sin, x0*sin + x1*cos)
    "--------------------------------------------------------------------
    DATA: lv_freq  TYPE f,
          lv_angle TYPE f,
          lv_cos   TYPE f,
          lv_sin   TYPE f,
          lv_q0    TYPE f,
          lv_q1    TYPE f,
          lv_k0    TYPE f,
          lv_k1    TYPE f.

    DATA(lo_q) = CAST zcl_llm_tensor( io_q ).
    DATA(lo_k) = CAST zcl_llm_tensor( io_k ).
    DATA(lv_q_size) = io_q->get_size( ).
    DATA(lv_k_size) = io_k->get_size( ).

    " Process Q tensor — may have multiple heads
    DATA(lv_num_q_heads) = lv_q_size / iv_head_dim.
    DATA(lv_head) = 0.
    WHILE lv_head < lv_num_q_heads.
      DATA(lv_base_q) = lv_head * iv_head_dim.
      DATA(lv_i) = 0.
      WHILE lv_i < iv_head_dim - 1.
        lv_freq  = 1 / iv_theta ** ( CONV f( lv_i ) / iv_head_dim ).
        lv_angle = iv_position * lv_freq.
        lv_cos   = cos( lv_angle ).
        lv_sin   = sin( lv_angle ).

        lv_q0 = lo_q->get_value( lv_base_q + lv_i ).
        lv_q1 = lo_q->get_value( lv_base_q + lv_i + 1 ).
        lo_q->set_value( iv_index = lv_base_q + lv_i
                         iv_value = lv_q0 * lv_cos - lv_q1 * lv_sin ).
        lo_q->set_value( iv_index = lv_base_q + lv_i + 1
                         iv_value = lv_q0 * lv_sin + lv_q1 * lv_cos ).

        lv_i = lv_i + 2.
      ENDWHILE.
      lv_head = lv_head + 1.
    ENDWHILE.

    " Process K tensor — may have fewer heads (GQA)
    DATA(lv_num_k_heads) = lv_k_size / iv_head_dim.
    lv_head = 0.
    WHILE lv_head < lv_num_k_heads.
      DATA(lv_base_k) = lv_head * iv_head_dim.
      lv_i = 0.
      WHILE lv_i < iv_head_dim - 1.
        lv_freq  = 1 / iv_theta ** ( CONV f( lv_i ) / iv_head_dim ).
        lv_angle = iv_position * lv_freq.
        lv_cos   = cos( lv_angle ).
        lv_sin   = sin( lv_angle ).

        lv_k0 = lo_k->get_value( lv_base_k + lv_i ).
        lv_k1 = lo_k->get_value( lv_base_k + lv_i + 1 ).
        lo_k->set_value( iv_index = lv_base_k + lv_i
                         iv_value = lv_k0 * lv_cos - lv_k1 * lv_sin ).
        lo_k->set_value( iv_index = lv_base_k + lv_i + 1
                         iv_value = lv_k0 * lv_sin + lv_k1 * lv_cos ).

        lv_i = lv_i + 2.
      ENDWHILE.
      lv_head = lv_head + 1.
    ENDWHILE.
  ENDMETHOD.

ENDCLASS.
