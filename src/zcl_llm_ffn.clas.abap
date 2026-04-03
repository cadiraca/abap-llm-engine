"! <p class="shorttext synchronized" lang="en">LLM Engine - SiLU-Gated Feed-Forward Network</p>
"! Implements the SwiGLU / SiLU-gated FFN used in Llama architecture:
"!   output = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
"! Default dimensions match SmolLM2-135M: 576 → 1536 → 576.
CLASS zcl_llm_ffn DEFINITION
  PUBLIC
  CREATE PUBLIC.

  PUBLIC SECTION.
    TYPES:
      ty_float_tab TYPE zif_llm_tensor=>ty_float_tab,
      ty_shape     TYPE zif_llm_tensor=>ty_shape.

    "! <p class="shorttext synchronized">Constructor</p>
    "! @parameter iv_hidden_size | Model hidden dimension (default 576)
    "! @parameter iv_intermediate_size | FFN intermediate dimension (default 1536)
    METHODS constructor
      IMPORTING iv_hidden_size       TYPE i DEFAULT 576
                iv_intermediate_size TYPE i DEFAULT 1536.

    "! <p class="shorttext synchronized">Set weight tensors</p>
    "! @parameter io_gate_proj | Gate projection (hidden → intermediate)
    "! @parameter io_up_proj | Up projection (hidden → intermediate)
    "! @parameter io_down_proj | Down projection (intermediate → hidden)
    METHODS set_weights
      IMPORTING io_gate_proj TYPE REF TO zif_llm_tensor
                io_up_proj   TYPE REF TO zif_llm_tensor
                io_down_proj TYPE REF TO zif_llm_tensor.

    "! <p class="shorttext synchronized">Forward pass</p>
    "! Computes: down_proj( SiLU(gate_proj(x)) ⊙ up_proj(x) )
    "! @parameter io_x | Input tensor (hidden_size)
    "! @parameter ro_output | Output tensor (hidden_size)
    METHODS forward
      IMPORTING io_x             TYPE REF TO zif_llm_tensor
      RETURNING VALUE(ro_output) TYPE REF TO zif_llm_tensor.

  PRIVATE SECTION.
    DATA:
      mv_hidden_size       TYPE i,
      mv_intermediate_size TYPE i,
      mo_gate_proj         TYPE REF TO zif_llm_tensor,
      mo_up_proj           TYPE REF TO zif_llm_tensor,
      mo_down_proj         TYPE REF TO zif_llm_tensor.

ENDCLASS.


CLASS zcl_llm_ffn IMPLEMENTATION.

  METHOD constructor.
    mv_hidden_size       = iv_hidden_size.
    mv_intermediate_size = iv_intermediate_size.
  ENDMETHOD.

  METHOD set_weights.
    mo_gate_proj = io_gate_proj.
    mo_up_proj   = io_up_proj.
    mo_down_proj = io_down_proj.
  ENDMETHOD.

  METHOD forward.
    "--------------------------------------------------------------------
    " SwiGLU Feed-Forward:
    " 1. gate = x @ gate_proj           → (intermediate_size)
    " 2. up   = x @ up_proj             → (intermediate_size)
    " 3. gate = SiLU(gate)              → element-wise activation
    " 4. hidden = gate ⊙ up             → element-wise multiply
    " 5. output = hidden @ down_proj    → (hidden_size)
    "--------------------------------------------------------------------

    " Reshape input to (1, hidden_size) for matmul
    DATA(lo_x_2d) = io_x->reshape( VALUE ty_shape(
      ( 1 ) ( mv_hidden_size ) ) ).

    " Gate projection: x → intermediate
    DATA(lo_gate) = lo_x_2d->matmul( mo_gate_proj ).

    " Up projection: x → intermediate
    DATA(lo_up) = lo_x_2d->matmul( mo_up_proj ).

    " Apply SiLU to gate
    DATA(lt_gate_data) = lo_gate->get_data( ).
    LOOP AT lt_gate_data ASSIGNING FIELD-SYMBOL(<lv_g>).
      <lv_g> = zcl_llm_math=>silu( <lv_g> ).
    ENDLOOP.

    DATA(lo_gate_activated) = zcl_llm_tensor=>create_from_float_table(
      it_data  = lt_gate_data
      it_shape = VALUE ty_shape( ( 1 ) ( mv_intermediate_size ) ) ).

    " Element-wise multiply: SiLU(gate) ⊙ up
    DATA(lo_hidden) = lo_gate_activated->multiply_elementwise( lo_up ).

    " Down projection: intermediate → hidden
    DATA(lo_out) = lo_hidden->matmul( mo_down_proj ).

    " Reshape back to 1D
    ro_output = lo_out->reshape( VALUE ty_shape( ( mv_hidden_size ) ) ).
  ENDMETHOD.

ENDCLASS.
