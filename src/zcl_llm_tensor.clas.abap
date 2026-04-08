"! <p class="shorttext synchronized" lang="en">LLM Engine - Tensor Implementation</p>
"! Multi-dimensional tensor backed by a flat ABAP float table.
"! Supports basic linear algebra required for transformer inference:
"! matrix multiplication (tiled), element-wise ops, reshape, slice.
CLASS zcl_llm_tensor DEFINITION
  PUBLIC
  CREATE PUBLIC.

  PUBLIC SECTION.
    INTERFACES zif_llm_tensor.

    ALIASES:
      ty_shape    FOR zif_llm_tensor~ty_shape,
      ty_float_tab FOR zif_llm_tensor~ty_float_tab.

    "! <p class="shorttext synchronized">Create tensor from float table and shape</p>
    "! @parameter it_data | Float values (row-major)
    "! @parameter it_shape | Dimensions
    "! @parameter ro_tensor | New tensor instance
    CLASS-METHODS create_from_float_table
      IMPORTING it_data          TYPE ty_float_tab
                it_shape         TYPE ty_shape
      RETURNING VALUE(ro_tensor) TYPE REF TO zif_llm_tensor.

    "! <p class="shorttext synchronized">Create zero-filled tensor</p>
    "! @parameter it_shape | Dimensions
    "! @parameter ro_tensor | New tensor instance
    CLASS-METHODS create_zeros
      IMPORTING it_shape         TYPE ty_shape
      RETURNING VALUE(ro_tensor) TYPE REF TO zif_llm_tensor.

    "! <p class="shorttext synchronized">Create uninitialized tensor with given shape</p>
    "! @parameter it_shape | Dimensions
    "! @parameter ro_tensor | New tensor instance
    CLASS-METHODS create_from_shape
      IMPORTING it_shape         TYPE ty_shape
      RETURNING VALUE(ro_tensor) TYPE REF TO zif_llm_tensor.

  PROTECTED SECTION.

  PRIVATE SECTION.
    CONSTANTS:
      "! Tile/block size for tiled matrix multiplication
      c_tile_size TYPE i VALUE 32.

    DATA:
      mt_data  TYPE ty_float_tab,
      mt_shape TYPE ty_shape,
      mv_size  TYPE i.

    "! Calculate total size from shape
    METHODS calc_size
      IMPORTING it_shape       TYPE ty_shape
      RETURNING VALUE(rv_size) TYPE i.

ENDCLASS.


CLASS zcl_llm_tensor IMPLEMENTATION.

  METHOD zif_llm_tensor~get_shape.
    rt_shape = mt_shape.
  ENDMETHOD.

  METHOD zif_llm_tensor~get_size.
    rv_size = mv_size.
  ENDMETHOD.

  METHOD zif_llm_tensor~get_data.
    rt_data = mt_data.
  ENDMETHOD.

  METHOD zif_llm_tensor~get_data_ref.
    GET REFERENCE OF mt_data INTO rr_data.
  ENDMETHOD.

  METHOD zif_llm_tensor~set_value.
    " iv_index is 0-based, internal table is 1-based
    mt_data[ iv_index + 1 ] = iv_value.
  ENDMETHOD.

  METHOD zif_llm_tensor~get_value.
    rv_value = mt_data[ iv_index + 1 ].
  ENDMETHOD.

  METHOD calc_size.
    rv_size = 1.
    LOOP AT it_shape INTO DATA(lv_dim).
      rv_size = rv_size * lv_dim.
    ENDLOOP.
  ENDMETHOD.

  METHOD create_from_float_table.
    DATA(lo_tensor) = NEW zcl_llm_tensor( ).
    lo_tensor->mt_data  = it_data.
    lo_tensor->mt_shape = it_shape.
    lo_tensor->mv_size  = lo_tensor->calc_size( it_shape ).
    ro_tensor = lo_tensor.
  ENDMETHOD.

  METHOD create_zeros.
    DATA(lo_tensor) = NEW zcl_llm_tensor( ).
    lo_tensor->mt_shape = it_shape.
    lo_tensor->mv_size  = lo_tensor->calc_size( it_shape ).
    " Pre-fill with zeros
    DO lo_tensor->mv_size TIMES.
      APPEND CONV f( 0 ) TO lo_tensor->mt_data.
    ENDDO.
    ro_tensor = lo_tensor.
  ENDMETHOD.

  METHOD create_from_shape.
    " Alias for create_zeros — same semantics in this PoC
    ro_tensor = create_zeros( it_shape ).
  ENDMETHOD.

  METHOD zif_llm_tensor~matmul.
    "--------------------------------------------------------------------
    " Matrix multiplication with tiled (blocked) algorithm.
    " this(M,K) @ other(K,N) = result(M,N)
    " For 1D tensors, treats them as row/column vectors.
    "--------------------------------------------------------------------
    DATA: lv_m     TYPE i,
          lv_k     TYPE i,
          lv_n     TYPE i,
          lv_ii    TYPE i,
          lv_jj    TYPE i,
          lv_kk    TYPE i,
          lv_i     TYPE i,
          lv_j     TYPE i,
          lv_k_idx TYPE i,
          lv_i_max TYPE i,
          lv_j_max TYPE i,
          lv_k_max TYPE i,
          lv_sum   TYPE f,
          lv_val_a TYPE f,
          lv_val_b TYPE f.

    DATA(lt_shape_a) = mt_shape.
    DATA(lt_shape_b) = io_other->get_shape( ).

    " Determine dimensions: support 1D and 2D tensors
    IF lines( lt_shape_a ) = 1.
      lv_m = 1.
      lv_k = lt_shape_a[ 1 ].
    ELSE.
      lv_m = lt_shape_a[ 1 ].
      lv_k = lt_shape_a[ 2 ].
    ENDIF.

    IF lines( lt_shape_b ) = 1.
      lv_n = 1.
    ELSE.
      lv_n = lt_shape_b[ 2 ].
    ENDIF.

    " Create result tensor
    DATA(lt_result_shape) = VALUE ty_shape( ( lv_m ) ( lv_n ) ).
    DATA(lo_result) = CAST zcl_llm_tensor( create_zeros( lt_result_shape ) ).

    " Get references for faster access
    DATA(lr_data_a) = zif_llm_tensor~get_data_ref( ).
    DATA(lr_data_b) = CAST zcl_llm_tensor( io_other )->zif_llm_tensor~get_data_ref( ).
    DATA(lr_data_r) = lo_result->zif_llm_tensor~get_data_ref( ).

    " Tiled matrix multiplication for cache efficiency
    " Iterate over tiles
    lv_ii = 0.
    WHILE lv_ii < lv_m.
      lv_i_max = nmin( val1 = lv_ii + c_tile_size  val2 = lv_m ).

      lv_kk = 0.
      WHILE lv_kk < lv_k.
        lv_k_max = nmin( val1 = lv_kk + c_tile_size  val2 = lv_k ).

        lv_jj = 0.
        WHILE lv_jj < lv_n.
          lv_j_max = nmin( val1 = lv_jj + c_tile_size  val2 = lv_n ).

          " Inner tile computation
          lv_i = lv_ii.
          WHILE lv_i < lv_i_max.
            lv_j = lv_jj.
            WHILE lv_j < lv_j_max.
              " Accumulate dot-product for this (i,j) element
              lv_sum = lr_data_r->*[ lv_i * lv_n + lv_j + 1 ].
              lv_k_idx = lv_kk.
              WHILE lv_k_idx < lv_k_max.
                lv_val_a = lr_data_a->*[ lv_i * lv_k + lv_k_idx + 1 ].
                lv_val_b = lr_data_b->*[ lv_k_idx * lv_n + lv_j + 1 ].
                lv_sum = lv_sum + lv_val_a * lv_val_b.
                lv_k_idx = lv_k_idx + 1.
              ENDWHILE.
              lr_data_r->*[ lv_i * lv_n + lv_j + 1 ] = lv_sum.
              lv_j = lv_j + 1.
            ENDWHILE.
            lv_i = lv_i + 1.
          ENDWHILE.

          lv_jj = lv_jj + c_tile_size.
        ENDWHILE.

        lv_kk = lv_kk + c_tile_size.
      ENDWHILE.

      lv_ii = lv_ii + c_tile_size.
    ENDWHILE.

    ro_result = lo_result.
  ENDMETHOD.

  METHOD zif_llm_tensor~add.
    DATA(lt_other_data) = io_other->get_data( ).
    DATA(lt_result) = mt_data.
    DATA(lv_idx) = 1.

    LOOP AT lt_result ASSIGNING FIELD-SYMBOL(<lv_val>).
      <lv_val> = <lv_val> + lt_other_data[ lv_idx ].
      lv_idx = lv_idx + 1.
    ENDLOOP.

    ro_result = create_from_float_table(
      it_data  = lt_result
      it_shape = mt_shape ).
  ENDMETHOD.

  METHOD zif_llm_tensor~multiply_elementwise.
    DATA(lt_other_data) = io_other->get_data( ).
    DATA(lt_result) = mt_data.
    DATA(lv_idx) = 1.

    LOOP AT lt_result ASSIGNING FIELD-SYMBOL(<lv_val>).
      <lv_val> = <lv_val> * lt_other_data[ lv_idx ].
      lv_idx = lv_idx + 1.
    ENDLOOP.

    ro_result = create_from_float_table(
      it_data  = lt_result
      it_shape = mt_shape ).
  ENDMETHOD.

  METHOD zif_llm_tensor~scale.
    DATA(lt_result) = mt_data.

    LOOP AT lt_result ASSIGNING FIELD-SYMBOL(<lv_val>).
      <lv_val> = <lv_val> * iv_factor.
    ENDLOOP.

    ro_result = create_from_float_table(
      it_data  = lt_result
      it_shape = mt_shape ).
  ENDMETHOD.

  METHOD zif_llm_tensor~reshape.
    " Validate total size matches
    DATA(lv_new_size) = calc_size( it_shape ).
    ASSERT lv_new_size = mv_size.

    ro_result = create_from_float_table(
      it_data  = mt_data
      it_shape = it_shape ).
  ENDMETHOD.

  METHOD zif_llm_tensor~slice.
    DATA(lt_result) = VALUE ty_float_tab( ).
    DATA(lv_end) = iv_start + iv_length.

    DATA(lv_i) = iv_start.
    WHILE lv_i < lv_end.
      APPEND mt_data[ lv_i + 1 ] TO lt_result.
      lv_i = lv_i + 1.
    ENDWHILE.

    ro_result = create_from_float_table(
      it_data  = lt_result
      it_shape = VALUE ty_shape( ( iv_length ) ) ).
  ENDMETHOD.

ENDCLASS.
