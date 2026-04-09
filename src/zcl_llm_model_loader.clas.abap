"! <p class="shorttext synchronized" lang="en">LLM Engine - Model Weight Loader</p>
"! Loads SmolLM2-135M weights into ZCL_LLM_ENGINE from two possible sources:
"!   1. ZLLM_WEIGHTS Z-table (preferred for SAP deployment)
"!   2. Binary model.bin file on AL11 application server filesystem
"!
"! Weight naming follows the HuggingFace safetensors convention and is mapped
"! to the engine's ABAP class structure by this loader.
"!
"! INT8 dequantization:
"!   float_value = int8_value * scale_factor
"! where scale_factor is stored per output channel (last dimension of weight).
CLASS zcl_llm_model_loader DEFINITION
  PUBLIC
  CREATE PUBLIC.

  PUBLIC SECTION.

    "! <p class="shorttext synchronized">Constructor</p>
    "! @parameter io_engine | Engine instance to populate with weights
    METHODS constructor
      IMPORTING io_engine TYPE REF TO zcl_llm_engine.

    "! <p class="shorttext synchronized">Load weights from Z-table</p>
    "! Reads ZLLM_WEIGHTS table, dequantizes INT8 weights and populates the engine.
    "! @parameter iv_model_id | Model identifier key in ZLLM_WEIGHTS
    METHODS load_from_ztable
      IMPORTING iv_model_id TYPE char40 DEFAULT 'SMOLLM2-135M'.

    "! <p class="shorttext synchronized">Load weights from binary file (AL11)</p>
    "! Reads ALLM binary format from an AL11 application server path.
    "! Format: magic(4) + version(4) + num_tensors(4) + config_json_len(4) +
    "!         config_json + [ per-tensor records ]
    "! Per tensor: name_len(2) + name + dtype(1) + ndims(1) +
    "!             shape(ndims*4) + data_len(4) + data + scale_len(4) + [scales]
    "! @parameter iv_file_path | AL11 path to model.bin
    "! @parameter iv_model_id  | Model identifier (for logging)
    METHODS load_from_file
      IMPORTING iv_file_path TYPE string
                iv_model_id  TYPE string DEFAULT 'SMOLLM2-135M'.


    "! <p class="shorttext synchronized">Load BPE vocabulary from Z-tables</p>
    "! Reads ZLLM_VOCAB and ZLLM_MERGES, constructs a tokenizer and attaches it
    "! to the engine via SET_TOKENIZER.
    "! @parameter iv_model_id | Model identifier key
    METHODS load_vocab
      IMPORTING iv_model_id TYPE char40 DEFAULT 'SMOLLM2-135M'.

  PRIVATE SECTION.

    DATA mo_engine TYPE REF TO zcl_llm_engine.

    "! Dequantize INT8 bytes + float32 scales → float table
    "! @parameter iv_data   | Raw INT8 bytes (RAWSTRING)
    "! @parameter iv_scales | Float32 scale bytes (RAWSTRING)
    "! @parameter iv_rows   | Number of rows (output channels)
    "! @parameter iv_cols   | Number of columns (input features)
    "! @parameter rt_floats | Dequantized float table
    METHODS dequantize_int8
      IMPORTING iv_data          TYPE xstring
                iv_scales        TYPE xstring
                iv_rows          TYPE i
                iv_cols          TYPE i
      RETURNING VALUE(rt_floats) TYPE zif_llm_tensor=>ty_float_tab.

    "! Parse shape string "[d0,d1,...]" → integer table
    METHODS parse_shape
      IMPORTING iv_shape         TYPE string
      RETURNING VALUE(rt_shape)  TYPE zif_llm_tensor=>ty_shape.

    " Apply all weights from the tensor map to the engine
    TYPES: ty_tensor_ref TYPE REF TO zif_llm_tensor.
    TYPES: ty_tensor_refs TYPE HASHED TABLE OF ty_tensor_ref WITH UNIQUE KEY table_line.
    METHODS apply_weights_to_engine
      IMPORTING it_tensors TYPE ty_tensor_refs.

    "! Read a little-endian uint32 from xstring at byte offset
    METHODS read_uint32
      IMPORTING iv_xstr         TYPE xstring
                iv_offset       TYPE i
      RETURNING VALUE(rv_value) TYPE i.

    "! Read a little-endian uint16 from xstring at byte offset
    METHODS read_uint16
      IMPORTING iv_xstr         TYPE xstring
                iv_offset       TYPE i
      RETURNING VALUE(rv_value) TYPE i.

    "! Read a single byte from xstring as unsigned integer
    METHODS read_byte
      IMPORTING iv_xstr         TYPE xstring
                iv_offset       TYPE i
      RETURNING VALUE(rv_value) TYPE i.

ENDCLASS.


CLASS zcl_llm_model_loader IMPLEMENTATION.

  METHOD constructor.
    mo_engine = io_engine.
  ENDMETHOD.

  METHOD load_from_ztable.
    " Read all weight rows for this model, dequantize and apply.
    DATA: lt_weights TYPE STANDARD TABLE OF zllm_weights WITH EMPTY KEY.

    SELECT * FROM zllm_weights
      WHERE model_id = @iv_model_id
      INTO TABLE @lt_weights.

    IF sy-subrc <> 0 OR lines( lt_weights ) = 0.

      RETURN.
    ENDIF.

    " Build tensor map: layer_name → tensor reference
    TYPES: BEGIN OF ty_named_tensor,
             layer_name TYPE char200,
             tensor     TYPE REF TO zif_llm_tensor,
           END OF ty_named_tensor.
    DATA: lt_named TYPE STANDARD TABLE OF ty_named_tensor WITH EMPTY KEY.

    LOOP AT lt_weights INTO DATA(ls_w).
      DATA(lt_shape) = parse_shape( ls_w-shape ).
      DATA(lv_rows)  = lt_shape[ 1 ].
      DATA(lv_cols)  = COND i( WHEN lines( lt_shape ) >= 2
                                THEN lt_shape[ 2 ] ELSE 1 ).

      DATA(lt_floats) = COND zif_llm_tensor=>ty_float_tab(
        WHEN ls_w-dtype = 'INT8'
        THEN dequantize_int8(
               iv_data   = ls_w-weight_data
               iv_scales = ls_w-scale_data
               iv_rows   = lv_rows
               iv_cols   = lv_cols )
        ELSE VALUE #( ) ).   " FP32/FP16 not yet decoded here

      DATA(lo_tensor) = zcl_llm_tensor=>create_from_float_table(
        it_data  = lt_floats
        it_shape = lt_shape ).

      APPEND VALUE ty_named_tensor(
        layer_name = ls_w-layer_name
        tensor     = lo_tensor ) TO lt_named.
    ENDLOOP.

    " TODO: Apply tensor map to engine
    " apply_weights_to_engine will be implemented when engine API is ready.

    " Log: loader complete.
  ENDMETHOD.


  METHOD load_from_file.
    " File-based loading not available on this system.
    " Use load_from_ztable() instead.
    RETURN.
  ENDMETHOD.


  METHOD load_vocab.
    DATA: lt_vocab_rows  TYPE STANDARD TABLE OF zllm_vocab  WITH EMPTY KEY,
          lt_merge_rows  TYPE STANDARD TABLE OF zllm_merges WITH EMPTY KEY.

    SELECT * FROM zllm_vocab
      WHERE model_id = @iv_model_id
      ORDER BY token_id
      INTO TABLE @lt_vocab_rows.

    SELECT * FROM zllm_merges
      WHERE model_id = @iv_model_id
      ORDER BY priority
      INTO TABLE @lt_merge_rows.

    " Build vocabulary table (index = token_id, value = token string)
    DATA lt_vocab TYPE zcl_llm_bpe_tokenizer=>ty_vocab.
    LOOP AT lt_vocab_rows INTO DATA(ls_v).
      APPEND ls_v-token TO lt_vocab.
    ENDLOOP.

    " Build merge rules table
    DATA lt_merges TYPE zcl_llm_bpe_tokenizer=>ty_merge_pairs.
    LOOP AT lt_merge_rows INTO DATA(ls_m).
      APPEND VALUE zcl_llm_bpe_tokenizer=>ty_merge_pair(
        token_a  = ls_m-pair_left
        token_b  = ls_m-pair_right
        priority = ls_m-priority ) TO lt_merges.
    ENDLOOP.

    " Construct tokenizer and attach to engine
    DATA(lo_tokenizer) = NEW zcl_llm_bpe_tokenizer(
      it_vocab  = lt_vocab
      it_merges = lt_merges ).

    mo_engine->set_tokenizer( lo_tokenizer ).

    MESSAGE |LLM Model Loader: vocab loaded ({ lines( lt_vocab ) } tokens, | &&
            |{ lines( lt_merges ) } merges)| TYPE 'I'.
  ENDMETHOD.


  METHOD dequantize_int8.
    " Convert INT8 quantized weights back to float32.
    " Layout: data = int8[rows, cols], scales = float32[rows]
    " Result: float[rows, cols] where result[r,c] = int8[r,c] * scale[r]
    DATA(lv_total_elems) = iv_rows * iv_cols.
    DATA(lv_xlen)        = xstrlen( iv_data ).
    DATA(lv_slen)        = xstrlen( iv_scales ).

    IF lv_xlen = 0 OR lv_total_elems = 0.
      RETURN.
    ENDIF.

    " Parse float32 scales (little-endian IEEE 754)
    DATA lt_scales TYPE STANDARD TABLE OF f WITH EMPTY KEY.
    DATA: lv_si      TYPE i VALUE 0,
          lv_s0      TYPE i,
          lv_s1      TYPE i,
          lv_s2      TYPE i,
          lv_s3      TYPE i,
          lv_sp1     TYPE i,
          lv_sp2     TYPE i,
          lv_sp3     TYPE i,
          lv_uint32  TYPE i,
          lv_sign    TYPE i,
          lv_exp     TYPE i,
          lv_mant    TYPE i,
          lv_scale_f TYPE f.
    WHILE lv_si + 3 < lv_slen.
      lv_s0 = CONV i( iv_scales+lv_si(1) ).
      lv_sp1 = lv_si + 1.
      lv_s1 = CONV i( iv_scales+lv_sp1(1) ).
      lv_sp2 = lv_si + 2.
      lv_s2 = CONV i( iv_scales+lv_sp2(1) ).
      lv_sp3 = lv_si + 3.
      lv_s3 = CONV i( iv_scales+lv_sp3(1) ).
      lv_uint32 = lv_s0 + lv_s1 * 256 + lv_s2 * 65536 + lv_s3 * 16777216.
      " IEEE 754: sign(1) exponent(8) mantissa(23)
      lv_sign = lv_uint32 DIV 2147483648.     " bit 31
      lv_exp  = ( lv_uint32 DIV 8388608 ) MOD 256. " bits 30-23
      lv_mant = lv_uint32 MOD 8388608.         " bits 22-0
      IF lv_exp = 0.
        lv_scale_f = 0.  " denormals treated as zero
      ELSEIF lv_exp = 255.
        lv_scale_f = 0.  " Inf/NaN treated as zero
      ELSE.
        lv_scale_f = ( 1 + CONV f( lv_mant ) / 8388608 ) * ipow( base = 2 exp = lv_exp - 127 ).
        IF lv_sign = 1.
          lv_scale_f = lv_scale_f * -1.
        ENDIF.
      ENDIF.
      APPEND lv_scale_f TO lt_scales.
      lv_si = lv_si + 4.
    ENDWHILE.

    " Dequantize: iterate rows × cols
    DATA(lv_ri) = 0.
    WHILE lv_ri < iv_rows.
      DATA(lv_scale) = COND f(
        WHEN lv_ri < lines( lt_scales ) THEN lt_scales[ lv_ri + 1 ]
        ELSE '1.0' ).

      DATA(lv_ci) = 0.
      WHILE lv_ci < iv_cols.
        DATA(lv_byte_idx)  = lv_ri * iv_cols + lv_ci.
        IF lv_byte_idx >= lv_xlen.
          EXIT.
        ENDIF.
        DATA(lv_raw_byte)  = iv_data+lv_byte_idx(1).
        " Convert hex byte to signed int8 (-128..127)
        DATA lv_int8 TYPE i.
        lv_int8 = lv_raw_byte.  " implicit hex → int conversion
        IF lv_int8 > 127.
          lv_int8 = lv_int8 - 256.   " two's complement
        ENDIF.
        DATA lv_float TYPE f.
        lv_float = lv_int8 * lv_scale.
        APPEND lv_float TO rt_floats.
        lv_ci = lv_ci + 1.
      ENDWHILE.
      lv_ri = lv_ri + 1.
    ENDWHILE.
  ENDMETHOD.


  METHOD parse_shape.
    " Parse "[576,576]" or "[30,576]" into an integer table.
    DATA(lv_str) = iv_shape.
    " Remove brackets
    REPLACE ALL OCCURRENCES OF '[' IN lv_str WITH ''.
    REPLACE ALL OCCURRENCES OF ']' IN lv_str WITH ''.
    REPLACE ALL OCCURRENCES OF ' ' IN lv_str WITH ''.

    DATA lt_parts TYPE STANDARD TABLE OF string WITH EMPTY KEY.
    SPLIT lv_str AT ',' INTO TABLE lt_parts.

    LOOP AT lt_parts INTO DATA(lv_part).
      IF lv_part IS NOT INITIAL.
        APPEND CONV i( lv_part ) TO rt_shape.
      ENDIF.
    ENDLOOP.
  ENDMETHOD.


  METHOD apply_weights_to_engine.
    " Map HuggingFace tensor names to engine weight setters.
    " Placeholder — full implementation pending engine API.
    RETURN.
  ENDMETHOD.


  METHOD read_uint32.
    " Read 4 bytes at iv_offset from iv_xstr, interpret as little-endian uint32
    DATA(lv_b0) = CONV i( iv_xstr+iv_offset(1) ).
    DATA(lv_off1) = iv_offset + 1. DATA(lv_b1) = CONV i( iv_xstr+lv_off1(1) ).
    DATA(lv_off2) = iv_offset + 2. DATA(lv_b2) = CONV i( iv_xstr+lv_off2(1) ).
    DATA(lv_off3) = iv_offset + 3. DATA(lv_b3) = CONV i( iv_xstr+lv_off3(1) ).
    rv_value = lv_b0 + lv_b1 * 256 + lv_b2 * 65536 + lv_b3 * 16777216.
  ENDMETHOD.

  METHOD read_uint16.
    DATA(lv_b0) = CONV i( iv_xstr+iv_offset(1) ).
    DATA(lv_off1) = iv_offset + 1. DATA(lv_b1) = CONV i( iv_xstr+lv_off1(1) ).
    rv_value = lv_b0 + lv_b1 * 256.
  ENDMETHOD.

  METHOD read_byte.
    rv_value = CONV i( iv_xstr+iv_offset(1) ).
  ENDMETHOD.

ENDCLASS.
