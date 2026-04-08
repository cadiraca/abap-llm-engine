"! <p class="shorttext synchronized" lang="en">ABAP LLM Engine - Upload Model Weights</p>
"!
"! Reads an ALLM binary weight file from an AL11 application server path
"! and inserts each tensor record into the ZLLM_WEIGHTS transparent table.
"!
"! Binary format (ALLM v1):
"!   Header:  "ALLM"(4) + version(4) + num_tensors(4) + config_json_len(4) + config_json
"!   Per tensor:
"!     name_len(2, LE) + name(UTF-8) + dtype(1) + ndims(1) +
"!     shape[ndims*4, LE uint32] + data_len(4) + data_bytes +
"!     scale_len(4) + scale_bytes
"!
"! Prerequisites:
"!   1. Generate model.bin:  python tools/convert_weights.py --format int8
"!   2. Upload model.bin to an AL11 directory (e.g. /usr/sap/trans/llm/)
"!   3. Activate the ZLLM_WEIGHTS table in your SAP system
"!
"! Selection screen parameters:
"!   P_PATH  - Full AL11 path to model.bin
"!   P_MODEL - Logical model ID stored as key in ZLLM_WEIGHTS
"!   P_RESET - If checked, deletes existing rows for that model before inserting
REPORT zllm_upload_weights.

"═══════════════════════════════════════════════════════════════════════════════
" Selection Screen
"═══════════════════════════════════════════════════════════════════════════════
PARAMETERS:
  p_path  TYPE string LOWER CASE DEFAULT '/usr/sap/trans/llm/model.bin',
  p_model TYPE char40             DEFAULT 'SMOLLM2-135M',
  p_reset TYPE abap_bool AS CHECKBOX DEFAULT abap_true.

"═══════════════════════════════════════════════════════════════════════════════
" Local class — parser + DB writer
"═══════════════════════════════════════════════════════════════════════════════
CLASS lcl_uploader DEFINITION FINAL.
  PUBLIC SECTION.
    CLASS-METHODS run
      IMPORTING iv_path  TYPE string
                iv_model TYPE char40
                iv_reset TYPE abap_bool.

  PRIVATE SECTION.
    CLASS-METHODS read_uint32
      IMPORTING iv_xstr        TYPE xstring
                iv_offset      TYPE i
      RETURNING VALUE(rv_val)  TYPE i.

    CLASS-METHODS read_uint16
      IMPORTING iv_xstr        TYPE xstring
                iv_offset      TYPE i
      RETURNING VALUE(rv_val)  TYPE i.

    CLASS-METHODS read_byte
      IMPORTING iv_xstr        TYPE xstring
                iv_offset      TYPE i
      RETURNING VALUE(rv_val)  TYPE i.
ENDCLASS.

CLASS lcl_uploader IMPLEMENTATION.

  METHOD run.
    DATA lv_xstr  TYPE xstring.
    DATA lv_chunk TYPE xstring.

    " ── Read file ────────────────────────────────────────────────────────────
    DATA lv_msg TYPE string.
    OPEN DATASET iv_path FOR INPUT IN BINARY MODE MESSAGE lv_msg.
    IF sy-subrc <> 0.
      WRITE: / |ERROR: Cannot open { iv_path }: { lv_msg }|.
      RETURN.
    ENDIF.
    DO.
      READ DATASET iv_path INTO lv_chunk MAXIMUM LENGTH 65536.
      lv_xstr = lv_xstr && lv_chunk.
      IF sy-subrc <> 0. EXIT. ENDIF.
    ENDDO.
    CLOSE DATASET iv_path.

    DATA(lv_total) = xstrlen( lv_xstr ).
    DATA(lv_mb) = CONV decfloat34( lv_total ) / 1048576.
    WRITE: / |File size: { lv_total } bytes ({ lv_mb } MB)|.

    " ── Verify magic "ALLM" ──────────────────────────────────────────────────
    IF lv_total < 16 OR lv_xstr(4) <> '414C4C4D'.
      WRITE: / 'ERROR: Not a valid ALLM file.'.
      RETURN.
    ENDIF.

    " ── Parse header ─────────────────────────────────────────────────────────
    DATA(lv_version)  = read_uint32( iv_xstr = lv_xstr iv_offset = 4 ).
    DATA(lv_n_tensor) = read_uint32( iv_xstr = lv_xstr iv_offset = 8 ).
    DATA(lv_cfg_len)  = read_uint32( iv_xstr = lv_xstr iv_offset = 12 ).
    WRITE: / |Version: { lv_version }  Tensors: { lv_n_tensor }  Config JSON: { lv_cfg_len } B|.
    SKIP.

    " ── Optional reset ───────────────────────────────────────────────────────
    IF iv_reset = abap_true.
      DELETE FROM zllm_weights WHERE model_id = iv_model.
      WRITE: / |Deleted existing rows for '{ iv_model }'. SY-DBCNT = { sy-dbcnt }|.
      SKIP.
    ENDIF.

    " ── Parse and insert each tensor ─────────────────────────────────────────
    DATA(lv_pos)      = 16 + lv_cfg_len.
    DATA(lv_inserted) = 0.
    DATA(lv_errors)   = 0.

    DO lv_n_tensor TIMES.
      DATA(lv_tidx) = sy-index.

      " Name
      DATA(lv_name_len)  = read_uint16( iv_xstr = lv_xstr iv_offset = lv_pos ).
      lv_pos = lv_pos + 2.
      DATA(lv_name_xstr) = lv_xstr+lv_pos(lv_name_len).
      DATA(lv_name)      = cl_abap_codepage=>convert_from( lv_name_xstr ).
      lv_pos = lv_pos + lv_name_len.

      " dtype
      DATA(lv_dtype_byte) = read_byte( iv_xstr = lv_xstr iv_offset = lv_pos ).
      DATA lv_dtype TYPE char10.
      CASE lv_dtype_byte.
        WHEN 0.       lv_dtype = 'FP32'.
        WHEN 1.       lv_dtype = 'INT8'.
        WHEN 2.       lv_dtype = 'FP16'.
        WHEN OTHERS.  lv_dtype = 'UNKNOWN'.
      ENDCASE.
      lv_pos = lv_pos + 1.

      " ndims + shape
      DATA(lv_ndims)    = read_byte( iv_xstr = lv_xstr iv_offset = lv_pos ).
      lv_pos = lv_pos + 1.
      DATA lv_shape TYPE string.
      lv_shape = '['.
      DATA lv_di TYPE i VALUE 0.
      WHILE lv_di < lv_ndims.
        DATA(lv_dim) = read_uint32( iv_xstr = lv_xstr iv_offset = lv_pos ).
        lv_pos = lv_pos + 4.
        IF lv_di > 0. lv_shape = lv_shape && ','. ENDIF.
        lv_shape = lv_shape && lv_dim.
        lv_di = lv_di + 1.
      ENDWHILE.
      lv_shape = lv_shape && ']'.

      " Weight data
      DATA(lv_data_len)  = read_uint32( iv_xstr = lv_xstr iv_offset = lv_pos ).
      lv_pos = lv_pos + 4.
      DATA(lv_wdata)     = lv_xstr+lv_pos(lv_data_len).
      lv_pos = lv_pos + lv_data_len.

      " Scale data
      DATA(lv_scale_len) = read_uint32( iv_xstr = lv_xstr iv_offset = lv_pos ).
      lv_pos = lv_pos + 4.
      DATA lv_sdata TYPE xstring.
      IF lv_scale_len > 0.
        lv_sdata = lv_xstr+lv_pos(lv_scale_len).
        lv_pos   = lv_pos + lv_scale_len.
      ELSE.
        CLEAR lv_sdata.
      ENDIF.

      " Insert DB row
      DATA ls_w TYPE zllm_weights.
      ls_w-model_id    = iv_model.
      ls_w-layer_name  = lv_name.
      ls_w-weight_data = lv_wdata.
      ls_w-shape       = lv_shape.
      ls_w-dtype       = lv_dtype.
      ls_w-scale_data  = lv_sdata.

      INSERT zllm_weights FROM ls_w.
      IF sy-subrc = 0.
        lv_inserted = lv_inserted + 1.
      ELSE.
        lv_errors = lv_errors + 1.
        WRITE: / |  WARN: Duplicate or error for { lv_name } (sy-subrc={ sy-subrc })|.
      ENDIF.

      " Progress every 10 tensors
      IF lv_tidx MOD 10 = 0 OR lv_tidx = lv_n_tensor.
        DATA(lv_pct) = lv_tidx * 100 / lv_n_tensor.
        WRITE: / |[{ lv_pct }%] { lv_tidx }/{ lv_n_tensor }  { lv_name }  { lv_shape }  { lv_dtype }|.
      ENDIF.
    ENDDO.

    SKIP.
    WRITE: / '════════════════════════════════════════════════'.
    WRITE: / |Upload complete: { lv_inserted } inserted, { lv_errors } errors.|.
    WRITE: / '════════════════════════════════════════════════'.
  ENDMETHOD.

  METHOD read_uint32.
    DATA(b0) = read_byte( iv_xstr = iv_xstr iv_offset = iv_offset ).
    DATA(b1) = read_byte( iv_xstr = iv_xstr iv_offset = iv_offset + 1 ).
    DATA(b2) = read_byte( iv_xstr = iv_xstr iv_offset = iv_offset + 2 ).
    DATA(b3) = read_byte( iv_xstr = iv_xstr iv_offset = iv_offset + 3 ).
    rv_val = b0 + b1 * 256 + b2 * 65536 + b3 * 16777216.
  ENDMETHOD.

  METHOD read_uint16.
    DATA(b0) = read_byte( iv_xstr = iv_xstr iv_offset = iv_offset ).
    DATA(b1) = read_byte( iv_xstr = iv_xstr iv_offset = iv_offset + 1 ).
    rv_val = b0 + b1 * 256.
  ENDMETHOD.

  METHOD read_byte.
    DATA lv_x TYPE x LENGTH 1.
    lv_x   = iv_xstr+iv_offset(1).
    rv_val = lv_x.
  ENDMETHOD.

ENDCLASS.

"═══════════════════════════════════════════════════════════════════════════════
" Entry Point
"═══════════════════════════════════════════════════════════════════════════════
START-OF-SELECTION.
  WRITE: / '════════════════════════════════════════════════'.
  WRITE: / 'ABAP LLM Engine — Upload Model Weights to ZLLM_WEIGHTS'.
  WRITE: / '════════════════════════════════════════════════'.
  WRITE: / 'File :', p_path.
  WRITE: / 'Model:', p_model.
  WRITE: / 'Reset:', p_reset.
  SKIP.
  lcl_uploader=>run(
    iv_path  = p_path
    iv_model = p_model
    iv_reset = p_reset ).
