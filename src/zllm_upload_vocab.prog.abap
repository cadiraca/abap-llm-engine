"! <p class="shorttext synchronized" lang="en">ABAP LLM Engine - Upload BPE Vocabulary</p>
"!
"! Reads a SmolLM2 tokenizer.json from an AL11 application server path
"! and parses BPE vocabulary entries into ZLLM_VOCAB and merge rules
"! into ZLLM_MERGES transparent tables.
"!
"! tokenizer.json structure (HuggingFace tokenizers library):
"!   model.vocab  — { "token": id, ... }  (BPE vocabulary)
"!   model.merges — [ "left right", ... ] (ordered merge rules)
"!
"! Prerequisites:
"!   1. Copy tokenizer.json to an AL11 directory (e.g. /usr/sap/trans/llm/)
"!   2. Activate ZLLM_VOCAB and ZLLM_MERGES tables in your SAP system
"!   3. Run this report with the correct path and model ID
REPORT zllm_upload_vocab.

"═══════════════════════════════════════════════════════════════════════════════
" Selection Screen
"═══════════════════════════════════════════════════════════════════════════════
PARAMETERS:
  p_path  TYPE string LOWER CASE DEFAULT '/usr/sap/trans/llm/tokenizer.json',
  p_model TYPE char40             DEFAULT 'SMOLLM2-135M',
  p_reset TYPE abap_bool AS CHECKBOX DEFAULT abap_true.

"═══════════════════════════════════════════════════════════════════════════════
" Local class — JSON parser + DB writer
"═══════════════════════════════════════════════════════════════════════════════
CLASS lcl_vocab_uploader DEFINITION FINAL.
  PUBLIC SECTION.
    CLASS-METHODS run
      IMPORTING iv_path  TYPE string
                iv_model TYPE char40
                iv_reset TYPE abap_bool.

  PRIVATE SECTION.
    "! Minimal JSON string value extractor — finds value of a quoted key.
    "! Returns the raw JSON value (may be a string, number, object, array).
    CLASS-METHODS extract_json_block
      IMPORTING iv_json   TYPE string
                iv_key    TYPE string
      RETURNING VALUE(rv_block) TYPE string.

    "! Extract a JSON object block { ... } starting at a given offset.
    CLASS-METHODS extract_object_at
      IMPORTING iv_json   TYPE string
                iv_offset TYPE i
      RETURNING VALUE(rv_block) TYPE string.

    "! Extract a JSON array block [ ... ] starting at a given offset.
    CLASS-METHODS extract_array_at
      IMPORTING iv_json   TYPE string
                iv_offset TYPE i
      RETURNING VALUE(rv_block) TYPE string.

    "! Parse vocab object { "token": id, ... } and insert into ZLLM_VOCAB.
    CLASS-METHODS parse_and_insert_vocab
      IMPORTING iv_vocab_json TYPE string
                iv_model      TYPE char40
      RETURNING VALUE(rv_count) TYPE i.

    "! Parse merges array [ "a b", ... ] and insert into ZLLM_MERGES.
    CLASS-METHODS parse_and_insert_merges
      IMPORTING iv_merges_json TYPE string
                iv_model       TYPE char40
      RETURNING VALUE(rv_count) TYPE i.

    "! Extract integer from a JSON number-valued key.
    CLASS-METHODS find_string_offset
      IMPORTING iv_text    TYPE string
                iv_search  TYPE string
                iv_from    TYPE i DEFAULT 0
      RETURNING VALUE(rv_offset) TYPE i.
ENDCLASS.

CLASS lcl_vocab_uploader IMPLEMENTATION.

  METHOD run.
    " ── Read file ────────────────────────────────────────────────────────────
    DATA lv_line TYPE string.
    DATA lv_json TYPE string.

    DATA lv_msg TYPE string.
    OPEN DATASET iv_path FOR INPUT IN TEXT MODE ENCODING DEFAULT MESSAGE lv_msg.
    IF sy-subrc <> 0.
      WRITE: / |ERROR: Cannot open { iv_path }: { lv_msg }|.
      RETURN.
    ENDIF.
    DO.
      READ DATASET iv_path INTO lv_line.
      IF sy-subrc <> 0. EXIT. ENDIF.
      lv_json = lv_json && lv_line.
    ENDDO.
    CLOSE DATASET iv_path.

    DATA(lv_len) = strlen( lv_json ).
    WRITE: / |File read: { lv_len } characters.|.

    " ── Optional reset ───────────────────────────────────────────────────────
    IF iv_reset = abap_true.
      DELETE FROM zllm_vocab WHERE model_id = iv_model.
      WRITE: / |Deleted { sy-dbcnt } vocab rows for '{ iv_model }'.|.
      DELETE FROM zllm_merges WHERE model_id = iv_model.
      WRITE: / |Deleted { sy-dbcnt } merge rows for '{ iv_model }'.|.
      SKIP.
    ENDIF.

    " ── Locate "model" block ─────────────────────────────────────────────────
    DATA(lv_model_block) = extract_json_block(
      iv_json = lv_json
      iv_key  = 'model' ).

    IF lv_model_block IS INITIAL.
      WRITE: / 'ERROR: "model" key not found in tokenizer.json.'.
      RETURN.
    ENDIF.

    " ── Extract vocab object ─────────────────────────────────────────────────
    DATA(lv_vocab_block) = extract_json_block(
      iv_json = lv_model_block
      iv_key  = 'vocab' ).

    IF lv_vocab_block IS INITIAL.
      WRITE: / 'ERROR: "vocab" key not found in model block.'.
      RETURN.
    ENDIF.

    " ── Extract merges array ─────────────────────────────────────────────────
    DATA(lv_merges_block) = extract_json_block(
      iv_json = lv_model_block
      iv_key  = 'merges' ).

    IF lv_merges_block IS INITIAL.
      WRITE: / 'ERROR: "merges" key not found in model block.'.
      RETURN.
    ENDIF.

    " ── Parse and insert vocab ───────────────────────────────────────────────
    WRITE: / 'Parsing vocabulary entries...'.
    DATA(lv_vocab_count) = parse_and_insert_vocab(
      iv_vocab_json = lv_vocab_block
      iv_model      = iv_model ).
    WRITE: / |  Inserted { lv_vocab_count } vocab entries into ZLLM_VOCAB.|.
    SKIP.

    " ── Parse and insert merges ──────────────────────────────────────────────
    WRITE: / 'Parsing BPE merge rules...'.
    DATA(lv_merge_count) = parse_and_insert_merges(
      iv_merges_json = lv_merges_block
      iv_model       = iv_model ).
    WRITE: / |  Inserted { lv_merge_count } merge rules into ZLLM_MERGES.|.
    SKIP.

    " ── Summary ──────────────────────────────────────────────────────────────
    WRITE: / '════════════════════════════════════════════════'.
    WRITE: / |Upload complete.|.
    WRITE: / |  Model : { iv_model }|.
    WRITE: / |  Vocab : { lv_vocab_count } tokens|.
    WRITE: / |  Merges: { lv_merge_count } rules|.
    WRITE: / '════════════════════════════════════════════════'.
  ENDMETHOD.

  METHOD find_string_offset.
    rv_offset = -1.
    DATA(lv_len)     = strlen( iv_text ).
    DATA(lv_srch_ln) = strlen( iv_search ).
    DATA(lv_pos)     = iv_from.

    WHILE lv_pos <= lv_len - lv_srch_ln.
      IF iv_text+lv_pos(lv_srch_ln) = iv_search.
        rv_offset = lv_pos.
        RETURN.
      ENDIF.
      lv_pos = lv_pos + 1.
    ENDWHILE.
  ENDMETHOD.

  METHOD extract_object_at.
    rv_block = ''.
    DATA(lv_len)   = strlen( iv_json ).
    DATA(lv_pos)   = iv_offset.
    DATA(lv_depth) = 0.
    DATA(lv_start) = -1.

    WHILE lv_pos < lv_len.
      DATA(lv_ch) = iv_json+lv_pos(1).
      CASE lv_ch.
        WHEN '{'.
          IF lv_start = -1. lv_start = lv_pos. ENDIF.
          lv_depth = lv_depth + 1.
        WHEN '}'.
          lv_depth = lv_depth - 1.
          IF lv_depth = 0 AND lv_start >= 0.
            DATA(lv_block_len) = lv_pos - lv_start + 1.
            rv_block = iv_json+lv_start(lv_block_len).
            RETURN.
          ENDIF.
      ENDCASE.
      lv_pos = lv_pos + 1.
    ENDWHILE.
  ENDMETHOD.

  METHOD extract_array_at.
    rv_block = ''.
    DATA(lv_len)   = strlen( iv_json ).
    DATA(lv_pos)   = iv_offset.
    DATA(lv_depth) = 0.
    DATA(lv_start) = -1.

    WHILE lv_pos < lv_len.
      DATA(lv_ch) = iv_json+lv_pos(1).
      CASE lv_ch.
        WHEN '['.
          IF lv_start = -1. lv_start = lv_pos. ENDIF.
          lv_depth = lv_depth + 1.
        WHEN ']'.
          lv_depth = lv_depth - 1.
          IF lv_depth = 0 AND lv_start >= 0.
            DATA(lv_block_len) = lv_pos - lv_start + 1.
            rv_block = iv_json+lv_start(lv_block_len).
            RETURN.
          ENDIF.
      ENDCASE.
      lv_pos = lv_pos + 1.
    ENDWHILE.
  ENDMETHOD.

  METHOD extract_json_block.
    rv_block = ''.
    " Find  "key"  then skip whitespace/colon, then extract value block
    DATA(lv_key_pattern) = |"{ iv_key }"|.
    DATA(lv_key_pos) = find_string_offset(
      iv_text   = iv_json
      iv_search = lv_key_pattern ).

    IF lv_key_pos < 0. RETURN. ENDIF.

    " Skip past the key and find the value start (after colon)
    DATA(lv_pos) = lv_key_pos + strlen( lv_key_pattern ).
    DATA(lv_len) = strlen( iv_json ).

    WHILE lv_pos < lv_len.
      DATA(lv_ch) = iv_json+lv_pos(1).
      CASE lv_ch.
        WHEN '{'.
          rv_block = extract_object_at( iv_json = iv_json iv_offset = lv_pos ).
          RETURN.
        WHEN '['.
          rv_block = extract_array_at( iv_json = iv_json iv_offset = lv_pos ).
          RETURN.
        WHEN '"'.
          " String value — find closing quote (simplified, no escape handling)
          DATA(lv_start) = lv_pos + 1.
          DATA(lv_end)   = find_string_offset(
            iv_text   = iv_json
            iv_search = '"'
            iv_from   = lv_start ).
          IF lv_end > lv_start.
            DATA(lv_str_len) = lv_end - lv_start.
            rv_block = iv_json+lv_start(lv_str_len).
          ENDIF.
          RETURN.
        WHEN ':' OR ' ' OR cl_abap_char_utilities=>newline OR cl_abap_char_utilities=>cr_lf.
          " skip whitespace and colon
        WHEN OTHERS.
          " Numeric or other literal — read until separator
          lv_start = lv_pos.
          WHILE lv_pos < lv_len.
            lv_ch = iv_json+lv_pos(1).
            IF lv_ch = ',' OR lv_ch = '}' OR lv_ch = ']'
               OR lv_ch = ' ' OR lv_ch = cl_abap_char_utilities=>newline.
              DATA(lv_num_len) = lv_pos - lv_start.
              rv_block = iv_json+lv_start(lv_num_len).
              RETURN.
            ENDIF.
            lv_pos = lv_pos + 1.
          ENDWHILE.
      ENDCASE.
      lv_pos = lv_pos + 1.
    ENDWHILE.
  ENDMETHOD.

  METHOD parse_and_insert_vocab.
    rv_count = 0.
    " Format: { "tokenstring": integer_id, ... }
    " Strip outer braces
    DATA(lv_len) = strlen( iv_vocab_json ).
    IF lv_len < 2. RETURN. ENDIF.

    DATA(lv_offset) = 1.
    DATA(lv_chars) = lv_len - 2.
    DATA(lv_inner) = iv_vocab_json+lv_offset(lv_chars).
    DATA(lv_pos)   = 0.
    DATA(lv_inner_len) = strlen( lv_inner ).

    DATA: lt_vocab   TYPE STANDARD TABLE OF zllm_vocab WITH EMPTY KEY,
          ls_vocab   TYPE zllm_vocab.

    DO.
      " Skip whitespace and commas
      WHILE lv_pos < lv_inner_len.
        DATA(lv_ch) = lv_inner+lv_pos(1).
        IF lv_ch <> ' ' AND lv_ch <> ',' AND lv_ch <> cl_abap_char_utilities=>newline
           AND lv_ch <> cl_abap_char_utilities=>cr_lf.
          EXIT.
        ENDIF.
        lv_pos = lv_pos + 1.
      ENDWHILE.

      IF lv_pos >= lv_inner_len. EXIT. ENDIF.

      " Expect opening quote of token
      IF lv_inner+lv_pos(1) <> '"'. EXIT. ENDIF.
      lv_pos = lv_pos + 1.

      " Read token string (handle escape sequences)
      DATA lv_token TYPE string.
      lv_token = ''.
      DATA lv_escaped TYPE abap_bool VALUE abap_false.
      DO.
        IF lv_pos >= lv_inner_len. EXIT. ENDIF.
        DATA(lv_c) = lv_inner+lv_pos(1).
        lv_pos = lv_pos + 1.

        IF lv_escaped = abap_true.
          CASE lv_c.
            WHEN 'n'.  lv_token = lv_token && cl_abap_char_utilities=>newline.
            WHEN 't'.  lv_token = lv_token && cl_abap_char_utilities=>horizontal_tab.
            WHEN 'r'.  lv_token = lv_token && cl_abap_char_utilities=>cr_lf+0(1).
            WHEN OTHERS. lv_token = lv_token && lv_c.
          ENDCASE.
          lv_escaped = abap_false.
        ELSEIF lv_c = '\'.
          lv_escaped = abap_true.
        ELSEIF lv_c = '"'.
          EXIT. " End of token string
        ELSE.
          lv_token = lv_token && lv_c.
        ENDIF.
      ENDDO.

      " Skip colon
      WHILE lv_pos < lv_inner_len.
        lv_ch = lv_inner+lv_pos(1).
        lv_pos = lv_pos + 1.
        IF lv_ch = ':'. EXIT. ENDIF.
      ENDWHILE.

      " Skip whitespace
      WHILE lv_pos < lv_inner_len AND ( lv_inner+lv_pos(1) = ' ' ).
        lv_pos = lv_pos + 1.
      ENDWHILE.

      " Read integer ID
      DATA lv_id_str TYPE string.
      lv_id_str = ''.
      WHILE lv_pos < lv_inner_len.
        lv_ch = lv_inner+lv_pos(1).
        IF lv_ch CA '0123456789'.
          lv_id_str = lv_id_str && lv_ch.
          lv_pos = lv_pos + 1.
        ELSE.
          EXIT.
        ENDIF.
      ENDWHILE.

      IF lv_id_str IS INITIAL. EXIT. ENDIF.

      DATA(lv_id) = CONV i( lv_id_str ).

      " Build DB row — truncate token if longer than field allows
      CLEAR ls_vocab.
      ls_vocab-model_id  = iv_model.
      ls_vocab-token_id  = lv_id.
      DATA(lv_tlen) = strlen( lv_token ).
      IF lv_tlen > 200.
        lv_tlen = 200.
      ENDIF.
      ls_vocab-token     = lv_token(lv_tlen).
      ls_vocab-score     = CONV decfloat34( lv_id ) * ( -1 ).  " GPT2-style score: -id
      APPEND ls_vocab TO lt_vocab.

      rv_count = rv_count + 1.

      " Batch insert every 1000 entries
      IF rv_count MOD 1000 = 0.
        INSERT zllm_vocab FROM TABLE lt_vocab.
        CLEAR lt_vocab.
        WRITE: / |  Progress: { rv_count } vocab entries inserted...|.
      ENDIF.
    ENDDO.

    " Insert remaining
    IF lt_vocab IS NOT INITIAL.
      INSERT zllm_vocab FROM TABLE lt_vocab.
    ENDIF.
  ENDMETHOD.

  METHOD parse_and_insert_merges.
    rv_count = 0.
    " Format: [ "left right", "left right", ... ]
    " Strip outer brackets
    DATA(lv_len) = strlen( iv_merges_json ).
    IF lv_len < 2. RETURN. ENDIF.

    DATA(lv_off2) = 1.
    DATA(lv_ch2) = lv_len - 2.
    DATA(lv_inner) = iv_merges_json+lv_off2(lv_ch2).
    DATA(lv_pos)   = 0.
    DATA(lv_inner_len) = strlen( lv_inner ).

    DATA: lt_merges   TYPE STANDARD TABLE OF zllm_merges WITH EMPTY KEY,
          ls_merge    TYPE zllm_merges.

    DO.
      " Skip whitespace and commas
      WHILE lv_pos < lv_inner_len.
        DATA(lv_ch) = lv_inner+lv_pos(1).
        IF lv_ch <> ' ' AND lv_ch <> ',' AND lv_ch <> cl_abap_char_utilities=>newline
           AND lv_ch <> cl_abap_char_utilities=>cr_lf.
          EXIT.
        ENDIF.
        lv_pos = lv_pos + 1.
      ENDWHILE.

      IF lv_pos >= lv_inner_len. EXIT. ENDIF.

      " Expect opening quote
      IF lv_inner+lv_pos(1) <> '"'. EXIT. ENDIF.
      lv_pos = lv_pos + 1.

      " Read until closing quote (merge rule is "left right")
      DATA lv_merge_rule TYPE string.
      lv_merge_rule = ''.
      DO.
        IF lv_pos >= lv_inner_len. EXIT. ENDIF.
        DATA(lv_c) = lv_inner+lv_pos(1).
        lv_pos = lv_pos + 1.
        IF lv_c = '"'. EXIT. ENDIF.
        lv_merge_rule = lv_merge_rule && lv_c.
      ENDDO.

      " Split on space: "left right"
      DATA(lv_space_pos) = find_string_offset(
        iv_text   = lv_merge_rule
        iv_search = ' ' ).

      IF lv_space_pos < 0. CONTINUE. ENDIF.

      DATA(lv_left)  = lv_merge_rule(lv_space_pos).
      DATA(lv_right_start) = lv_space_pos + 1.
      DATA(lv_right_len)   = strlen( lv_merge_rule ) - lv_right_start.
      DATA(lv_right) = lv_merge_rule+lv_right_start(lv_right_len).

      " Build DB row
      CLEAR ls_merge.
      ls_merge-model_id   = iv_model.
      ls_merge-priority   = rv_count.  " 0-based insertion order = priority
      DATA(lv_llen) = strlen( lv_left ).
      IF lv_llen > 100.
        lv_llen = 100.
      ENDIF.
      DATA(lv_rlen) = strlen( lv_right ).
      IF lv_rlen > 100.
        lv_rlen = 100.
      ENDIF.
      ls_merge-pair_left  = lv_left(lv_llen).
      ls_merge-pair_right = lv_right(lv_rlen).
      APPEND ls_merge TO lt_merges.

      rv_count = rv_count + 1.

      " Batch insert every 1000 entries
      IF rv_count MOD 1000 = 0.
        INSERT zllm_merges FROM TABLE lt_merges.
        CLEAR lt_merges.
        WRITE: / |  Progress: { rv_count } merge rules inserted...|.
      ENDIF.
    ENDDO.

    " Insert remaining
    IF lt_merges IS NOT INITIAL.
      INSERT zllm_merges FROM TABLE lt_merges.
    ENDIF.
  ENDMETHOD.

ENDCLASS.

"═══════════════════════════════════════════════════════════════════════════════
" Entry Point
"═══════════════════════════════════════════════════════════════════════════════
START-OF-SELECTION.
  WRITE: / '════════════════════════════════════════════════'.
  WRITE: / 'ABAP LLM Engine — Upload BPE Vocabulary'.
  WRITE: / '════════════════════════════════════════════════'.
  WRITE: / 'File :', p_path.
  WRITE: / 'Model:', p_model.
  WRITE: / 'Reset:', p_reset.
  SKIP.
  lcl_vocab_uploader=>run(
    iv_path  = p_path
    iv_model = p_model
    iv_reset = p_reset ).
