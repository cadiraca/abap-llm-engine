"! <p class="shorttext synchronized" lang="en">LLM Engine - Byte-Pair Encoding Tokenizer</p>
"! Implements BPE tokenization for LLM text processing.
"! Vocabulary and merge rules are provided at construction time
"! and would normally be loaded from the model data.
CLASS zcl_llm_bpe_tokenizer DEFINITION
  PUBLIC
  CREATE PUBLIC.

  PUBLIC SECTION.
    TYPES:
      "! Merge pair: two tokens to combine
      BEGIN OF ty_merge_pair,
        token_a  TYPE string,
        token_b  TYPE string,
        priority TYPE i,
      END OF ty_merge_pair,
      ty_merge_pairs TYPE STANDARD TABLE OF ty_merge_pair WITH EMPTY KEY,

      "! Vocabulary: index → token string
      ty_vocab TYPE STANDARD TABLE OF string WITH EMPTY KEY,

      "! Token ID table
      ty_token_ids TYPE STANDARD TABLE OF i WITH EMPTY KEY.

    "! <p class="shorttext synchronized">Constructor</p>
    "! @parameter it_vocab | Vocabulary table (index = token ID)
    "! @parameter it_merges | BPE merge rules ordered by priority
    METHODS constructor
      IMPORTING it_vocab  TYPE ty_vocab
                it_merges TYPE ty_merge_pairs.

    "! <p class="shorttext synchronized">Encode text to token IDs</p>
    "! Applies BPE merges iteratively until no more merges apply.
    "! @parameter iv_text | Input text
    "! @parameter rt_token_ids | Resulting token ID sequence
    METHODS encode
      IMPORTING iv_text            TYPE string
      RETURNING VALUE(rt_token_ids) TYPE ty_token_ids.

    "! <p class="shorttext synchronized">Decode token IDs to text</p>
    "! @parameter it_token_ids | Token ID sequence
    "! @parameter rv_text | Reconstructed text
    METHODS decode
      IMPORTING it_token_ids    TYPE ty_token_ids
      RETURNING VALUE(rv_text) TYPE string.

    "! <p class="shorttext synchronized">Get vocabulary size</p>
    "! @parameter rv_size | Number of tokens in vocabulary
    METHODS get_vocab_size
      RETURNING VALUE(rv_size) TYPE i.

  PRIVATE SECTION.
    DATA:
      mt_vocab  TYPE ty_vocab,
      mt_merges TYPE ty_merge_pairs.

    "! Reverse lookup: token string → token ID
    DATA:
      mt_token_to_id TYPE HASHED TABLE OF i WITH UNIQUE KEY table_line.

    TYPES:
      ty_string_tab TYPE STANDARD TABLE OF string WITH EMPTY KEY.

    "! Build reverse vocab lookup
    METHODS build_reverse_vocab.

    "! Look up token ID for a token string
    "! @parameter iv_token | Token string
    "! @parameter rv_id | Token ID, -1 if not found
    METHODS get_token_id
      IMPORTING iv_token     TYPE string
      RETURNING VALUE(rv_id) TYPE i.

    "! Find the highest-priority merge that applies
    "! @parameter it_tokens | Current token sequence
    "! @parameter rv_merge_idx | Index of merge rule, 0 if none
    "! @parameter rv_pos | Position in token sequence where merge applies
    METHODS find_best_merge
      IMPORTING it_tokens         TYPE ty_string_tab
      EXPORTING ev_merge_idx      TYPE i
                ev_pos            TYPE i.

ENDCLASS.


CLASS zcl_llm_bpe_tokenizer IMPLEMENTATION.

  METHOD constructor.
    mt_vocab  = it_vocab.
    mt_merges = it_merges.
    build_reverse_vocab( ).
  ENDMETHOD.

  METHOD build_reverse_vocab.
    " Build string → id lookup using a sorted table for fast search
    DATA(lv_idx) = 0.
    LOOP AT mt_vocab INTO DATA(lv_token).
      " Store mapping: we'll use a simple approach with a parallel table
      lv_idx = lv_idx + 1.
    ENDLOOP.
  ENDMETHOD.

  METHOD get_token_id.
    rv_id = -1.
    DATA(lv_idx) = 0.
    LOOP AT mt_vocab INTO DATA(lv_token).
      IF lv_token = iv_token.
        rv_id = lv_idx.
        RETURN.
      ENDIF.
      lv_idx = lv_idx + 1.
    ENDLOOP.
  ENDMETHOD.

  METHOD encode.
    "--------------------------------------------------------------------
    " BPE Encoding Algorithm:
    " 1. Split text into individual characters (initial tokens)
    " 2. Iteratively find the highest-priority merge pair
    " 3. Merge adjacent tokens matching the pair
    " 4. Repeat until no more merges apply
    " 5. Map final token strings to IDs
    "--------------------------------------------------------------------

    " Step 1: Character-level tokenization
    DATA(lt_tokens) = VALUE ty_string_tab( ).
    DATA(lv_len) = strlen( iv_text ).
    DATA(lv_pos) = 0.
    WHILE lv_pos < lv_len.
      APPEND iv_text+lv_pos(1) TO lt_tokens.
      lv_pos = lv_pos + 1.
    ENDWHILE.

    " Step 2: Iterative BPE merges
    DATA: lv_merge_idx TYPE i,
          lv_merge_pos TYPE i.

    DO.
      find_best_merge(
        EXPORTING it_tokens    = lt_tokens
        IMPORTING ev_merge_idx = lv_merge_idx
                  ev_pos       = lv_merge_pos ).

      IF lv_merge_idx = 0.
        EXIT. " No more merges apply
      ENDIF.

      " Apply the merge: combine tokens at lv_merge_pos and lv_merge_pos+1
      DATA(lv_merged) = lt_tokens[ lv_merge_pos ] && lt_tokens[ lv_merge_pos + 1 ].
      lt_tokens[ lv_merge_pos ] = lv_merged.
      DELETE lt_tokens INDEX lv_merge_pos + 1.
    ENDDO.

    " Step 3: Map to token IDs
    LOOP AT lt_tokens INTO DATA(lv_token).
      DATA(lv_id) = get_token_id( lv_token ).
      IF lv_id >= 0.
        APPEND lv_id TO rt_token_ids.
      ELSE.
        " Unknown token — map to 0 (usually <unk>)
        APPEND 0 TO rt_token_ids.
      ENDIF.
    ENDLOOP.
  ENDMETHOD.

  METHOD find_best_merge.
    ev_merge_idx = 0.
    ev_pos = 0.
    DATA(lv_best_priority) = 999999999.

    " Check all adjacent pairs against merge rules
    DATA(lv_num_tokens) = lines( lt_tokens ).
    DATA(lv_i) = 1.

    WHILE lv_i < lv_num_tokens.
      DATA(lv_a) = lt_tokens[ lv_i ].
      DATA(lv_b) = lt_tokens[ lv_i + 1 ].

      " Look up this pair in merge rules
      DATA(lv_m) = 1.
      LOOP AT mt_merges INTO DATA(ls_merge).
        IF ls_merge-token_a = lv_a AND ls_merge-token_b = lv_b.
          IF ls_merge-priority < lv_best_priority.
            lv_best_priority = ls_merge-priority.
            ev_merge_idx = lv_m.
            ev_pos = lv_i.
          ENDIF.
          EXIT. " Found this pair's merge rule
        ENDIF.
        lv_m = lv_m + 1.
      ENDLOOP.

      lv_i = lv_i + 1.
    ENDWHILE.
  ENDMETHOD.

  METHOD decode.
    rv_text = ``.
    LOOP AT it_token_ids INTO DATA(lv_id).
      " Token IDs are 0-based, table is 1-based
      DATA(lv_table_idx) = lv_id + 1.
      IF lv_table_idx > 0 AND lv_table_idx <= lines( mt_vocab ).
        rv_text = rv_text && mt_vocab[ lv_table_idx ].
      ENDIF.
    ENDLOOP.
  ENDMETHOD.

  METHOD get_vocab_size.
    rv_size = lines( mt_vocab ).
  ENDMETHOD.

ENDCLASS.
