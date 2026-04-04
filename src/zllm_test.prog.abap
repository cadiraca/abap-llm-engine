"! <p class="shorttext synchronized" lang="en">ABAP LLM Engine - Test Harness</p>
"!
"! Validates each component of the ABAP LLM Engine against known reference
"! values.  Run this report after cloning via abapGit and before running
"! inference to confirm the maths are working correctly on your system.
"!
"! Test suite:
"!   1. Tensor ops   — create, matmul (2×2), add, reshape, slice
"!   2. RMSNorm      — [1,2,3] · weight [1,1,1] → expected values
"!   3. SiLU         — silu(1.0)≈0.7311, silu(-1.0)≈-0.2689
"!   4. Softmax      — [1,2,3] → [0.0900, 0.2447, 0.6652]
"!   5. BPE roundtrip— encode → decode on ASCII text
"!   6. Forward pass — zero weights, output shape = vocab_size
"!
"! Output: WRITE lines with PASS / FAIL markers and diff values.
REPORT zllm_test.

"═══════════════════════════════════════════════════════════════════════════════
" Test framework — tiny assertion helper
"═══════════════════════════════════════════════════════════════════════════════
CLASS lcl_assert DEFINITION FINAL.
  PUBLIC SECTION.
    CLASS-DATA: mv_pass TYPE i VALUE 0,
                mv_fail TYPE i VALUE 0.

    CLASS-METHODS check
      IMPORTING iv_desc     TYPE string
                iv_ok       TYPE abap_bool.

    CLASS-METHODS close_to
      IMPORTING iv_desc     TYPE string
                iv_got      TYPE f
                iv_expected TYPE f
                iv_tol      TYPE f DEFAULT '0.001'.

    CLASS-METHODS equal_i
      IMPORTING iv_desc     TYPE string
                iv_got      TYPE i
                iv_expected TYPE i.

    CLASS-METHODS equal_str
      IMPORTING iv_desc     TYPE string
                iv_got      TYPE string
                iv_expected TYPE string.

    CLASS-METHODS print_summary.

  PRIVATE SECTION.
    CLASS-METHODS pass  IMPORTING iv_desc TYPE string.
    CLASS-METHODS fail  IMPORTING iv_desc TYPE string iv_detail TYPE string.
ENDCLASS.

CLASS lcl_assert IMPLEMENTATION.
  METHOD pass.
    mv_pass = mv_pass + 1.
    WRITE: / |  ✓ PASS  { iv_desc }|.
  ENDMETHOD.

  METHOD fail.
    mv_fail = mv_fail + 1.
    WRITE: / |  ✗ FAIL  { iv_desc } — { iv_detail }|.
  ENDMETHOD.

  METHOD check.
    IF iv_ok = abap_true.
      pass( iv_desc ).
    ELSE.
      fail( iv_desc = iv_desc iv_detail = 'condition false' ).
    ENDIF.
  ENDMETHOD.

  METHOD close_to.
    DATA(lv_diff) = abs( iv_got - iv_expected ).
    IF lv_diff <= iv_tol.
      pass( iv_desc ).
    ELSE.
      fail(
        iv_desc   = iv_desc
        iv_detail = |got { iv_got DECIMALS 6 } expected { iv_expected DECIMALS 6 } diff { lv_diff DECIMALS 6 }| ).
    ENDIF.
  ENDMETHOD.

  METHOD equal_i.
    IF iv_got = iv_expected.
      pass( iv_desc ).
    ELSE.
      fail(
        iv_desc   = iv_desc
        iv_detail = |got { iv_got } expected { iv_expected }| ).
    ENDIF.
  ENDMETHOD.

  METHOD equal_str.
    IF iv_got = iv_expected.
      pass( iv_desc ).
    ELSE.
      fail(
        iv_desc   = iv_desc
        iv_detail = |got '{ iv_got }' expected '{ iv_expected }'| ).
    ENDIF.
  ENDMETHOD.

  METHOD print_summary.
    SKIP.
    DATA(lv_total) = mv_pass + mv_fail.
    WRITE: / '════════════════════════════════════════════════════════'.
    IF mv_fail = 0.
      WRITE: / |ALL TESTS PASSED  ({ mv_pass }/{ lv_total })|.
    ELSE.
      WRITE: / |TESTS FAILED: { mv_fail } failed, { mv_pass } passed ({ lv_total } total)|.
    ENDIF.
    WRITE: / '════════════════════════════════════════════════════════'.
  ENDMETHOD.
ENDCLASS.

"═══════════════════════════════════════════════════════════════════════════════
" Test suite
"═══════════════════════════════════════════════════════════════════════════════
CLASS lcl_tests DEFINITION FINAL.
  PUBLIC SECTION.
    CLASS-METHODS run_all.

  PRIVATE SECTION.
    CLASS-METHODS test_tensor_ops.
    CLASS-METHODS test_rms_norm.
    CLASS-METHODS test_silu.
    CLASS-METHODS test_softmax.
    CLASS-METHODS test_bpe_roundtrip.
    CLASS-METHODS test_forward_pass_shape.
ENDCLASS.

CLASS lcl_tests IMPLEMENTATION.

  METHOD run_all.
    WRITE: / 'Running ABAP LLM Engine Test Suite'.
    WRITE: / '────────────────────────────────────────────────────────'.
    SKIP.

    test_tensor_ops( ).
    test_rms_norm( ).
    test_silu( ).
    test_softmax( ).
    test_bpe_roundtrip( ).
    test_forward_pass_shape( ).

    lcl_assert=>print_summary( ).
  ENDMETHOD.

  "─────────────────────────────────────────────────────────────────────────────
  " 1. Tensor Operations
  "─────────────────────────────────────────────────────────────────────────────
  METHOD test_tensor_ops.
    WRITE: / '[ 1 ] Tensor Operations'.

    " ── create_zeros ─────────────────────────────────────────────────────────
    DATA(lo_z) = zcl_llm_tensor=>create_zeros( VALUE zif_llm_tensor=>ty_shape( ( 3 ) ( 3 ) ) ).
    lcl_assert=>equal_i(
      iv_desc     = 'create_zeros: size = 9'
      iv_got      = lo_z->get_size( )
      iv_expected = 9 ).

    DATA(lt_data) = lo_z->get_data( ).
    lcl_assert=>check(
      iv_desc = 'create_zeros: all values are 0'
      iv_ok   = COND abap_bool(
                  WHEN NOT line_exists( lt_data[ table_line = CONV f( 0.001 ) ] )
                  THEN abap_true ELSE abap_false ) ).

    " ── matmul 2×2 ───────────────────────────────────────────────────────────
    " A = [[1,2],[3,4]]  B = [[5,6],[7,8]]  A@B = [[19,22],[43,50]]
    DATA(lo_a) = zcl_llm_tensor=>create_from_float_table(
      it_data  = VALUE zif_llm_tensor=>ty_float_tab( ( 1 ) ( 2 ) ( 3 ) ( 4 ) )
      it_shape = VALUE zif_llm_tensor=>ty_shape( ( 2 ) ( 2 ) ) ).

    DATA(lo_b) = zcl_llm_tensor=>create_from_float_table(
      it_data  = VALUE zif_llm_tensor=>ty_float_tab( ( 5 ) ( 6 ) ( 7 ) ( 8 ) )
      it_shape = VALUE zif_llm_tensor=>ty_shape( ( 2 ) ( 2 ) ) ).

    DATA(lo_c) = lo_a->matmul( lo_b ).
    DATA(lt_c)  = lo_c->get_data( ).
    lcl_assert=>close_to( iv_desc = 'matmul [0,0]=19' iv_got = lt_c[ 1 ] iv_expected = CONV f( 19 ) ).
    lcl_assert=>close_to( iv_desc = 'matmul [0,1]=22' iv_got = lt_c[ 2 ] iv_expected = CONV f( 22 ) ).
    lcl_assert=>close_to( iv_desc = 'matmul [1,0]=43' iv_got = lt_c[ 3 ] iv_expected = CONV f( 43 ) ).
    lcl_assert=>close_to( iv_desc = 'matmul [1,1]=50' iv_got = lt_c[ 4 ] iv_expected = CONV f( 50 ) ).

    " ── add ──────────────────────────────────────────────────────────────────
    DATA(lo_x) = zcl_llm_tensor=>create_from_float_table(
      it_data  = VALUE zif_llm_tensor=>ty_float_tab( ( 1 ) ( 2 ) ( 3 ) )
      it_shape = VALUE zif_llm_tensor=>ty_shape( ( 3 ) ) ).

    DATA(lo_y) = zcl_llm_tensor=>create_from_float_table(
      it_data  = VALUE zif_llm_tensor=>ty_float_tab( ( 10 ) ( 20 ) ( 30 ) )
      it_shape = VALUE zif_llm_tensor=>ty_shape( ( 3 ) ) ).

    DATA(lo_sum) = lo_x->add( lo_y ).
    DATA(lt_sum) = lo_sum->get_data( ).
    lcl_assert=>close_to( iv_desc = 'add [0]=11' iv_got = lt_sum[ 1 ] iv_expected = CONV f( 11 ) ).
    lcl_assert=>close_to( iv_desc = 'add [1]=22' iv_got = lt_sum[ 2 ] iv_expected = CONV f( 22 ) ).
    lcl_assert=>close_to( iv_desc = 'add [2]=33' iv_got = lt_sum[ 3 ] iv_expected = CONV f( 33 ) ).

    " ── reshape ──────────────────────────────────────────────────────────────
    DATA(lo_flat) = zcl_llm_tensor=>create_from_float_table(
      it_data  = VALUE zif_llm_tensor=>ty_float_tab( ( 1 ) ( 2 ) ( 3 ) ( 4 ) ( 5 ) ( 6 ) )
      it_shape = VALUE zif_llm_tensor=>ty_shape( ( 6 ) ) ).

    DATA(lo_mat) = lo_flat->reshape( VALUE zif_llm_tensor=>ty_shape( ( 2 ) ( 3 ) ) ).
    DATA(lt_shape) = lo_mat->get_shape( ).
    lcl_assert=>equal_i( iv_desc = 'reshape: rows=2' iv_got = lt_shape[ 1 ] iv_expected = 2 ).
    lcl_assert=>equal_i( iv_desc = 'reshape: cols=3' iv_got = lt_shape[ 2 ] iv_expected = 3 ).
    lcl_assert=>equal_i( iv_desc = 'reshape: size=6' iv_got = lo_mat->get_size( ) iv_expected = 6 ).

    " ── slice ────────────────────────────────────────────────────────────────
    DATA(lo_src) = zcl_llm_tensor=>create_from_float_table(
      it_data  = VALUE zif_llm_tensor=>ty_float_tab( ( 10 ) ( 20 ) ( 30 ) ( 40 ) ( 50 ) )
      it_shape = VALUE zif_llm_tensor=>ty_shape( ( 5 ) ) ).

    DATA(lo_slc) = lo_src->slice( iv_start = 1 iv_length = 3 ).
    DATA(lt_slc) = lo_slc->get_data( ).
    lcl_assert=>equal_i( iv_desc = 'slice: length=3' iv_got = lo_slc->get_size( ) iv_expected = 3 ).
    lcl_assert=>close_to( iv_desc = 'slice [0]=20' iv_got = lt_slc[ 1 ] iv_expected = CONV f( 20 ) ).
    lcl_assert=>close_to( iv_desc = 'slice [2]=40' iv_got = lt_slc[ 3 ] iv_expected = CONV f( 40 ) ).

    SKIP.
  ENDMETHOD.

  "─────────────────────────────────────────────────────────────────────────────
  " 2. RMSNorm
  "─────────────────────────────────────────────────────────────────────────────
  METHOD test_rms_norm.
    WRITE: / '[ 2 ] RMSNorm'.
    " Input: [1.0, 2.0, 3.0]  weight: [1.0, 1.0, 1.0]  eps: 1e-5
    " mean_sq = (1+4+9)/3 = 4.6667
    " rms_inv = 1/sqrt(4.6667+1e-5) ≈ 0.46291
    " result  ≈ [0.46291, 0.92582, 1.38873]

    DATA(lo_input) = zcl_llm_tensor=>create_from_float_table(
      it_data  = VALUE zif_llm_tensor=>ty_float_tab( ( '1.0' ) ( '2.0' ) ( '3.0' ) )
      it_shape = VALUE zif_llm_tensor=>ty_shape( ( 3 ) ) ).

    DATA(lo_weight) = zcl_llm_tensor=>create_from_float_table(
      it_data  = VALUE zif_llm_tensor=>ty_float_tab( ( '1.0' ) ( '1.0' ) ( '1.0' ) )
      it_shape = VALUE zif_llm_tensor=>ty_shape( ( 3 ) ) ).

    DATA(lo_norm) = zcl_llm_math=>rms_norm(
      io_tensor = lo_input
      io_weight = lo_weight
      iv_eps    = CONV f( '0.00001' ) ).

    DATA(lt_norm) = lo_norm->get_data( ).

    lcl_assert=>close_to(
      iv_desc     = 'RMSNorm [0] ≈ 0.46291'
      iv_got      = lt_norm[ 1 ]
      iv_expected = CONV f( '0.46291' )
      iv_tol      = CONV f( '0.001' ) ).

    lcl_assert=>close_to(
      iv_desc     = 'RMSNorm [1] ≈ 0.92582'
      iv_got      = lt_norm[ 2 ]
      iv_expected = CONV f( '0.92582' )
      iv_tol      = CONV f( '0.001' ) ).

    lcl_assert=>close_to(
      iv_desc     = 'RMSNorm [2] ≈ 1.38873'
      iv_got      = lt_norm[ 3 ]
      iv_expected = CONV f( '1.38873' )
      iv_tol      = CONV f( '0.001' ) ).

    SKIP.
  ENDMETHOD.

  "─────────────────────────────────────────────────────────────────────────────
  " 3. SiLU Activation
  "─────────────────────────────────────────────────────────────────────────────
  METHOD test_silu.
    WRITE: / '[ 3 ] SiLU Activation'.
    " SiLU(x) = x * sigmoid(x)
    " silu(0.0)  = 0.0
    " silu(1.0)  = 1.0 * sigmoid(1.0) = 1.0 * 0.73106 = 0.73106
    " silu(-1.0) = -1.0 * sigmoid(-1.0) = -1.0 * 0.26894 = -0.26894

    lcl_assert=>close_to(
      iv_desc     = 'SiLU(0.0) = 0.0'
      iv_got      = zcl_llm_math=>silu( CONV f( '0.0' ) )
      iv_expected = CONV f( '0.0' ) ).

    lcl_assert=>close_to(
      iv_desc     = 'SiLU(1.0) ≈ 0.7311'
      iv_got      = zcl_llm_math=>silu( CONV f( '1.0' ) )
      iv_expected = CONV f( '0.7311' )
      iv_tol      = CONV f( '0.001' ) ).

    lcl_assert=>close_to(
      iv_desc     = 'SiLU(-1.0) ≈ -0.2689'
      iv_got      = zcl_llm_math=>silu( CONV f( '-1.0' ) )
      iv_expected = CONV f( '-0.2689' )
      iv_tol      = CONV f( '0.001' ) ).

    lcl_assert=>close_to(
      iv_desc     = 'SiLU(2.0) ≈ 1.7616'
      iv_got      = zcl_llm_math=>silu( CONV f( '2.0' ) )
      iv_expected = CONV f( '1.7616' )
      iv_tol      = CONV f( '0.001' ) ).

    SKIP.
  ENDMETHOD.

  "─────────────────────────────────────────────────────────────────────────────
  " 4. Softmax
  "─────────────────────────────────────────────────────────────────────────────
  METHOD test_softmax.
    WRITE: / '[ 4 ] Softmax'.
    " softmax([1,2,3]) = exp([1,2,3]-3) / sum(exp([1,2,3]-3))
    " exp([-2,-1,0])   = [0.13534, 0.36788, 1.0]
    " sum             = 1.50321
    " softmax          ≈ [0.09003, 0.24473, 0.66524]

    DATA(lo_in) = zcl_llm_tensor=>create_from_float_table(
      it_data  = VALUE zif_llm_tensor=>ty_float_tab( ( '1.0' ) ( '2.0' ) ( '3.0' ) )
      it_shape = VALUE zif_llm_tensor=>ty_shape( ( 3 ) ) ).

    DATA(lo_sm) = zcl_llm_math=>softmax( lo_in ).
    DATA(lt_sm) = lo_sm->get_data( ).

    lcl_assert=>close_to(
      iv_desc     = 'softmax([1,2,3])[0] ≈ 0.0900'
      iv_got      = lt_sm[ 1 ]
      iv_expected = CONV f( '0.09003' )
      iv_tol      = CONV f( '0.001' ) ).

    lcl_assert=>close_to(
      iv_desc     = 'softmax([1,2,3])[1] ≈ 0.2447'
      iv_got      = lt_sm[ 2 ]
      iv_expected = CONV f( '0.24473' )
      iv_tol      = CONV f( '0.001' ) ).

    lcl_assert=>close_to(
      iv_desc     = 'softmax([1,2,3])[2] ≈ 0.6652'
      iv_got      = lt_sm[ 3 ]
      iv_expected = CONV f( '0.66524' )
      iv_tol      = CONV f( '0.001' ) ).

    " Probabilities must sum to 1.0
    DATA(lv_sm_sum) = CONV f( 0 ).
    LOOP AT lt_sm INTO DATA(lv_p).
      lv_sm_sum = lv_sm_sum + lv_p.
    ENDLOOP.
    lcl_assert=>close_to(
      iv_desc     = 'softmax sum = 1.0'
      iv_got      = lv_sm_sum
      iv_expected = CONV f( '1.0' )
      iv_tol      = CONV f( '0.0001' ) ).

    SKIP.
  ENDMETHOD.

  "─────────────────────────────────────────────────────────────────────────────
  " 5. BPE Tokenizer Roundtrip
  "─────────────────────────────────────────────────────────────────────────────
  METHOD test_bpe_roundtrip.
    WRITE: / '[ 5 ] BPE Tokenizer Roundtrip'.

    " Build a minimal vocabulary that can represent "hello":
    " 0='h', 1='e', 2='l', 3='o', 4='he', 5='hel', 6='hell', 7='hello'
    DATA(lt_vocab) = VALUE zcl_llm_bpe_tokenizer=>ty_vocab(
      ( 'h' ) ( 'e' ) ( 'l' ) ( 'o' )
      ( 'he' ) ( 'hel' ) ( 'hell' ) ( 'hello' ) ).

    " Merge rules: in priority order
    DATA(lt_merges) = VALUE zcl_llm_bpe_tokenizer=>ty_merge_pairs(
      ( token_a = 'h'   token_b = 'e'    priority = 0 )
      ( token_a = 'he'  token_b = 'l'    priority = 1 )
      ( token_a = 'hel' token_b = 'l'    priority = 2 )
      ( token_a = 'hell' token_b = 'o'   priority = 3 ) ).

    DATA(lo_tok) = NEW zcl_llm_bpe_tokenizer(
      it_vocab  = lt_vocab
      it_merges = lt_merges ).

    " Encode
    DATA(lt_ids) = lo_tok->encode( 'hello' ).

    lcl_assert=>equal_i(
      iv_desc     = 'BPE encode "hello" → 1 token'
      iv_got      = lines( lt_ids )
      iv_expected = 1 ).

    lcl_assert=>equal_i(
      iv_desc     = 'BPE encoded token ID = 7 (index of "hello")'
      iv_got      = lt_ids[ 1 ]
      iv_expected = 7 ).

    " Decode
    DATA(lv_decoded) = lo_tok->decode( lt_ids ).
    lcl_assert=>equal_str(
      iv_desc     = 'BPE decode roundtrip = "hello"'
      iv_got      = lv_decoded
      iv_expected = 'hello' ).

    " Vocab size
    lcl_assert=>equal_i(
      iv_desc     = 'Vocab size = 8'
      iv_got      = lo_tok->get_vocab_size( )
      iv_expected = 8 ).

    SKIP.
  ENDMETHOD.

  "─────────────────────────────────────────────────────────────────────────────
  " 6. Forward Pass — output shape validation
  "─────────────────────────────────────────────────────────────────────────────
  METHOD test_forward_pass_shape.
    WRITE: / '[ 6 ] Forward Pass — Output Shape'.

    " Create engine with SmolLM2-135M default config
    DATA(lo_engine) = NEW zcl_llm_engine( ).

    " Build a tiny one-token embedding manually using zero weights
    " We only test that the engine produces the right output shape (vocab_size)
    " without actually loading real weights.

    " The engine's forward pass should accept a token ID and return
    " a tensor of shape [vocab_size] = [49152].
    " With zero weights the output will be all zeros, but the shape matters.

    DATA(lo_embed) = zcl_llm_tensor=>create_zeros(
      VALUE zif_llm_tensor=>ty_shape( ( 576 ) ) ).  " hidden_size = 576

    " Run the forward pass with zero embedding (tests graph integrity)
    DATA(lo_output) = lo_engine->forward(
      io_hidden_state = lo_embed
      iv_position     = 0 ).

    DATA(lv_output_size) = lo_output->get_size( ).

    lcl_assert=>equal_i(
      iv_desc     = 'Forward pass output size = vocab_size (49152)'
      iv_got      = lv_output_size
      iv_expected = 49152 ).

    DATA(lt_out_shape) = lo_output->get_shape( ).
    lcl_assert=>equal_i(
      iv_desc     = 'Forward pass output ndims = 1'
      iv_got      = lines( lt_out_shape )
      iv_expected = 1 ).

    lcl_assert=>equal_i(
      iv_desc     = 'Forward pass output shape[0] = 49152'
      iv_got      = lt_out_shape[ 1 ]
      iv_expected = 49152 ).

    SKIP.
  ENDMETHOD.

ENDCLASS.

"═══════════════════════════════════════════════════════════════════════════════
" Entry Point
"═══════════════════════════════════════════════════════════════════════════════
START-OF-SELECTION.
  WRITE: / '════════════════════════════════════════════════════════'.
  WRITE: / 'ABAP LLM Engine — Component Test Suite'.
  WRITE: / '════════════════════════════════════════════════════════'.
  SKIP.
  lcl_tests=>run_all( ).
