*&---------------------------------------------------------------------*
*& Report ZLLM_DEMO
*&---------------------------------------------------------------------*
*& ABAP LLM Engine - Inference Demo
*& Demonstrates text generation using a transformer model running
*& entirely within the ABAP application server.
*&
*& Prerequisites:
*&   - Model weights loaded into ZLLM_WEIGHTS table (run ZLLM_LOAD_WEIGHTS)
*&   - Or use 'FILE' source with weights on application server
*&---------------------------------------------------------------------*
REPORT zllm_demo.

PARAMETERS:
  p_prompt TYPE string DEFAULT 'The capital of France is' LOWER CASE,
  p_maxtok TYPE i DEFAULT 20,
  p_temp   TYPE f DEFAULT '0.7',
  p_topk   TYPE i DEFAULT 40,
  p_topp   TYPE f DEFAULT '0.9',
  p_source TYPE string DEFAULT 'ZTAB' LOWER CASE,
  p_path   TYPE string DEFAULT 'ZLLM_WEIGHTS' LOWER CASE.

START-OF-SELECTION.

  WRITE: / '================================================='.
  WRITE: / '  ABAP LLM Engine - SmolLM2-135M Inference Demo'.
  WRITE: / '================================================='.
  SKIP.

  " ---------------------------------------------------------------
  " Step 1: Initialize the engine
  " ---------------------------------------------------------------
  WRITE: / 'Initializing LLM Engine...'.
  DATA(lo_engine) = NEW zcl_llm_engine( ).
  DATA(ls_config) = lo_engine->get_config( ).

  WRITE: / |  Model: SmolLM2-135M|.
  WRITE: / |  Hidden size: { ls_config-hidden_size }|.
  WRITE: / |  Layers: { ls_config-num_layers }|.
  WRITE: / |  Attention heads: { ls_config-num_heads } (KV: { ls_config-num_kv_heads })|.
  WRITE: / |  Vocab size: { ls_config-vocab_size }|.
  SKIP.

  " ---------------------------------------------------------------
  " Step 2: Load model weights
  " ---------------------------------------------------------------
  WRITE: / |Loading weights from { p_source }/{ p_path }...|.

  DATA(lv_start_time) = sy-uzeit.
  lo_engine->load_weights(
    iv_source = p_source
    iv_path   = p_path ).
  DATA(lv_load_time) = sy-uzeit - lv_start_time.

  WRITE: / |  Weight loading completed in { lv_load_time } seconds.|.
  SKIP.

  " ---------------------------------------------------------------
  " Step 3: Set up tokenizer (simplified for demo)
  " ---------------------------------------------------------------
  WRITE: / 'Initializing tokenizer...'.

  " In production, vocab and merges would be loaded from model data.
  " For this demo, we use a minimal character-level tokenizer.
  DATA(lt_vocab) = VALUE zcl_llm_bpe_tokenizer=>ty_vocab( ).
  DATA(lt_merges) = VALUE zcl_llm_bpe_tokenizer=>ty_merge_pairs( ).

  " Build basic ASCII + common token vocabulary
  DATA(lv_char) = 0.
  WHILE lv_char < 128.
    DATA(lv_char_str) = cl_abap_conv_codepage=>create_out( )->convert(
      source = VALUE xstring( ) ).
    " Simplified: add space and printable ASCII as individual tokens
    APPEND |{ CONV char1( lv_char ) }| TO lt_vocab.
    lv_char = lv_char + 1.
  ENDWHILE.

  " Add some common English tokens for better tokenization
  APPEND 'the' TO lt_vocab.
  APPEND 'The' TO lt_vocab.
  APPEND ' is' TO lt_vocab.
  APPEND ' of' TO lt_vocab.
  APPEND ' in' TO lt_vocab.

  " Add corresponding merge rules
  APPEND VALUE zcl_llm_bpe_tokenizer=>ty_merge_pair(
    token_a = 't' token_b = 'h' priority = 1 ) TO lt_merges.
  APPEND VALUE zcl_llm_bpe_tokenizer=>ty_merge_pair(
    token_a = 'th' token_b = 'e' priority = 2 ) TO lt_merges.

  DATA(lo_tokenizer) = NEW zcl_llm_bpe_tokenizer(
    it_vocab  = lt_vocab
    it_merges = lt_merges ).

  lo_engine->set_tokenizer( lo_tokenizer ).
  WRITE: / |  Vocabulary size: { lo_tokenizer->get_vocab_size( ) }|.
  SKIP.

  " ---------------------------------------------------------------
  " Step 4: Generate text
  " ---------------------------------------------------------------
  WRITE: / |Prompt: "{ p_prompt }"|.
  WRITE: / |Generating { p_maxtok } tokens (temp={ p_temp }, top_k={ p_topk }, top_p={ p_topp })...|.
  SKIP.

  lv_start_time = sy-uzeit.

  DATA(lv_result) = lo_engine->generate(
    iv_prompt      = p_prompt
    iv_max_tokens  = p_maxtok
    iv_temperature = p_temp
    iv_top_k       = p_topk
    iv_top_p       = p_topp ).

  DATA(lv_gen_time) = sy-uzeit - lv_start_time.

  WRITE: / '-------------------------------------------------'.
  WRITE: / 'Generated text:'.
  WRITE: / lv_result.
  WRITE: / '-------------------------------------------------'.
  SKIP.
  WRITE: / |Generation time: { lv_gen_time } seconds|.
  WRITE: / |Tokens generated: { p_maxtok }|.
  IF p_maxtok > 0.
    DATA(lv_tok_per_sec) = CONV f( lv_gen_time ) / p_maxtok.
    WRITE: / |Speed: ~{ lv_tok_per_sec } sec/token|.
  ENDIF.

  WRITE: / '================================================='.
  WRITE: / '  Note: With zero-initialized weights, output is'.
  WRITE: / '  random. Load real SmolLM2 weights for coherent'.
  WRITE: / '  text generation.'.
  WRITE: / '================================================='.
