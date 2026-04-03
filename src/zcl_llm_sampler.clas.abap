"! <p class="shorttext synchronized" lang="en">LLM Engine - Token Sampling</p>
"! Implements token sampling strategies for autoregressive generation:
"! temperature scaling, top-K filtering, top-P (nucleus) filtering,
"! and random weighted sampling from the filtered distribution.
CLASS zcl_llm_sampler DEFINITION
  PUBLIC
  CREATE PUBLIC.

  PUBLIC SECTION.
    TYPES:
      ty_float_tab TYPE zif_llm_tensor=>ty_float_tab,
      ty_shape     TYPE zif_llm_tensor=>ty_shape.

    "! <p class="shorttext synchronized">Sample a token from logits</p>
    "! @parameter io_logits | Raw logits tensor (vocab_size)
    "! @parameter iv_temperature | Sampling temperature (0 = greedy, default 1.0)
    "! @parameter iv_top_k | Top-K filter (0 = disabled, default 40)
    "! @parameter iv_top_p | Top-P nucleus filter (1.0 = disabled, default 0.9)
    "! @parameter rv_token_id | Sampled token ID
    METHODS sample
      IMPORTING io_logits          TYPE REF TO zif_llm_tensor
                iv_temperature     TYPE f DEFAULT '1.0'
                iv_top_k           TYPE i DEFAULT 40
                iv_top_p           TYPE f DEFAULT '0.9'
      RETURNING VALUE(rv_token_id) TYPE i.

  PRIVATE SECTION.
    TYPES:
      BEGIN OF ty_logit_entry,
        index TYPE i,
        value TYPE f,
      END OF ty_logit_entry,
      ty_logit_entries TYPE STANDARD TABLE OF ty_logit_entry WITH EMPTY KEY.

    "! Apply temperature scaling to logits
    METHODS temperature_scale
      IMPORTING iv_temperature    TYPE f
      CHANGING  ct_logits         TYPE ty_float_tab.

    "! Apply top-K filtering: keep only K highest logits
    METHODS top_k_filter
      IMPORTING iv_k              TYPE i
      CHANGING  ct_entries        TYPE ty_logit_entries.

    "! Apply top-P (nucleus) filtering
    METHODS top_p_filter
      IMPORTING iv_p              TYPE f
      CHANGING  ct_entries        TYPE ty_logit_entries.

    "! Random weighted sampling from filtered distribution
    METHODS random_sample
      IMPORTING it_entries          TYPE ty_logit_entries
      RETURNING VALUE(rv_token_id) TYPE i.

ENDCLASS.


CLASS zcl_llm_sampler IMPLEMENTATION.

  METHOD sample.
    DATA(lt_logits) = io_logits->get_data( ).

    " Greedy decoding: temperature = 0 or very low
    IF iv_temperature < '0.001'.
      " Find argmax
      DATA(lv_max_val) = CONV f( '-1E38' ).
      DATA(lv_max_idx) = 0.
      DATA(lv_idx) = 0.
      LOOP AT lt_logits INTO DATA(lv_logit).
        IF lv_logit > lv_max_val.
          lv_max_val = lv_logit.
          lv_max_idx = lv_idx.
        ENDIF.
        lv_idx = lv_idx + 1.
      ENDLOOP.
      rv_token_id = lv_max_idx.
      RETURN.
    ENDIF.

    " Step 1: Temperature scaling
    temperature_scale(
      EXPORTING iv_temperature = iv_temperature
      CHANGING  ct_logits      = lt_logits ).

    " Build indexed entries for filtering
    DATA(lt_entries) = VALUE ty_logit_entries( ).
    lv_idx = 0.
    LOOP AT lt_logits INTO lv_logit.
      APPEND VALUE ty_logit_entry(
        index = lv_idx
        value = lv_logit ) TO lt_entries.
      lv_idx = lv_idx + 1.
    ENDLOOP.

    " Sort by value descending for top-K/top-P
    SORT lt_entries BY value DESCENDING.

    " Step 2: Top-K filtering
    IF iv_top_k > 0.
      top_k_filter(
        EXPORTING iv_k = iv_top_k
        CHANGING ct_entries = lt_entries ).
    ENDIF.

    " Step 3: Top-P filtering
    IF iv_top_p < '1.0'.
      top_p_filter(
        EXPORTING iv_p = iv_top_p
        CHANGING ct_entries = lt_entries ).
    ENDIF.

    " Step 4: Random weighted sampling
    rv_token_id = random_sample( lt_entries ).
  ENDMETHOD.

  METHOD temperature_scale.
    LOOP AT ct_logits ASSIGNING FIELD-SYMBOL(<lv_logit>).
      <lv_logit> = <lv_logit> / iv_temperature.
    ENDLOOP.
  ENDMETHOD.

  METHOD top_k_filter.
    " Keep only top K entries (table is already sorted descending)
    IF lines( ct_entries ) > iv_k.
      DELETE ct_entries FROM iv_k + 1.
    ENDIF.
  ENDMETHOD.

  METHOD top_p_filter.
    " Convert to probabilities via softmax, then cumulatively filter
    " Find max for numerical stability
    DATA(lv_max) = ct_entries[ 1 ]-value.

    " Compute softmax probabilities
    DATA(lv_sum) = CONV f( 0 ).
    LOOP AT ct_entries ASSIGNING FIELD-SYMBOL(<ls_entry>).
      <ls_entry>-value = exp( <ls_entry>-value - lv_max ).
      lv_sum = lv_sum + <ls_entry>-value.
    ENDLOOP.

    " Normalize and accumulate
    DATA(lv_cum_prob) = CONV f( 0 ).
    DATA(lv_cutoff) = 0.
    LOOP AT ct_entries ASSIGNING <ls_entry>.
      <ls_entry>-value = <ls_entry>-value / lv_sum.
      lv_cum_prob = lv_cum_prob + <ls_entry>-value.
      lv_cutoff = sy-tabix.
      IF lv_cum_prob >= iv_p.
        EXIT.
      ENDIF.
    ENDLOOP.

    " Keep entries up to cutoff (ensure at least 1)
    IF lv_cutoff < lines( ct_entries ) AND lv_cutoff > 0.
      DELETE ct_entries FROM lv_cutoff + 1.
    ENDIF.
  ENDMETHOD.

  METHOD random_sample.
    " Compute softmax over remaining entries
    DATA(lv_max) = it_entries[ 1 ]-value.
    DATA(lv_sum) = CONV f( 0 ).
    DATA(lt_probs) = VALUE ty_logit_entries( ).

    LOOP AT it_entries INTO DATA(ls_entry).
      DATA(lv_prob) = exp( ls_entry-value - lv_max ).
      lv_sum = lv_sum + lv_prob.
      APPEND VALUE ty_logit_entry(
        index = ls_entry-index
        value = lv_prob ) TO lt_probs.
    ENDLOOP.

    " Normalize
    LOOP AT lt_probs ASSIGNING FIELD-SYMBOL(<ls_prob>).
      <ls_prob>-value = <ls_prob>-value / lv_sum.
    ENDLOOP.

    " Generate random number and sample
    DATA(lo_random) = cl_abap_random_float=>create( seed = CONV i( sy-uzeit ) ).
    DATA(lv_rand) = lo_random->get_next( ).
    DATA(lv_cum) = CONV f( 0 ).

    LOOP AT lt_probs INTO ls_entry.
      lv_cum = lv_cum + ls_entry-value.
      IF lv_cum >= lv_rand.
        rv_token_id = ls_entry-index.
        RETURN.
      ENDIF.
    ENDLOOP.

    " Fallback: return last entry
    rv_token_id = lt_probs[ lines( lt_probs ) ]-index.
  ENDMETHOD.

ENDCLASS.
