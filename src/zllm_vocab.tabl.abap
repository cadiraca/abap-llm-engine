"! <p class="shorttext synchronized" lang="en">LLM Engine - BPE Tokenizer Vocabulary</p>
"! Transparent table storing the Byte-Pair Encoding vocabulary for a given model.
"! Each row maps an integer token ID to its string representation, as loaded from
"! the HuggingFace tokenizer.json file.  The SCORE field stores the token's
"! log-probability from the sentencepiece / unigram model if present.
"!
"! For SmolLM2-135M the vocabulary has 49152 entries.
"! The table is keyed on MODEL_ID + TOKEN_ID so multiple model vocabularies
"! can coexist in the same client.
@EndUserText.label: 'LLM Engine - BPE Tokenizer Vocabulary'
@AbapCatalog.enhancement.category: #NOT_EXTENSIBLE
@AbapCatalog.tableCategory: #TRANSPARENT
@AbapCatalog.deliveryClass: #APPLICATION_DATA
@AbapCatalog.dataMaintenance: #RESTRICTED
define table zllm_vocab {
  key mandt    : mandt not null;
  key model_id : abap.char(40) not null;
  key token_id : abap.int4 not null;
  token        : abap.string(0);
  score        : abap.fltp;
}
