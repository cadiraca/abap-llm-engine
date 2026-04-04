"! <p class="shorttext synchronized" lang="en">LLM Engine - BPE Tokenizer Merge Rules</p>
"! Transparent table storing the Byte-Pair Encoding merge rules for a given model.
"! Each row is one BPE merge operation: when the two tokens PAIR_LEFT and PAIR_RIGHT
"! appear adjacent in the token sequence they are merged into a single token.
"! The PRIORITY field determines the order in which merges are tried —
"! lower value = higher priority (applied first), matching the order in tokenizer.json.
"!
"! For SmolLM2-135M there are approximately 48900 merge rules.
@EndUserText.label: 'LLM Engine - BPE Tokenizer Merge Rules'
@AbapCatalog.enhancement.category: #NOT_EXTENSIBLE
@AbapCatalog.tableCategory: #TRANSPARENT
@AbapCatalog.deliveryClass: #APPLICATION_DATA
@AbapCatalog.dataMaintenance: #RESTRICTED
define table zllm_merges {
  key mandt      : mandt not null;
  key model_id   : abap.char(40) not null;
  key priority   : abap.int4 not null;
  pair_left      : abap.string(0);
  pair_right     : abap.string(0);
}
