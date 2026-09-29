# Paper -> code map

Reference: `iclr2027_docpatch.pdf`, "DocPatch: Continual Knowledge
Internalization with Conflict-Aware LoRA Memories" (ICLR 2027 submission).

| Paper section / equation | What it specifies | Code |
|---|---|---|
| 3.1 Overview, Fig. 1 | Four components: knowledge bank, router, conflict resolution, reasoning adapter | `docpatch/pipeline.py` (`DocPatchPipeline`) wires all four |
| 3.2, Eq. 1, `M = {(L_i, z_i, mu_i)}` | Offline knowledge bank: chunk -> LoRA via frozen D2L hypernetwork H, cached with feature `z_i` and metadata `mu_i` | `docpatch/knowledge_bank.py` (`KnowledgeBank`); reuses `ctx_to_lora.modeling.hypernet.ModulatedPretrainedModel` as H and `ctx_to_lora.docpatch.chunking.chunk_document` for chunking |
| 3.3, Eq. 2-3 | Router score `s_i = R(e_q, z_i)`, softmax `p_i` | `ctx_to_lora.docpatch.router.SourceAwareRouter` (aliased as `docpatch.router.DocPatchRouter`) |
| 3.3, Eq. 4-5 | Normalized multi-hot chunk target `y_i`, loss `L_chunk` | `docpatch/router.py`: `build_multi_hot_targets`, `compute_chunk_selection_loss` |
| 3.3, Eq. 6-8 | Source-level probability `p_d`, `L_source`, `L_router = L_chunk + lambda_source L_source` | `docpatch/router.py`: `compute_source_probabilities`, `compute_source_loss`, `compute_router_loss` |
| 3.4, Eq. 9-10 | Conflict-group applicability mask `m_i`, composition weight `g_i` | `docpatch/conflict_resolution.py`: `compute_applicability_mask`, `resolve_conflicts` |
| 3.5, Eq. 11-12 | Composed update `Delta W_know`, exact rank-concat construction | `ctx_to_lora.docpatch.lora_composition.concatenate_lora_trees`, wrapped by `docpatch/composition.py::compose` |
| 3.6, Eq. 13-14 | `h' = W_0 h + Delta W_know h + Delta W_reason h` | `docpatch/reasoning_adapter.py`: `ReasoningAdapter.compose_with_knowledge`, `apply_reasoning_and_knowledge` |
| 3.6, Eq. 15 | Answer-level loss `L_answer`, reasoning adapter is the only trainable part | `docpatch/training/train_reasoning_adapter.py` |
| 3.6, staged curriculum (closing paragraphs) | ground-truth -> counterfactual -> multi-LoRA -> conflict-resolved | `docpatch/data/curriculum.py::ReasoningCurriculum` |
| Table 1 | QA EM/F1 vs. Frozen LLM / RAG / D2L baselines | `docpatch/evaluation/evaluate_qa.py` |
| Table 2, Eq. 16 | Routing Recall / F1 / Accuracy@top-5 | `docpatch/evaluation/metrics.py::routing_metrics`, `docpatch/evaluation/evaluate_router.py` |
| Table 3, Eq. 17-18 | Uniform / router-weighted composition baselines | `docpatch/composition.py::compose(mode="uniform"\|"router_weighted")` |
| Table 4 | Continual updating: Unchanged/Updated/Old-Only/Override/Unrelated | `docpatch/evaluation/evaluate_continual.py`, `docpatch/evaluation/metrics.py::continual_update_accuracy` |
| Table 5 | Ablations: w/o reasoning adapter, top-1 only, uniform weights, w/o counterfactual training | `evaluate_qa.py --mode {docpatch_no_reasoning,top1_only,uniform_composition}`; "w/o counterfactual" = train with `docpatch/data/curriculum.py` stage 2 skipped |
| Table 6 | Input tokens / latency / peak GPU memory | `docpatch/evaluation/measure_efficiency.py` |
| Appendix A.1, Eq. 19-25 | Exact composition derivation; naive independent-factor sum introduces cross terms | `docpatch/composition.py::verify_exact_composition` (unit-tests the Eq. 25 identity); `docpatch/tests/test_composition.py` |
| Appendix A.2, Eq. 26-28 | Query representation: frozen LLM -> pooled hidden state -> learned `P_q` | `docpatch/query_encoder.py`: `QueryEncoder`, `QueryProjector` |
| Appendix A.3 | LoRA-to-feature sketch: fixed `P_in`/`P_out`, learned `P_LoRA` | `docpatch/lora_sketch.py`: `FixedRandomProjections`, `compute_lora_sketch`, `LoRAFeatureProjector` |
| Appendix A.4 | Constitution QA construction | Pre-existing: `data_preprocess/build_constitution_qa.py`, `data_preprocess/outputs/constitution_qa/` |
| Appendix A.5 | Conflict identification vs. resolution split; conflict-group + version metadata assumed given | `docpatch/metadata.py::infer_conflict_group` (derives groups for the synthetic dataset's `fact_slot` annotation); `docpatch/conflict_resolution.py` (resolution only, never re-detects conflicts) |
| Appendix A.6 | Counterfactual pairs: same question, different active LoRA/target | `docpatch/data/counterfactual.py::build_counterfactual_pairs` |
| Appendix A.7 | Storage vs. inference-time trade-off discussion | No code needed; `docpatch/evaluation/measure_efficiency.py` reports the inference-time side |
| Appendix A.8 | Multi-level router supervision (chunk + source) | Same as Eq. 4-8 above |

## What this package reuses vs. adds

`docpatch/` is built **on top of** the pre-existing `ctx_to_lora.docpatch.*`
scaffolding rather than duplicating it. Reused as-is:

- `ctx_to_lora.modeling.hypernet.ModulatedPretrainedModel` -- the frozen,
  pretrained Doc-to-LoRA hypernetwork H (Eq. 1), including the `qwen_4b_d2l`
  checkpoint the paper uses as its Qwen3-4B backbone.
- `ctx_to_lora.docpatch.chunking`, `.cache`, `.chunk_store`, `.data` --
  chunking, cache-key/manifest bookkeeping, and the `DocPatchDataset` /
  `SourceChunkRef` types used throughout `docpatch/`.
- `ctx_to_lora.docpatch.router.SourceAwareRouter` -- the router architecture
  itself (the scorer `R` in Eq. 2). `docpatch/router.py` only adds the
  paper's exact multi-hot/source losses on top.
- `ctx_to_lora.docpatch.lora_composition.concatenate_lora_trees` -- the exact
  rank-concatenation primitive (Eq. 12/23-25). `docpatch/composition.py` and
  `docpatch/reasoning_adapter.py` are thin, paper-notation wrappers around it.

Added fresh in `docpatch/` (these did not exist anywhere in the repo before):

- Explicit conflict-group-id + version-precedence resolution over a
  *retrieved* candidate set (Eq. 9-10) -- the existing
  `ctx_to_lora.docpatch.conflict_detection` only does pairwise regex-based
  fact-conflict *detection* for synthetic-data generation, which is a
  different (and, per Appendix A.5, explicitly out-of-scope-for-this-paper)
  problem from *resolving* conflicts once conflict-group metadata is known.
- The global reasoning adapter end to end: module, composition-as-addition,
  answer-level training loop, and staged curriculum
  (`reasoning_adapter.py`, `training/train_reasoning_adapter.py`,
  `data/curriculum.py`, `data/counterfactual.py`).
- The paper-exact query encoder (Appendix A.2) and LoRA sketch feature
  (Appendix A.3) -- the pre-existing code used a `HashTextEmbedder` and
  hand-crafted tensor statistics as simpler stand-ins for these.
- The end-to-end inference pipeline (`pipeline.py`) and the full evaluation
  suite reproducing Tables 1-6 (`evaluation/`).
