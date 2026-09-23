# Phase 0 Analysis — Ambiguities & Discrepancies

## Confirmed details (spec matches paper)

1. **Equations 5–14** — all match; token weight uses std-only normalization (no mean subtraction)
2. **Layer mapping** — Table 11 matches spec; formula `l_T = floor(l_S * N_T / N_S)`
3. **Hyperparameters** — Tables 8–10 match spec exactly
4. **Main results** — Table 1 matches spec
5. **Eval protocol** — confirmed from MiniLLM: seeds [10,20,30,40,50], temp=1.0, top_p=1.0, top_k=0, do_sample=True
6. **Data format** — max_length=512, max_prompt_length=256 (confirmed from MiniLLM eval scripts)
7. **L_DSA** — averaged over layers (Eq. 10: `1/|L_key|`)
8. **L_Hid** — summed over layers (Eq. 13: `Σ_l`)
9. **Span weights** — computed from teacher hidden states (Eq. 9 uses `w_t^T` superscript)

## Discrepancies between spec and paper

### Minor
1. **Eq. 7 — N normalization**: Paper just says `1/N`. Spec interprets as N_valid.
   → Decision: use N_valid (D02). The alternative is mathematically inconsistent because
   padding columns are masked to −∞.

2. **L_DSA pairwise weights**: Paper Eq. 11 says `w_{ij}^{sp}` without defining it explicitly.
   Spec says `w_i^sp · w_j^sp`. This is the standard pairwise importance weighting.
   → Confirmed from paper text: "product of span weights."

### Potentially significant
3. **Table 8 structure**: Paper has columns for both teacher and student models.
   The spec's table only shows one row set matching student hyperparameters.
   → Not a real discrepancy — the teacher columns show identical values (same LR/batch
   for teacher SFT across all pairs). Spec covers what matters.

4. **OPT LR for FDD/DistiLLM**: Paper Table 8 shows student LR = 5e-4 for OPT-1.3B,
   different from GPT-2's 1e-4. Spec's table only shows GPT-2 values.
   → Addressed: full Table 8 values will go into per-pair configs (gpt2.yaml, opt.yaml).

## Ambiguities not resolvable without access to code

1. **Student initialization** — pretrained vs SFT (D01)
2. **SKL/SRKL alpha** — assumed 0.1 (D09)
3. **DistiLLM loss variant** — SKL or SRKL (D23)
4. **FDD loss weights** — no values in paper (D08)
5. **DistiLLM replay buffer details** — SGO frequency, buffer size
6. **DistiLLM-2 curriculum schedule** — exact alpha/beta update formulas beyond Eq. 18
7. **VP extraction specifics** — paper says "verb phrases" without algorithm (D13)
8. **MTA on DistiLLM-2** — TGO, SGO, or both (D16)

## Proposed implementation plan

### P1 — Skeleton (est. 30 min)
Config schema (OmegaConf dataclasses), CLI parser, requirements, .gitignore,
.gitattributes, README skeleton, empty `__init__.py` files, `pipeline.py --help`.

### P2 — Data pipeline (est. 1.5 hr)
download.py, prepare.py, dataset.py (Dataset + collator), prompt template.
Tests: labels mask prompt+padding, shapes correct.

### P3 — Span extraction (est. 1.5 hr)
spans.py with word/phrase extraction, char→token mapping, caching.
Test: "The red car overtook the truck" golden test.

### P4 — MTA loss + layers (est. 2 hr)
layers.py (mapping logic), mta.py (token weight, span weight, pooling, L_DSA, L_Hid).
Tests: weight properties, gradient flow, zero-loss identity, perf benchmark.

### P5 — Base losses (est. 1.5 hr)
divergences.py (KL, SKL, SRKL chunked+masked), fdd.py, distillm.py, distillm2.py.
Tests: zero-divergence, chunked==non-chunked, mask correctness.

### P6 — Trainer (est. 2 hr)
trainer.py (generic loop), sft.py, checkpoint/resume/best-ckpt logic.
Test: 4-step continuous == 2+2 with resume.

### P7 — Generation + eval (est. 1.5 hr)
backend.py (vLLM/HF), gen_pairs.py, evaluate.py, rouge.py, report.py, judge.py.

### P8 — Pipeline + smoke (est. 1 hr)
pipeline.py orchestrator, setup.sh, run_all.sh, smoke end-to-end on Windows CPU.

### P9 — Multi-pair configs + docs (est. 1 hr)
qwen.yaml, opt.yaml, README completion, DECISIONS review.
