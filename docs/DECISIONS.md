# Decisions & Assumptions Log

Every item the paper does not fully specify is recorded here with the chosen
default, the rationale, and how to override it via config.

---

## D01 — Student initialization: pretrained (not SFT)

**Issue:** Paper does not state whether the student starts from a pretrained
checkpoint or an SFT checkpoint before distillation.

**Choice:** Default `student_init: "pretrained"`.  DistiLLM repo scripts
reference an "init" phase that selects by validation loss, suggesting SFT init
is at least an option.  However, Table 1 lists SFT as a separate baseline, and
using SFT init would conflate the SFT contribution with distillation gains.

**Override:** `student_init: "sft"` in config; the trainer will then load the
SFT checkpoint produced by the `sft_student` stage.

**Impact:** Medium — SFT init typically gives the student a head start.

---

## D02 — Token weight: N = number of valid (non-padding) tokens

**Issue:** Eq. 7 says `w_t = (1/N) Σ_s α_{s→t}`.  Appendix C defines the mask
M that sets padding columns to −∞ but does not clarify whether N is the full
sequence length or just the valid count.

**Choice:** `N = N_valid` (non-padding tokens).  Padding columns are masked to
−∞ so their attention is 0; using full N would under-scale the valid weights.
This also ensures Σ_t w_t ≈ 1 over valid tokens.

**Override:** Not configurable (the alternative is mathematically inconsistent).

**Impact:** Low — only affects weight magnitude, not ranking.

---

## D03 — Token weight: padding rows zeroed out

**Issue:** Paper only masks padding *columns* (Eq. 21).  Rows corresponding to
padding tokens still attend to valid tokens after softmax.

**Choice:** Zero out the entire row for padding tokens after softmax, so padding
tokens contribute 0 to every other token's weight.

**Override:** Not configurable.

**Impact:** Low.

---

## D04 — Span pooling weights: use teacher weights for student pooling

**Issue:** Eq. 8 uses `w_t` without subscript indicating teacher or student.
Using teacher weights means both teacher and student spans are pooled identically;
using student weights introduces student's own importance estimate.

**Choice:** Default `mta.student_pool_weights: "teacher"`.  This ensures span
representations are comparable and the structural alignment is not confounded by
differing weight distributions.

**Override:** `mta.student_pool_weights: "student"` (weights are detached).

**Impact:** Medium — affects span representations and thus L_DSA gradients.

---

## D05 — FDD logit lens: apply final_norm before lm_head

**Issue:** Eq. 3 says `y_l = log f_head(h_l)`.  It's ambiguous whether
`final_norm` (LayerNorm / RMSNorm) is applied before the LM head.

**Choice:** Default `fdd.apply_final_norm: true`.  Without the norm,
intermediate hidden states have different scale than the last layer, producing
poor logit-space representations.

**Override:** `fdd.apply_final_norm: false`.

**Impact:** High — strongly affects FDD trajectory quality.

---

## D06 — FDD layer set defaults to MTA key layers

**Issue:** The FDD paper (Gong et al.) does not have public code as of writing.
MTA paper uses FDD as a base but does not specify which layers FDD itself uses.

**Choice:** Default `fdd.layers` = same as `mta.word_layers + mta.phrase_layers`.
This is consistent with MTA applying on top of FDD's layer set.

**Override:** `fdd.layers: [list of student layer indices]`.

**Impact:** Medium.

---

## D07 — FDD: include final-layer KD loss

**Issue:** Standard KD includes KL on the final logits.  It's unclear if FDD
replaces or supplements this.

**Choice:** Default `fdd.include_final_kd: true` — add standard KL on the last
layer's logits alongside trajectory and derivative losses.

**Override:** `fdd.include_final_kd: false`.

**Impact:** Medium.

---

## D08 — FDD loss weights

**Issue:** Paper does not provide coefficients for `L_Traj` and `L_Der`.

**Choice:** Default `fdd.lambda_traj: 1.0`, `fdd.lambda_der: 1.0`,
`fdd.lambda_kd: 1.0`.  Equal weighting as a starting point.

**Override:** `fdd.lambda_traj`, `fdd.lambda_der`, `fdd.lambda_kd`.

**Impact:** High — needs tuning if results don't match.

---

## D09 — SKL/SRKL alpha default = 0.1

**Issue:** Paper references DistiLLM's skew divergences but does not restate α.

**Choice:** `distillm.alpha: 0.1`, matching the default commonly used in
DistiLLM implementations.

**Override:** `distillm.alpha`.

**Impact:** Medium.

---

## D10 — Warmup steps = 0

**Issue:** Paper does not mention warmup.

**Choice:** `train.warmup_steps: 0` (also supports `train.warmup_ratio`).

**Override:** config key.

**Impact:** Low–Medium.

---

## D11 — Gradient clipping = 1.0

**Issue:** Paper does not specify gradient clipping.

**Choice:** `train.max_grad_norm: 1.0` — standard default for LLM training.

**Override:** config key; set to `null` to disable.

**Impact:** Low.

---

## D12 — Optimizer: AdamW

**Issue:** Paper does not name the optimizer.

**Choice:** `train.optimizer: "adamw"`, `train.weight_decay: 0.01`,
`train.betas: [0.9, 0.999]`.

**Override:** config keys.

**Impact:** Low.

---

## D13 — VP extraction algorithm

**Issue:** Paper says "verb phrases" without specifying the extraction algorithm.

**Choice:** For each VERB token in spaCy parse:
1. Extend left to include contiguous `neg`, `aux`, `auxpass` dependents.
2. Extend right to the rightmost subtree end of `dobj/obj/attr/prt/dative/acomp/oprd` children.
3. If no such children, VP = verb alone.
4. Cap at `spans.max_vp_words: 12` words; if exceeded, keep verb + `prt` only.
5. Remove NPs fully contained within a VP (keep spans disjoint).

**Override:** `spans.max_vp_words`, `spans.allow_overlap`.

**Impact:** Medium — span quality affects MTA alignment.

---

## D14 — Word spans: exclude punctuation and whitespace-only tokens

**Issue:** Paper says "word spans" but doesn't define filtering.

**Choice:** Skip spaCy tokens where `is_punct` or `is_space` is True.

**Override:** Not configurable (punctuation alignment is unlikely to matter).

**Impact:** Low.

---

## D15 — L_Hid: no renormalization of token weights on M_l

**Issue:** Eq. 13 sums `w_t^T` over tokens in M_l (tokens covered by spans).
It's unclear whether weights should be renormalized so they sum to 1 over M_l.

**Choice:** Default `mta.hid_renorm: false` — use raw teacher token weights.
Renormalizing would upweight span-covered tokens and change the loss scale.

**Override:** `mta.hid_renorm: true`.

**Impact:** Low–Medium.

---

## D16 — DistiLLM-2 + MTA: apply MTA on both TGO and SGO, average

**Issue:** Paper says MTA is a plug-in on top of base losses but doesn't specify
whether MTA loss is computed on TGO forward, SGO forward, or both for CALD.

**Choice:** Compute MTA loss on both TGO and SGO forwards, average the two
contributions: `L_MTA = (L_MTA_tgo + L_MTA_sgo) / 2`.

**Override:** `distillm2.mta_on: "both"` | `"tgo"` | `"sgo"`.

**Impact:** Medium.

---

## D17 — SFT teacher training

**Issue:** Paper reports teacher performance but does not describe how the
teacher was fine-tuned.

**Choice:** `teacher.lr: 5e-5`, `teacher.epochs: 10`, `teacher.batch_size: 16`,
cosine scheduler, select best by valid ROUGE-L.  Can be skipped entirely with
`teacher.path` pointing to an existing checkpoint.

**Override:** All values in `teacher.*` section.

**Impact:** High — teacher quality affects all distillation experiments.

---

## D18 — Evaluation: max_new_tokens

**Issue:** MiniLLM eval uses `max_length: 512` total (prompt + generation).
This means max_new_tokens = 512 − prompt_length, varying per sample.

**Choice:** Use `max_length: 512` (not a fixed `max_new_tokens`), matching
MiniLLM protocol exactly.

**Override:** `eval.max_length`, `eval.max_new_tokens`.

**Impact:** Medium — longer generations can inflate ROUGE.

---

## D19 — Best checkpoint selection: greedy decoding on dolly_valid

**Issue:** Using 5-seed sampling for checkpoint selection at every epoch is
expensive.

**Choice:** Evaluate with greedy decoding (do_sample=False) on dolly_valid for
speed; final evaluation uses 5-seed sampling per MiniLLM protocol.

**Override:** `train.val_do_sample: true` to use sampling for validation too.

**Impact:** Low — greedy ROUGE correlates well with sampled ROUGE for selection.

---

## D20 — Token weight H standardization: no mean subtraction

**Issue:** Eq. 5/19 says `Ĥ = H / σ(H)` — divides by std but does NOT subtract
mean, unlike standard LayerNorm.

**Choice:** Implement exactly as paper states: divide by std(dim=-1), no mean
subtraction.

**Override:** Not configurable (follows paper exactly).

**Impact:** N/A — matches paper.

---

## D21 — Prompt template: MiniLLM format

**Issue:** Confirmed from MiniLLM repo structure; exact template needs final
verification.

**Choice:**
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:

```
Omit `### Input:` block when input is empty.

**Override:** `data.prompt_template` (path to a Jinja2 or Python format string).

**Impact:** High — prompt format must match teacher's training for correct eval.

---

## D22 — Dolly train/valid/test split

**Issue:** Exact split sizes vary across repos.

**Choice:** Follow DistiLLM/MiniLLM processed data if available (priority 1).
Fallback: random split with seed 42, roughly 11.4k/1k/500.

**Override:** `data.split_seed`, `data.valid_size`, `data.test_size`.

**Impact:** Medium — different splits give different numbers.

---

## D23 — DistiLLM loss type: SRKL

**Issue:** DistiLLM supports both SKL and SRKL.  Paper does not specify which
variant is used in the MTA experiments.

**Choice:** Default `distillm.loss_type: "srkl"` — Skew Reverse KL is the
recommended variant in the original DistiLLM paper.

**Override:** `distillm.loss_type: "skl"`.

**Impact:** Medium.

---

## D24 — Span extraction covers full sequence (prompt + response)

**Issue:** Paper says "spans are extracted over the entire input-output sequence."

**Choice:** Extract spans from the full concatenated `prompt + output` text.
The MTA loss itself uses all spans regardless of position.

**Override:** Not configurable (matches paper statement directly).

**Impact:** Confirmed by paper.

---

## D25 — Max spans per sample = 256

**Issue:** Paper does not cap the number of spans.

**Choice:** `spans.max_spans: 256` per type (word/phrase) per sample.  If
exceeded, keep the earliest spans and log a warning.

**Override:** config key.

**Impact:** Low — 256 is generous for 512-token sequences.
