# MTA — Multi-Granular Trajectory Alignment for LLM Distillation

## What this is
Reproduction of arXiv 2605.01374v2.  Distill large LLMs into small ones using
token- and span-level trajectory alignment on top of existing KD methods (FDD,
DistiLLM, DistiLLM-2).

## Quick commands
```bash
# Windows dev — install + run tests
pip install -r requirements.txt
python -m pytest tests/ -x -q

# Smoke test (CPU, <10 min)
python -m mta.pipeline --config configs/smoke.yaml

# Server (Linux GPU)
bash setup.sh
bash run_all.sh configs/gpt2.yaml
```

## Project layout
```
src/mta/          — all source code (package)
  config.py       — YAML + OmegaConf schema, CLI overrides
  pipeline.py     — stage orchestrator (data→spans→train→eval→report)
  data/           — download, prepare, dataset/collator, span extraction
  models/         — teacher/student loading, layer mapping
  losses/         — MTA, FDD, DistiLLM, DistiLLM-2, divergences
  train/          — generic trainer loop, SFT
  generation/     — vLLM/HF generate backend, gen_pairs for DistiLLM-2
  eval/           — ROUGE-L, evaluate, report, optional GPT-4 judge
  utils/          — seed, logging, I/O
configs/          — YAML configs (base, smoke, gpt2, qwen, opt)
tests/            — unit + integration tests
docs/             — SPEC.md, DECISIONS.md, paper PDF
```

## Conventions
- Python ≥ 3.10, type hints, English docstrings.
- Paths via `pathlib`; no hard-coded `/` or `\`.
- Config: YAML + OmegaConf; CLI override `key=value`.
- `logging` module, never `print`.
- All loss computation in fp32; model forward in bf16 autocast (fp32 on CPU).
- MTA loss must be fully vectorized — no Python loops over spans or tokens.
- `DataLoader.num_workers` configurable; 0 on Windows/smoke.
- `vllm` and `bitsandbytes` are optional (Linux-only); graceful fallback.
- `*.sh` files use LF line endings (enforced by `.gitattributes`).

## Test commands
```bash
python -m pytest tests/test_spans.py -v
python -m pytest tests/test_mta_loss.py -v
python -m pytest tests/test_layers.py -v
python -m pytest tests/test_divergences.py -v
python -m pytest tests/test_resume.py -v
python -m pytest tests/test_smoke_pipeline.py -v
```

## Key decisions
See `docs/DECISIONS.md` for every assumption with rationale and how to change it.
