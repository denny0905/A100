# MTA — Multi-Granular Trajectory Alignment for LLM Distillation

Reproduction of [arXiv 2605.01374v2](https://arxiv.org/abs/2605.01374v2).

## Development (Windows, CPU)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python -m spacy download en_core_web_sm

# Run tests
python -m pytest tests/ -x -q

# Smoke test (CPU, <10 minutes)
python -m mta.pipeline --config configs/smoke.yaml
```

## Server (Linux, GPU)

> **Important:** Install vLLM *first* — it pins its own torch/CUDA versions.

```bash
git clone <repo-url> && cd MTA
bash setup.sh                          # default: configs/gpt2.yaml
bash run_all.sh configs/gpt2.yaml
# or: bash setup.sh configs/opt.yaml && bash run_all.sh configs/opt.yaml
```

`setup.sh` creates the venv, installs deps, downloads data, and precomputes
spans. `run_all.sh` runs the full pipeline (SFT teacher → 7 distillation
experiments → evaluation → report) and prints the results summary when done.
Add `--shutdown` to power off after completion.

### Custom runs

```bash
# Single experiment
python -m mta.pipeline --config configs/gpt2.yaml --experiments distillm+mta

# Resume from a stage
python -m mta.pipeline --config configs/gpt2.yaml --from train

# Override any config value
python -m mta.pipeline --config configs/gpt2.yaml train.lr=5e-5 train.epochs=10
```

## Model pairs

| Config | Teacher | Student | Fine-tuning |
|--------|---------|---------|-------------|
| `configs/gpt2.yaml` | GPT-2 XL 1.5B | GPT-2 120M | Full |
| `configs/qwen.yaml` | Qwen1.5-1.8B | Qwen1.5-0.5B | Full |
| `configs/opt.yaml` | OPT-6.7B | OPT-1.3B | LoRA |

## Decisions

All assumptions not covered by the paper are documented in
[`docs/DECISIONS.md`](docs/DECISIONS.md) with rationale and config overrides.
