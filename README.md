# tune-reason — Vietnamese Math Reasoning for a 1.1B Model

Distill step-by-step math reasoning from **Phi-3** (teacher) into **TinyLlama-1.1B** (student) with QLoRA, on Vietnamese math problems. The student learns to emit structured `<think>…</think>` / `<answer>…</answer>` outputs.

## Pipeline

```
5CD-AI/Vietnamese-395k-meta-math-MetaMathQA  (395k Vietnamese math problems)
        │
        ▼
generate_reasoning.py   Phi-3-mini (GGUF, llama.cpp) writes step-by-step Vietnamese
        │               solutions in <think>/<answer> format
        │               → an LLM judge checks each solution against the gold answer
        │                 and keeps only the correct ones
        ▼
train.py                QLoRA fine-tune of TinyLlama-1.1B on the filtered traces
        ▼
inference.py            merge LoRA adapters and chat with the tuned model
```

## Files

| File | Purpose |
|---|---|
| `generate_reasoning.py` | Teacher generation + LLM-as-judge filtering of reasoning traces |
| `train.py` | QLoRA fine-tuning of `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` |
| `inference.py` | Load + merge the LoRA adapter, run interactive inference (LangChain memory) |
| `dataset_loading_fix.py`, `json_fix.py` | Helpers for loading/repairing the generated JSON dataset |
| `generate_reasoning.ipynb` | Notebook version of the generation step |

## Setup

```bash
git clone https://github.com/congmnguyen/tune-reason.git
cd tune-reason
pip install -r requirements.txt
```

Requirements: Python 3.8+, PyTorch 2.0+, a CUDA GPU for training (QLoRA on 1.1B fits consumer GPUs); `llama-cpp-python` runs the GGUF teacher for data generation.

## Usage

```bash
# 1. Generate + filter reasoning data (needs Phi-3-mini GGUF weights locally)
python generate_reasoning.py

# 2. Fine-tune TinyLlama-1.1B with QLoRA
python train.py

# 3. Chat with the tuned model
python inference.py
```

## Notes

- Prompts constrain answers to Vietnamese, "explain like I'm 6" style, with reasoning inside `<think>` tags — the judge only sees the `<answer>` span when comparing to the gold answer.
- The LoRA adapter output dir defaults to `TinyLlama-1.1B-qlora-quantization` (see `train.py`).
