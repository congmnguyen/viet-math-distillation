# Project Overview

This repository provides tools for fine-tuning language models to enhance their reasoning capabilities, specifically targeting Vietnamese mathematical problem-solving. The project uses Parameter-Efficient Fine-Tuning (PEFT) techniques with QLoRA quantization to train small language models on reasoning-enhanced datasets.

## Architecture

### Core Components

1. **Reasoning Dataset Generation** (`generate_reasoning.py`)
   - Uses LangChain with LlamaCpp to generate step-by-step reasoning
   - Prompts models to explain math problems in Vietnamese (as if to a 6-year-old)
   - Evaluates generated reasoning against ground truth answers
   - Outputs structured data with `<think>` and `<answer>` tags

2. **Training Pipeline** (`train.py`)
   - Fine-tunes models using Supervised Fine-Tuning (SFT) with QLoRA
   - Default base model: TinyLlama-1.1B-intermediate-step-1431k-3T
   - Uses 4-bit quantization for memory efficiency
   - LoRA configuration: r=64, alpha=32, dropout=0.05
   - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

3. **Inference System** (`inference.py`)
   - Loads fine-tuned PEFT models
   - Provides conversational interface with memory (k=3 recent conversations)
   - Uses LangChain for conversation management
   - Generates step-by-step reasoning with final answers

4. **Utility Scripts**
   - `dataset_loading_fix.py`: Handles dataset loading with proper validation
   - `json_fix.py`: Fixes malformed JSON in datasets

## Data Flow

```
Vietnamese Math Dataset (Hugging Face)
    ↓
generate_reasoning.py (adds step-by-step reasoning)
    ↓
Processed JSON with <think> and <answer> tags
    ↓
train.py (fine-tunes with QLoRA)
    ↓
Fine-tuned PEFT model
    ↓
inference.py (interactive Q&A)
```

## Key Technologies

- **Transformers**: Model loading and tokenization
- **PEFT (Parameter-Efficient Fine-Tuning)**: LoRA adapters for efficient training
- **BitsAndBytes**: 4-bit quantization for reduced memory footprint
- **TRL (Transformer Reinforcement Learning)**: SFTTrainer for supervised fine-tuning
- **LangChain**: Prompt engineering and conversation management
- **Datasets**: Hugging Face datasets library

## Training Configuration

### Quantization (Default)
- 4-bit quantization enabled
- Quantization type: NF4
- Compute dtype: float16
- Double quantization: enabled

### LoRA Parameters
- Rank (r): 64
- Alpha: 32
- Dropout: 0.05
- Bias: none
- Task type: CAUSAL_LM

### Training Arguments
- Max sequence length: 2048
- Default epochs: 1
- Optimizer: adamw_torch
- Output dir: TinyLlama-1.1B-qlora-quantization

## Dataset Format

The training data expects JSON with the following structure:
```json
{
  "query_vi": "Vietnamese question",
  "response_vi": "Vietnamese answer",
  "reasoning": "<think>reasoning process</think>\n<answer>final answer</answer>"
}
```

## Prompt Template

The project uses a specialized prompt template that:
- Instructs the model to act as a math expert
- Requires step-by-step explanations in simple Vietnamese
- Enforces structured output with `<think>` and `<answer>` tags
- Formats responses appropriately for 6-year-old understanding level

## Development Notes

### When Modifying Training
- Adjust LoRA parameters in `train.py` if targeting different model sizes
- Modify `max_seq_length` if working with longer reasoning chains
- Update `format_prompt()` function to change instruction formatting

### When Modifying Dataset Generation
- Edit prompt templates in `generate_reasoning.py` for different reasoning styles
- Adjust evaluation criteria in `evaluate_reasoning()` function
- Change seed values for different data shuffling

### When Modifying Inference
- Adjust `max_new_tokens` in pipeline for longer/shorter responses
- Modify temperature for more/less creative outputs
- Change `k` in ConversationBufferWindowMemory for different context windows

## Common Issues

1. **Dataset Loading Errors**: Use `dataset_loading_fix.py` which includes proper field validation
2. **JSON Parsing Errors**: Run `json_fix.py` to clean malformed JSON files
3. **OOM Errors**: Reduce batch size, max_seq_length, or LoRA rank
4. **Model Loading**: Ensure correct model paths and Hugging Face authentication

## File Purposes

- `train.py`: Main training script with QLoRA configuration
- `generate_reasoning.py`: Creates reasoning-enhanced datasets
- `inference.py`: Interactive inference with conversation memory
- `dataset_loading_fix.py`: Robust dataset loading with validation
- `json_fix.py`: Repairs malformed JSON in datasets
- `generate_reasoning.ipynb`: Notebook version of reasoning generation
- `requirements.txt`: Python dependencies

## Git Workflow

- Default branch for development: Branches prefixed with `claude/`
- All commits should be descriptive and atomic
- Push to feature branches for review before merging

## Model Outputs

The fine-tuned models produce structured reasoning in this format:
```
<think>
[Step-by-step reasoning in Vietnamese, explaining the problem-solving process]
</think>
<answer>
[Final answer in Vietnamese]
</answer>
```

This structure helps with:
- Transparency in model reasoning
- Debugging incorrect answers
- Training more robust reasoning capabilities
- Evaluating intermediate steps
