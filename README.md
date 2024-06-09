# Axolotl Showcase

#### A public showcase of configs, WandB logs, and the resulting models built with Axolotl OSS

This resource of public configurations using public datasets exists to help people inspect the datasets and corresponding configurations so that you can more easily grok the relationships between the two.


### Table of Contents

- [Llama-3 6B v0.1](#llama-3-6b-v01)


### Llama-3 6B v0.1

A continued pretrained version of [Prince Canuma](https://huggingface.co/prince-canuma)'s downcycled Llama 3 6B.

- Model: https://huggingface.co/prince-canuma/Llama-3-6B-v0.1
- W&B: https://wandb.ai/prince-canuma/llama-3-6b
- Dataset: https://huggingface.co/datasets/prince-canuma/fineweb-CC-MAIN-2024-10-1B-en
<details><summary>See YAML</summary>


axolotl version: `0.4.0`
```yaml
base_model: prince-canuma/Llama-3-6B-v0.1
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

load_in_8bit: false
load_in_4bit: true
strict: false

datasets:
  - path: prince-canuma/fineweb-CC-MAIN-2024-10-1B-en
    type: completion
    split: train
dataset_prepared_path: last_run_prepared
val_set_size: 0.001
output_dir: ./llama-3-6b
save_safetensors: true
adapter: qlora
lora_model_dir:

sequence_len: 8192
sample_packing: false
pad_to_sequence_len: false

lora_r: 128
lora_alpha: 128
lora_dropout: 0.05
lora_target_modules:
lora_target_linear: true
lora_fan_in_fan_out:


wandb_project: llama-3-6b
wandb_entity: 
wandb_watch:
wandb_name:
wandb_log_model:

gradient_accumulation_steps: 8
micro_batch_size: 2
num_epochs: 2
optimizer: paged_adamw_32bit
lr_scheduler: cosine
learning_rate: 2e-4

train_on_inputs: false
group_by_length: false
bf16: auto
fp16:
tf32: false

gradient_checkpointing: true
early_stopping_patience:
resume_from_checkpoint:
local_rank:
logging_steps: 1
xformers_attention:
flash_attention: true

warmup_steps: 100
evals_per_epoch: 4
eval_table_size:
save_steps: 4000
debug:
deepspeed:
weight_decay: 0.0
fsdp:
fsdp_config:
special_tokens:
    pad_token: "<|reserved_special_token_0|>"


```

</details>