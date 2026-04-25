import os

# Reduce memory fragmentation and random OOMs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import itertools
from pathlib import Path
from typing import Any

import torch
from config import CustomLoraConfig, CustomSFTConfig, EvalConfig
from peft import PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback
from trl import SFTConfig, SFTTrainer

import wandb

# ---------------------------------------------------------------------------
# W&B authentication — set WANDB_API_KEY in your environment before running:
#   export WANDB_API_KEY=your40charkey
# ---------------------------------------------------------------------------
_wandb_key = os.environ.get("WANDB_API_KEY")
if _wandb_key:
    wandb.login(key=_wandb_key)
else:
    # Disable W&B silently if no key is provided (training still runs fine)
    os.environ["WANDB_MODE"] = "disabled"
    print("[INFO] WANDB_API_KEY not set — W&B logging disabled.")
from datasets import Dataset, concatenate_datasets, load_dataset

# ---------------------------------------------------------------------------
# Per-model batch sizes (match other training scripts)
# ---------------------------------------------------------------------------

MODEL_NAME_TO_BATCH_SIZE = {
    "meta-llama/Llama-3.1-8B-Instruct": 4,
    "google/gemma-2-9b-it": 4,
    "google/gemma-2-27b-it": 4,
    "Qwen/Qwen3-14B": 8,
    "Qwen/Qwen3-8B": 8,
    "mistralai/Mistral-Small-24B-Instruct-2501": 1,
    "Qwen/Qwen3-32B": 8,
    "meta-llama/Llama-3.3-70B-Instruct": 8,
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def print_trainable_parameters(model) -> None:
    total = trainable = lora_trainable = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            if "lora_" in name:
                lora_trainable += n
    pct = 100 * trainable / total if total else 0.0
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
    if lora_trainable:
        print(f"  LoRA trainable subset: {lora_trainable:,}")


def create_assistant_mask(
    messages: list[dict[str, str]], tokenizer: AutoTokenizer
) -> dict[str, torch.Tensor]:
    """
    Build input_ids and an assistant_masks tensor (1 = loss, 0 = no loss).

    Works for any chat-format tokenizer by comparing the length of the
    prompt-only tokenization vs. the full conversation tokenization.
    The gender dataset is single-turn (user, assistant), so we only need
    to handle that case.
    """
    assert len(messages) == 2, f"Expected 2 messages (user + assistant), got {len(messages)}"
    assert messages[0]["role"] == "user" and messages[1]["role"] == "assistant"

    chat_kwargs = dict(
        tokenize=True,
        return_tensors=None,
        padding=False,
        enable_thinking=False,  # no chain-of-thought for target models
    )

    # Prompt-only length (where assistant reply starts)
    prompt_ids = tokenizer.apply_chat_template(
        [messages[0]],
        add_generation_prompt=True,
        **chat_kwargs,
    )

    # Full conversation
    full_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        **chat_kwargs,
    )

    assistant_start = len(prompt_ids)
    mask = torch.zeros(len(full_ids), dtype=torch.long)
    mask[assistant_start:] = 1

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "assistant_masks": mask,
    }


def prepare_sft_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """Tokenize every example and drop all columns except input_ids / assistant_masks."""
    remove_cols = [c for c in dataset.column_names if c not in {"messages"}]
    ds = dataset.map(
        lambda ex: create_assistant_mask(ex["messages"], tokenizer),
        remove_columns=remove_cols,
        desc="Tokenising with chat template",
    )
    return ds.remove_columns(["messages"])


def combine_with_ultrachat(
    raw_gender_ds: Dataset,
    tokenized_gender_ds: Dataset,
    chat_dataset_name: str,
    tokenizer: AutoTokenizer,
    random_seed: int,
) -> Dataset:
    """
    Sample N examples from UltraChat (first turn only, length-filtered to match
    the gender dataset) and concatenate + shuffle with the main training data.

    This prevents catastrophic forgetting of general chat ability.
    """
    num_target = len(tokenized_gender_ds)
    print(f"Sampling {num_target} examples from UltraChat ({chat_dataset_name})")

    # Max character length from the gender dataset (so UltraChat isn't too long)
    def char_len(ex):
        return sum(len(m["content"]) for m in ex["messages"])

    max_chars = max(char_len(ex) for ex in raw_gender_ds)
    print(f"Max character length in gender dataset: {max_chars}")

    chat_ds = load_dataset(chat_dataset_name, split="train_sft", streaming=True)
    kept: list[dict] = []

    for ex in chat_ds:
        msgs = ex["messages"]
        if len(msgs) < 2:
            continue
        truncated = msgs[:2]
        if sum(len(m["content"]) for m in truncated) <= max_chars:
            kept.append({"messages": truncated})
        if len(kept) >= num_target:
            break

    print(f"UltraChat examples kept: {len(kept)}")
    chat_tok = prepare_sft_dataset(Dataset.from_list(kept), tokenizer)

    combined = concatenate_datasets([tokenized_gender_ds, chat_tok])
    combined = combined.shuffle(seed=random_seed)
    print(
        f"Combined dataset: {len(combined)} total"
        f" ({len(tokenized_gender_ds)} gender + {len(chat_tok)} UltraChat)"
    )
    return combined


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------


def train_with_sft_only(
    sft_train_ds: Dataset,
    sft_eval_ds: Dataset,
    wandb_project: str,
    config: EvalConfig,
    sft_config: SFTConfig,
    callbacks: list[TrainerCallback],
    save_lora_path: Path | None = None,
    load_lora_path: Path | None = None,
    quantize: bool = False,
) -> None:
    torch.manual_seed(config.random_seed)
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    llm_kwargs: dict[str, Any] = dict(
        pretrained_model_name_or_path=config.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )

    if quantize:
        llm_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(**llm_kwargs)
    model.enable_input_require_grads()
    model.use_cache = False
    model.gradient_checkpointing_enable()

    if load_lora_path is not None:
        assert load_lora_path.exists(), f"LoRA path does not exist: {load_lora_path}"
        model = PeftModel.from_pretrained(model, load_lora_path, is_trainable=True)
    else:
        model = get_peft_model(model, CustomLoraConfig())

    print_trainable_parameters(model)

    trainer = SFTTrainer(
        model=model,
        train_dataset=sft_train_ds,
        eval_dataset=sft_eval_ds,
        args=sft_config,
        callbacks=callbacks,
    )

    if trainer.is_world_process_zero():
        wandb.init(
            project=wandb_project,
            name=f"sft_{config.model_name}_gender{config.wandb_info}",
        )

    trainer.train()

    if trainer.is_world_process_zero():
        if save_lora_path is not None:
            trainer.save_model(str(save_lora_path))
        wandb.finish()

    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ---- Config -------------------------------------------------------
    model_names = [
        "Qwen/Qwen3-8B",
        # "google/gemma-2-9b-it",   # uncomment to train on Gemma too
        # "meta-llama/Llama-3.3-70B-Instruct",
    ]

    # HuggingFace datasets with user-gender conversations.
    # Each dataset has a `messages` column: list of {role, content} dicts.
    # "gender_label" is used purely to name the saved LoRA.
    gender_datasets = [
        ("bcywinski/user-gender-male-merged", "male"),
        ("bcywinski/user-gender-female-merged", "female"),
    ]

    chat_dataset_name = "HuggingFaceH4/ultrachat_200k"  # for regularisation mix
    mix_with_ultrachat = True   # set False to skip the UltraChat mix
    final_message_loss_only = True  # only compute loss on the assistant turn

    # ---- Training loop ------------------------------------------------
    for model_name, (dataset_hf_name, gender_label) in itertools.product(
        model_names, gender_datasets
    ):
        print(f"\n{'=' * 60}")
        print(f"Model : {model_name}")
        print(f"Gender: {gender_label}  ({dataset_hf_name})")
        print(f"{'=' * 60}")

        # Local save path, e.g. model_lora/Qwen3-8B-user-male
        model_short = model_name.split("/")[-1]
        lora_name = f"{model_short}-user-{gender_label}"
        config = EvalConfig(model_name=model_name, model_lora_dir="model_lora")
        lora_path = Path(config.model_lora_dir) / lora_name

        if lora_path.exists():
            print(f"{lora_path} already exists — skipping.")
            continue

        torch.cuda.empty_cache()
        gc.collect()

        # ---- Dataset --------------------------------------------------
        ds = load_dataset(dataset_hf_name, split="train")
        print(f"Loaded {len(ds)} examples from {dataset_hf_name}")

        eval_frac = 0.05
        train_size = int(len(ds) * (1 - eval_frac))
        raw_train_ds = ds.select(range(train_size))
        raw_eval_ds = ds.select(range(train_size, len(ds)))

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        train_ds = prepare_sft_dataset(raw_train_ds, tokenizer)
        eval_ds = prepare_sft_dataset(raw_eval_ds, tokenizer)

        if mix_with_ultrachat:
            train_ds = combine_with_ultrachat(
                raw_gender_ds=raw_train_ds,
                tokenized_gender_ds=train_ds,
                chat_dataset_name=chat_dataset_name,
                tokenizer=tokenizer,
                random_seed=config.random_seed,
            )

        # ---- SFT config -----------------------------------------------
        batch_size = MODEL_NAME_TO_BATCH_SIZE.get(model_name, 4)
        real_batch_size = 8

        sft_config = CustomSFTConfig(
            model_name=model_name,
            batch_size=batch_size,
            real_batch_size=real_batch_size,
        )
        sft_config.num_train_epochs = 10.0

        # Evaluate / save twice per epoch
        eval_frequency = max(1, len(train_ds) // (real_batch_size * 2))
        sft_config.eval_steps = eval_frequency
        sft_config.save_steps = eval_frequency

        early_stopping = EarlyStoppingCallback(early_stopping_patience=2)

        # ---- Train ----------------------------------------------------
        train_with_sft_only(
            sft_train_ds=train_ds,
            sft_eval_ds=eval_ds,
            wandb_project=config.wandb_project,
            config=config,
            sft_config=sft_config,
            callbacks=[early_stopping],
            save_lora_path=lora_path,
            quantize=False,
        )

        print(f"\nSaved LoRA to: {lora_path}")

    # ----------------------------------------------------------------
    # After training, upload to HuggingFace with utility_scripts/upload.py
    # Set username and repo_ids there, then run:
    #   python utility_scripts/upload.py
    # ----------------------------------------------------------------
