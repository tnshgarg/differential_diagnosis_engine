import unsloth
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

import os
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemma-4-E4B")
DATASET_PATH = os.environ.get("DATASET_PATH", "symptom_ddx_train.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs_symptom_ddx_plain_v3")

MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "1024"))
BATCH_SIZE = 1
GRAD_ACCUM = 4
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "5e-5"))
EPOCHS = int(os.environ.get("EPOCHS", "1"))

# -----------------------------
# LOAD MODEL
# -----------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma",
)

# -----------------------------
# ADD LORA
# -----------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train",
)

# -----------------------------
# TRAINING CONFIG
# -----------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,

    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_num_proc=1,
        max_length=MAX_SEQ_LENGTH,
        completion_only_loss=True,
        packing=False,

        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,

        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,

        logging_steps=1,
        save_strategy="epoch",

        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),

        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",

        report_to="none",
    ),
)

# -----------------------------
# TRAIN
# -----------------------------
trainer.train()

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training completed successfully!")
