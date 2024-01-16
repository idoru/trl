#!/usr/bin/env bash

LR="2e-4"

MAX_SEQ_LEN=512

MODEL_NAME="mistralai/Mistral-7B-v.01"
MODEL_NAME="OpenPipe/mistral-ft-optimized-1218"

DATASET_NAME="timdettmers/openassistant-guanaco"
DATASET_TEXT_FIELD="text"

BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
GRADIENT_CHECKPOINTING=false

TRAIN_EPOCHS=25

OUTPUT_DIR="output"

python examples/scripts/sft.py --model_name "$MODEL_NAME" \
  --dataset_name "$DATASET_NAME" \
  --dataset_text_field "$DATASET_TEXT_FIELD" \
  --report_to wandb \
  --learning_rate "$LR" \
  --seq_length "$MAX_SEQ_LEN" \
  --use_peft \
  --output_dir "$OUTPUT_DIR" \
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
  --batch_size "$BATCH_SIZE" \
  --peft_lora_r 64 \
  --num_train_epochs "$TRAIN_EPOCHS" \
  #--load_in_4bit