accelerate launch \
  --config_file ./examples/accelerate_configs/deepspeed_zero3.yaml \
  ./examples/scripts/dpo.py \
  --model_name_or_path="microsoft/phi-2" \
  --trust_remote_code