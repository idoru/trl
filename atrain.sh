# NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=0 to enable nvlink, infiniband
NCCL_DEBUG=INFO \
TORCH_CPP_LOG_LEVEL=INFO \
TORCH_DISTRIBUTED_DEBUG=INFO \
NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=0 \
accelerate launch \
--config_file /home/sam/.cache/huggingface/accelerate/default_config.yaml \
./examples/scripts/sft_trainer.py
