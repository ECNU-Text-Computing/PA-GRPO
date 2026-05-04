#!/bin/bash

# MCQ Qwen3 8B PA-GRPO Training Script
# Usage: bash scripts/run_mcq_qwen.sh
# Note: Qwen uses *_thinking.parquet datasets, batch_size=40

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ============ 配置区域 - 请根据实际情况修改 ============
# 模型路径 (需要修改为你的实际路径)
MODEL_PATH="/path/to/your/Qwen3-8B"

# 设置可见的GPU
export CUDA_VISIBLE_DEVICES=0
# =====================================================

# 运行训练
nohup python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${PROJECT_ROOT}/dataset/train/mmlu_raw_5perm_thinking.parquet \
    data.val_files=${PROJECT_ROOT}/dataset/train/mmlu_raw_5perm_thinking.parquet \
    data.train_batch_size=40 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=false \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.tokenizer_path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=40 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.83 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0.005 \
    trainer.logger=['tensorboard'] \
    trainer.project_name='mcqqwen3_8b' \
    trainer.experiment_name='qwen3_8b' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=2 \
    reward_model.reward_manager=batch \
    custom_reward_function.path=${PROJECT_ROOT}/my_reward/mcq_qwen.py \
    custom_reward_function.name=compute_score \
    "$@" > mcq_qwen3_8b.log 2>&1 &

echo "Training started in background. Check mcq_qwen3_8b.log for progress."
