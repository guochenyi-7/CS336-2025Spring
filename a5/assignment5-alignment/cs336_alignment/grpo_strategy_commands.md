# GRPO Strategy Commands

Assume the current working directory is `assignment5-alignment/cs336_alignment`.

## `no_baseline`

```bash
python3 grpo_experiment.py \
  --experiment grpo_no_baseline \
  --loss-type no_baseline \
  --policy-device cuda:0 \
  --vllm-device cuda:1 \
  --n-grpo-steps 200 \
  --rollout-batch-size 128 \
  --train-batch-size 128 \
  --gradient-accumulation-steps 32 \
  --group-size 4 \
  --sampling-max-tokens 384 \
  --gpu-memory-utilization 0.8
```

## `reinforce_with_baseline`

```bash
python3 grpo_experiment.py \
  --experiment grpo_reinforce_with_baseline \
  --loss-type reinforce_with_baseline \
  --policy-device cuda:0 \
  --vllm-device cuda:1 \
  --n-grpo-steps 200 \
  --rollout-batch-size 128 \
  --train-batch-size 128 \
  --gradient-accumulation-steps 32 \
  --group-size 4 \
  --sampling-max-tokens 384 \
  --gpu-memory-utilization 0.8
```

## `grpo_clip`

`grpo_clip` is slightly heavier on memory because it caches `old_log_probs`, so this uses a slightly safer `sampling-max-tokens` setting.

```bash
python3 grpo_experiment.py \
  --experiment grpo_clip \
  --loss-type grpo_clip \
  --policy-device cuda:0 \
  --vllm-device cuda:1 \
  --n-grpo-steps 200 \
  --rollout-batch-size 64 \
  --train-batch-size 64 \
  --gradient-accumulation-steps 16 \
  --group-size 4 \
  --sampling-max-tokens 256 \
  --gpu-memory-utilization 0.8
```
