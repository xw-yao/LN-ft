MODEL_NAME="facebook/opt-2.7b"
GPU_DEVICE=1
project="max_model"

#python run_glue.py \
#  --output-path ./output \
#  --gpu-device $GPU_DEVICE \
#  --epochs 10 \
#  --learning-rate 5e-3 \
#  --weight-decay 0.1 \
#  --batch-size 2 \
#  --gradient-accumulation-steps 32 \
#  --warmup-ratio 0.06 \
#  --task-name cola \
#  --sample-size 100 \
#  --model-name $MODEL_NAME \
#  --fine-tune-type bitfit \
#  --group bitfit \
#  --project $project
#
#if [ $? -ne 0 ]; then
#  echo "bitfit failed. Exiting script..."
#  exit 1
#fi
#
#python run_glue.py \
#  --output-path ./output \
#  --gpu-device $GPU_DEVICE \
#  --epochs 10 \
#  --learning-rate 3e-3 \
#  --weight-decay 0.1 \
#  --batch-size 2 \
#  --gradient-accumulation-steps 32 \
#  --warmup-ratio 0.06 \
#  --task-name cola \
#  --sample-size 100 \
#  --model-name $MODEL_NAME \
#  --fine-tune-type lora \
#  --apply-lora \
#  --param-terms lora \
#  --group lora \
#  --project $project
#
#if [ $? -ne 0 ]; then
#  echo "lora failed. Exiting script..."
#  exit 1
#fi
#
#python run_glue.py \
#  --output-path ./output \
#  --gpu-device $GPU_DEVICE \
#  --epochs 10 \
#  --learning-rate 5e-3 \
#  --weight-decay 0.1 \
#  --batch-size 2 \
#  --gradient-accumulation-steps 32 \
#  --warmup-ratio 0.06 \
#  --task-name cola \
#  --sample-size 100 \
#  --model-name $MODEL_NAME \
#  --fine-tune-type layernorm \
#  --param-terms self_attn_layer_norm \
#  --group ln \
#  --project $project
#
#if [ $? -ne 0 ]; then
#  echo "ln failed. Exiting script..."
#  exit 1
#fi

python run_glue.py \
  --output-path ./output \
  --gpu-device $GPU_DEVICE \
  --epochs 10 \
  --learning-rate 3e-3 \
  --weight-decay 0.1 \
  --batch-size 2 \
  --gradient-accumulation-steps 32 \
  --warmup-ratio 0.06 \
  --task-name cola \
  --sample-size 100 \
  --model-name $MODEL_NAME \
  --fine-tune-type lora_ln \
  --apply-lora \
  --param-terms lora self_attn_layer_norm \
  --group lora_ln \
  --project $project

#if [ $? -ne 0 ]; then
#  echo "lora_ln failed. Exiting script..."
#  exit 1
#fi

