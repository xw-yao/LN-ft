MODEL_NAME="facebook/opt-350m"
GPU_DEVICE=1
task="qqp"
group="opt350_ln_qqp_lr_samp100"
#bitfit_lrs: 9e-2 7e-2 5e-2 3e-2 1e-2 9e-3 7e-3 5e-3 3e-3 1e-3
#full_lrs: 1e-3 9e-4 7e-4 5e-4 3e-4 1e-4 7e-5 5e-5 3e-5 1e-5
#lora_lrs: 9e-3 7e-3 5e-3 3e-3 1e-3 9e-4 7-e4 5e-4
#ln_lrs: 5e-2 3e-2 1e-2 9e-3 7e-3 5e-3 3e-3 1e-3 9e-4 7e-4 5e-4
for lr in 5e-2 3e-2 1e-2 9e-3 7e-3 5e-3 3e-3 1e-3 9e-4 7e-4 5e-4; do
	    sample_size=100
	    epochs=10
	    
	    #full_ft
#	    python run_glue.py \
#		    --output-path ./output \
#                    --gpu-device $GPU_DEVICE \
#                    --epochs $epochs \
#                    --learning-rate $lr \
#		    --weight-decay 0.1 \
#		    --batch-size 32 \
#		    --gradient-accumulation-steps 2 \
#                    --warmup-ratio 0.06 \
#                    --task-name $task \
#		    --sample-size $sample_size \
#		    --model-name $MODEL_NAME \
#		    --fine-tune-type full_ft \
#		    --group $group
#
#
#	    #bitfit
#	    python run_glue.py \
#		    --output-path ./output \
#		    --gpu-device $GPU_DEVICE \
#		    --epochs $epochs \
#		    --learning-rate $lr \
#		    --weight-decay 0.1 \
#		    --batch-size 32 \
#		    --gradient-accumulation-steps 2 \
#                    --warmup-ratio 0.06 \
#		    --task-name $task \
#		    --sample-size $sample_size \
#		    --model-name $MODEL_NAME \
#		    --fine-tune-type bitfit \
#		    --group $group
#
#	    #layernorm
	    python run_glue.py \
		    --output-path ./output \
		    --gpu-device $GPU_DEVICE \
		    --epochs $epochs \
		    --learning-rate $lr \
                    --weight-decay 0.1 \
                    --batch-size 32 \
                    --gradient-accumulation-steps 2 \
		    --warmup-ratio 0.06 \
                    --task-name $task \
                    --sample-size $sample_size \
                    --model-name $MODEL_NAME \
                    --fine-tune-type layernorm \
		    --param-terms self_attn_layer_norm \
                    --group $group
#
#	    #lora
#            python run_glue.py \
#                    --output-path ./output \
#                    --gpu-device $GPU_DEVICE \
#                    --epochs $epochs \
#                    --learning-rate $lr \
#                    --weight-decay 0.1 \
#                    --batch-size 32 \
#                    --gradient-accumulation-steps 2 \
#                    --warmup-ratio 0.06 \
#                    --task-name $task \
#                    --sample-size $sample_size \
#                    --model-name $MODEL_NAME \
#                    --fine-tune-type lora \
#		    --apply-lora \
#                    --param-terms lora \
#                    --group $group
#
#	    #lora_ln
#            python run_glue.py \
#                    --output-path ./output \
#                    --gpu-device $GPU_DEVICE \
#                    --epochs $epochs \
#                    --learning-rate $lr \
#                    --weight-decay 0.1 \
#                    --batch-size 32 \
#                    --gradient-accumulation-steps 2 \
#                    --warmup-ratio 0.06 \
#                    --task-name $task \
#                    --sample-size $sample_size \
#                    --model-name $MODEL_NAME \
#                    --fine-tune-type lora_ln \
#                    --apply-lora \
#                    --param-terms lora lora self_attn_layer_norm\
#                    --group $group
		    
done            
