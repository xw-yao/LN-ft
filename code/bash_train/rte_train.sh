MODEL_NAME="facebook/opt-350m"
GPU_DEVICE=3
task="rte"
group="opt350_loraln_rte_lr_samp4000"
project="temp"

#full_ft_lrs: 1e-3 9e-4 7e-4 5e-4 3e-4 1e-4 7e-5 5e-5 3e-5 1e-5
#other_lrs: 5e-2 3e-2 1e-2 9e-3 7e-3 5e-3 3e-3 1e-3 9e-4 7e-4 5e-4

for size in 100 500 1000 2000 4000; do
	    
	    #full_ft
	    python run_glue.py \
		    --output-path ./output \
                    --gpu-device $GPU_DEVICE \
                    --epochs $epochs \
                    --learning-rate 1e-4 \
		    --weight-decay 0.1 \
		    --batch-size 32 \
		    --gradient-accumulation-steps 2 \
                    --warmup-ratio 0.06 \
                    --task-name $task \
		    --sample-size $sample_size \
		    --model-name $MODEL_NAME \
		    --fine-tune-type full_ft \
		    --group $group
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
#	    python run_glue.py \
#		    --output-path ./output \
#		    --gpu-device $GPU_DEVICE \
#		    --epochs $epochs \
#		    --learning-rate $lr \
#                    --weight-decay 0.1 \
#                    --batch-size 32 \
#                    --gradient-accumulation-steps 2 \
#		    --warmup-ratio 0.06 \
#                    --task-name $task \
#                    --sample-size $sample_size \
#                    --model-name $MODEL_NAME \
#                    --fine-tune-type layernorm \
#		    --param-terms self_attn_layer_norm \
#                    --group $group
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
                    --fine-tune-type lora_ln \
                    --apply-lora \
                    --param-terms lora self_attn_layer_norm\
                    --group $group
		    
done            
