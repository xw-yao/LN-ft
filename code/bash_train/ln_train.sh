MODEL_NAME="facebook/opt-350m"
GPU_DEVICE=7
project="bfloat16"

for task in cola sst2 mrpc stsb qqp mnli qnli rte; do
	    
	    python run_glue.py \
		    --output-path ./output \
                    --gpu-device $GPU_DEVICE \
                    --epochs 100 \
                    --learning-rate 5e-3 \
		    --weight-decay 0.1 \
		    --batch-size 32 \
		    --gradient-accumulation-steps 2 \
                    --warmup-ratio 0.06 \
                    --task-name $task \
		    --sample-size 100 \
		    --model-name $MODEL_NAME \
		    --fine-tune-type layernorm \
		    --param-terms self_attn_layer_norm \
		    --group 350m_ln_100_$task \
		    --project $project \
	    	    --dtype bfloat16


           python run_glue.py \
                    --output-path ./output \
                    --gpu-device $GPU_DEVICE \
                    --epochs 50 \
                    --learning-rate 5e-3 \
                    --weight-decay 0.1 \
                    --batch-size 32 \
                    --gradient-accumulation-steps 2 \
                    --warmup-ratio 0.06 \
                    --task-name $task \
                    --sample-size 500 \
                    --model-name $MODEL_NAME \
                    --fine-tune-type layernorm \
		    --param-terms self_attn_layer_norm \
                    --group 350m_ln_500_$task \
                    --project $project \
		    --dtype bfloat16

	   
	   python run_glue.py \
                    --output-path ./output \
                    --gpu-device $GPU_DEVICE \
                    --epochs 25 \
                    --learning-rate 5e-3 \
                    --weight-decay 0.1 \
                    --batch-size 32 \
                    --gradient-accumulation-steps 2 \
                    --warmup-ratio 0.06 \
                    --task-name $task \
                    --sample-size 1000 \
                    --model-name $MODEL_NAME \
                    --fine-tune-type layernorm \
		    --param-terms self_attn_layer_norm \
                    --group 350m_ln_1000_$task \
                    --project $project \
                    --dtype bfloat16


	  python run_glue.py \
                    --output-path ./output \
                    --gpu-device $GPU_DEVICE \
                    --epochs 20 \
                    --learning-rate 5e-3 \
                    --weight-decay 0.1 \
                    --batch-size 32 \
                    --gradient-accumulation-steps 2 \
                    --warmup-ratio 0.06 \
                    --task-name $task \
                    --sample-size 2000 \
                    --model-name $MODEL_NAME \
                    --fine-tune-type layernorm \
		    --param-terms self_attn_layer_norm \
                    --group 350m_ln_2000_$task \
                    --project $project \
                    --dtype bfloat16


          python run_glue.py \
                    --output-path ./output \
                    --gpu-device $GPU_DEVICE \
                    --epochs 20 \
                    --learning-rate 5e-3 \
                    --weight-decay 0.1 \
                    --batch-size 32 \
                    --gradient-accumulation-steps 2 \
                    --warmup-ratio 0.06 \
                    --task-name $task \
                    --sample-size 4000 \
                    --model-name $MODEL_NAME \
                    --fine-tune-type layernorm \
		    --param-terms self_attn_layer_norm \
                    --group 350m_ln_4000_$task \
                    --project $project \
                    --dtype bfloat16
		    
done            
