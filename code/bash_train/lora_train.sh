MODEL_NAME="facebook/opt-350m"
GPU_DEVICE=6
project="bfloat16"

for task in cola sst2 mrpc stsb qqp mnli qnli rte; do
	    
	    #full_ft
	    python run_glue.py \
		    --output-path ./output \
                    --gpu-device $GPU_DEVICE \
                    --epochs 100 \
                    --learning-rate 3e-3 \
		    --weight-decay 0.1 \
		    --batch-size 32 \
		    --gradient-accumulation-steps 2 \
                    --warmup-ratio 0.06 \
                    --task-name $task \
		    --sample-size 100 \
		    --model-name $MODEL_NAME \
		    --fine-tune-type lora \
		    --apply-lora \
		    --param-terms lora \
		    --group 350m_lora_100_$task \
		    --project $project \
	    	    --dtype bfloat16


           python run_glue.py \
                    --output-path ./output \
                    --gpu-device $GPU_DEVICE \
                    --epochs 50 \
                    --learning-rate 3e-3 \
                    --weight-decay 0.1 \
                    --batch-size 32 \
                    --gradient-accumulation-steps 2 \
                    --warmup-ratio 0.06 \
                    --task-name $task \
                    --sample-size 500 \
                    --model-name $MODEL_NAME \
                    --fine-tune-type lora \
		    --apply-lora \
		    --param-terms lora \
                    --group 350m_lora_500_$task \
	            --project $project \
                    --dtype bfloat16

	   
	   python run_glue.py \
                    --output-path ./output \
                    --gpu-device $GPU_DEVICE \
                    --epochs 25 \
                    --learning-rate 3e-3 \
                    --weight-decay 0.1 \
                    --batch-size 32 \
                    --gradient-accumulation-steps 2 \
                    --warmup-ratio 0.06 \
                    --task-name $task \
                    --sample-size 1000 \
                    --model-name $MODEL_NAME \
                    --fine-tune-type lora \
		    --apply-lora \
		    --param-terms lora \
                    --group 350m_lora_1000_$task \
                    --project $project \
		    --dtype bfloat16


	  python run_glue.py \
                    --output-path ./output \
                    --gpu-device $GPU_DEVICE \
                    --epochs 20 \
                    --learning-rate 3e-3 \
                    --weight-decay 0.1 \
                    --batch-size 32 \
                    --gradient-accumulation-steps 2 \
                    --warmup-ratio 0.06 \
                    --task-name $task \
                    --sample-size 2000 \
                    --model-name $MODEL_NAME \
                    --fine-tune-type lora \
		    --apply-lora \
		    --param-terms lora \
                    --group 350m_lora_2000_$task \
                    --project $project \
                    --dtype bfloat16


          python run_glue.py \
                    --output-path ./output \
                    --gpu-device $GPU_DEVICE \
                    --epochs 20 \
                    --learning-rate 3e-3 \
                    --weight-decay 0.1 \
                    --batch-size 32 \
                    --gradient-accumulation-steps 2 \
                    --warmup-ratio 0.06 \
                    --task-name $task \
                    --sample-size 4000 \
                    --model-name $MODEL_NAME \
                    --fine-tune-type lora \
		    --apply-lora \
		    --param-terms lora \
                    --group 350m_lora_4000_$task \
                    --project $project \
                    --dtype bfloat16
		    
done            
