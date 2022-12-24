# LN-ft
Code is modified on bit fit code base (https://github.com/benzakenelad/BitFit.git)

How to run:

create environment with all the dependencies by:
$ conda env create -n bitfit_env -f environment.yml

Run example:
python run_glue.py 
       --output-path <output_path>\
       --task-name rte\  
       --model-name bert-base-cased\
       --fine-tune-type outlier\
       --bias-terms layernorm\
       --gpu-device <gpu_device>\
       --learning-rate 1e-5
       --save-evaluator

To run outlier fine-tuning and layernorm fine-tuning, set arg --bias-terms to 'layer norm'.
