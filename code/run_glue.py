"""This file contains a tool that wraps the GLUEvaluator API, the tool supports all the evaluations that were
performed in BitFit paper (https://arxiv.org/abs/1804.07461), such as: 'full_ft', 'bitfit', 'frozen', 'rand_uniform'
and 'rand_row_col'.
For questions please reach: benzakenelad@gmail.com
Author Elad Ben-Zaken
"""
import argparse
import os
import logging

import torch
import wandb

from utils import setup_logging
from glue import glue_evaluator, set_seed

setup_logging()
LOGGER = logging.getLogger(__file__)

PADDING = "max_length"
MAX_SEQUENCE_LEN = 128


def _parse_args():
    parser = argparse.ArgumentParser(description='BitFit GLUE evaluation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--output-path', '-o', required=True, type=str,
                        help='output directory path for evaluation products.')
    parser.add_argument('--task-name', '-t', required=True, type=str, help='GLUE task name for evaluation.',
                        choices={'cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli'})
    parser.add_argument('--model-name', '-m', type=str, default='bert-base-cased', help='model-name to evaluate with.',
                        choices={'bert-base-cased', 'bert-large-cased', 'roberta-base', 'facebook/opt-350m'})
    parser.add_argument('--fine-tune-type', '-f', required=True, type=str,
                        help='Which fine tuning process to perform, types are the types that were performed in BitFit paper.',
                        choices={'full_ft', 'bitfit', 'outlier', 'layernorm', 'bitfit_ln', 'lora', 'lora_ln'})
    parser.add_argument('--param-terms', metavar='N', type=str, nargs='+', default=['all'],
                        choices={'intermediate', 'key', 'query', 'value', 'output', 'layernorm', 'output_layernorm',
                                 'attention_layernorm', 'self_attn_layer_norm', 'lora', 'all'},
                        help='bias terms to BitFit, should be given in case --fine-tune-type is bitfit '
                             '(choose \'all\' for BitFit all bias terms)')
    parser.add_argument('--gpu-device', '-d', type=int, default=None,
                        help='GPU id for BitFit, if not mentioned will train on CPU.')
    parser.add_argument('--dtype', '-dt', type=str, choices={'float32', 'bfloat16'}, default="float32",
                        help='choose dtype between float32 and bfloat16')
    parser.add_argument('--seed', '-s', type=int, default=0, help='seed value to set.')
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-3, help='learning rate for training.')
    parser.add_argument('--epochs', '-e', type=int, default=15, help='number of training epochs.')
    parser.add_argument('--gradient-accumulation-steps', '-g', type=int, default=1, help='steps of gradient accumulation.')
    parser.add_argument('--warmup-ratio', '-w', type=float, default=0,
                        help='learning rate warm-up ratio.')
    parser.add_argument('--weight-decay', '-wd', type=float, default=0,
                        help='Weight decay.')
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='training and evaluation batch size.')
    parser.add_argument('--apply-lora', action='store_true', default=False,
                        help='if given, will apply LoRA.')
    parser.add_argument('--lora_alpha', '-la', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_r', '-lr', type=int, default=8, help='LoRa r')
    parser.add_argument('--save-evaluator', action='store_true', default=False,
                        help='if given, will save the evaluator for later inference/examination.')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='if given, will plot a list of trainable weights.')
    return parser.parse_args()


def _validate_args(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.isdir(args.output_path):
        raise ValueError("--output_path must be a path to directory")
    if len(os.listdir(args.output_path)):
        raise ValueError("--output_path directory isn't empty, please supply an empty directory path.")
    if args.fine_tune_type == 'rand_uniform' and args.model_name not in RAND_UNIFORM_MASK_SIZE.keys():
        raise ValueError(f'Currently the rand_uniform fine-tune type is not supported for {args.model_name}.')


def _plot_training_details(args):
    [LOGGER.info('############################################################################################') for _
     in range(3)]
    LOGGER.info('')

    LOGGER.info('Training Details: ')
    LOGGER.info('----------------------------------------------')
    LOGGER.info(f'Model Name: {args.model_name}')
    LOGGER.info(f'Task Name: {args.task_name}')
    LOGGER.info(f'Fine Tuning Type: {args.fine_tune_type}')
    LOGGER.info(f'Output Directory: {args.output_path}')

    if args.gpu_device is not None:
        LOGGER.info(f'Running on GPU #{args.gpu_device}')
    else:
        LOGGER.info(f'Running on CPU')

    LOGGER.info(f'Epochs: {args.epochs}')
    LOGGER.info(f'Gradient Accumulation Steps: {args.gradient_accumulation_steps}')
    LOGGER.info(f'Warmup Ratio: {args.warmup_ratio}')
    LOGGER.info(f'Weight Decay: {args.weight_decay}')
    LOGGER.info(f'Learning Rate: {args.learning_rate}')
    LOGGER.info(f'Batch Size: {args.batch_size}')
    LOGGER.info(f"Optimizer: 'AdamW'")

    LOGGER.info('')
    [LOGGER.info('############################################################################################') for _
     in range(3)]


def _perform_training_preparations(evaluator, args, trainable_components):
    if args.fine_tune_type == 'full_ft':
        evaluator.training_preparation(learning_rate=args.learning_rate,
                                       encoder_trainable=True,
                                       weight_decay=args.weight_decay,
                                       verbose=args.verbose)

    if args.fine_tune_type == 'bitfit':
        evaluator.training_preparation(learning_rate=args.learning_rate,
                                       encoder_trainable=False,
                                       ft_type='bitfit',
                                       trainable_components=trainable_components,
                                       weight_decay=args.weight_decay,
                                       verbose=args.verbose)

    if args.fine_tune_type == 'outlier':
        evaluator.training_preparation(learning_rate=args.learning_rate,
                                       encoder_trainable=False,
                                       ft_type='outlier',
                                       trainable_components=trainable_components,
                                       weight_decay=args.weight_decay,
                                       verbose=args.verbose)

    if args.fine_tune_type == 'layernorm':
        evaluator.training_preparation(learning_rate=args.learning_rate,
                                       encoder_trainable=False,
                                       ft_type='layernorm',
                                       trainable_components=trainable_components,
                                       weight_decay=args.weight_decay,
                                       verbose=args.verbose)

    if args.fine_tune_type == 'bitfit_ln':
        evaluator.training_preparation(learning_rate=args.learning_rate,
                                       encoder_trainable=False,
                                       ft_type='bitfit_ln',
                                       trainable_components=trainable_components,
                                       weight_decay=args.weight_decay,
                                       verbose=args.verbose)

    if args.fine_tune_type == 'lora_ln':
        evaluator.training_preparation(learning_rate=args.learning_rate,
                                       encoder_trainable=False,
                                       ft_type='lora_ln',
                                       trainable_components=trainable_components,
                                       apply_lora=args.apply_lora,
                                       lora_alpha=args.lora_alpha,
                                       lora_r=args.lora_r,
                                       weight_decay=args.weight_decay,
                                       verbose=args.verbose)

    if args.fine_tune_type == 'lora':
        evaluator.training_preparation(learning_rate=args.learning_rate,
                                       encoder_trainable=False,
                                       ft_type='lora',
                                       trainable_components=trainable_components,
                                       apply_lora=args.apply_lora,
                                       lora_alpha=args.lora_alpha,
                                       lora_r=args.lora_r,
                                       weight_decay=args.weight_decay,
                                       verbose=args.verbose)
def main():
    # args parsing
    args = _parse_args()
    _validate_args(args)
    _plot_training_details(args)

    wandb.init(
        project="ft-opt",
        group='opt350m-loraln-sst2',
        entity="xwynlp",
        config=vars(args),
    )

    # seed
    set_seed(args.seed)

    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "bfloat16":  # only works on new GPUs, 30XX, A100, A6000 and so on
        dtype = torch.bfloat16
    else:
        raise ValueError(args.dtype)

    with torch.autocast(device_type='cuda', dtype=dtype):
        # evaluator creation
        evaluator = glue_evaluator(args.task_name, args.model_name, args.gpu_device)

        # data preprocessing
        evaluator.preprocess_dataset(PADDING, MAX_SEQUENCE_LEN, args.batch_size)

        # training preparation
        trainable_components = glue_evaluator.convert_to_actual_components(args.param_terms)

        _perform_training_preparations(evaluator, args, trainable_components)

        # train and evaluate
        evaluator.train_and_evaluate(args.epochs, args.gradient_accumulation_steps, args.warmup_ratio, ft_type=args.fine_tune_type)

        # save model
        if args.save_evaluator:
            evaluator.save(os.path.join(args.output_path, 'evaluator'))


if __name__ == '__main__':
    main()
