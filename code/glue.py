import math
import random
import pickle
import logging

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from datasets import load_dataset

import wandb
from tqdm import tqdm

from utils import setup_logging
from lora_opt.modeling_opt import OPTForCausalLM
from lora_opt.configuration_opt import OPTConfig  # local version that is modified for lora
from lora_bert.modeling_bert import BertForSequenceClassification
from lora_bert.configuration_bert import BertConfig

setup_logging()
LOGGER = logging.getLogger(__file__)
MAX_GRAD_NORM = 1.0

OPT_30B_DEVICE_MAP = {
    'model.decoder.embed_tokens': 0,
    'model.decoder.embed_positions': 0,
    'model.decoder.layers.0': 0,
    'model.decoder.layers.1': 0,
    'model.decoder.layers.2': 0,
    'model.decoder.layers.3': 0,
    'model.decoder.layers.4': 0,
    'model.decoder.layers.5': 0,
    'model.decoder.layers.6': 0,
    'model.decoder.layers.7': 0,
    'model.decoder.layers.8': 0,
    'model.decoder.layers.9': 0,
    'model.decoder.layers.10': 0,
    'model.decoder.layers.11': 0,
    'model.decoder.layers.12': 0,
    'model.decoder.layers.13': 0,
    'model.decoder.layers.14': 0,
    'model.decoder.layers.15': 0,
    'model.decoder.layers.16': 0,
    'model.decoder.layers.17': 0,
    'model.decoder.layers.18': 0,
    'model.decoder.layers.19': 0,
    'model.decoder.layers.20': 0,
    'model.decoder.layers.21': 0,
    'model.decoder.layers.22': 0,
    'model.decoder.layers.23': 0,
    'model.decoder.layers.24': 1,
    'model.decoder.layers.25': 1,
    'model.decoder.layers.26': 1,
    'model.decoder.layers.27': 1,
    'model.decoder.layers.28': 1,
    'model.decoder.layers.29': 1,
    'model.decoder.layers.30': 1,
    'model.decoder.layers.31': 1,
    'model.decoder.layers.32': 1,
    'model.decoder.layers.33': 1,
    'model.decoder.layers.34': 1,
    'model.decoder.layers.35': 1,
    'model.decoder.layers.36': 1,
    'model.decoder.layers.37': 1,
    'model.decoder.layers.38': 1,
    'model.decoder.layers.39': 1,
    'model.decoder.layers.40': 1,
    'model.decoder.layers.41': 1,
    'model.decoder.layers.42': 1,
    'model.decoder.layers.43': 1,
    'model.decoder.layers.44': 1,
    'model.decoder.layers.45': 1,
    'model.decoder.layers.46': 1,
    'model.decoder.layers.47': 1,
    'model.decoder.final_layer_norm': 1,
    'lm_head': 1,
}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


TASK_TO_METRICS = {
    "cola": ["MCC"],
    "mnli": ["Accuracy"],
    "mrpc": ["Accuracy", "F1"],
    "qnli": ["Accuracy"],
    "qqp": ["Accuracy", "F1"],
    "rte": ["Accuracy"],
    "sst2": ["Accuracy"],
    "stsb": ["Spearman", "Pearson"],
    "wnli": ["Accuracy"],
}

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

TASK_TO_PROMPT_TRAIN = {
    "cola": """The following sentence is either "acceptable", meaning it is grammatically correct and makes sense, or "unacceptable". Which is it?\n{sentence1}\n{label}\n""",
    "sst2": """{sentence1}\nWas that sentence "positive" or "negative"? It was\n{label}\n""",
    "mrpc": """Does the sentence\n{sentence1}\nparaphrase (that is, mean the same thing as) this sentence? (yes or no)\n{sentence2}\n{label}\n""",
    # "qqp": """Are the questions "{sentence1}" and "{sentence2}" asking the same thing? (yes or no)\n{label}\n""",
    "qqp": """Does the question\n{sentence1}\nparaphrase (that is, asking the same thing as) this question? (yes or no)\n{sentence2}\n{label}\n""",
    "stsb": """Rate on a scale from 0.0 to 5.0 how similar the sentences "{sentence1}" and "{sentence2}" are.\n{label}\n""",
    # note that you might need to round the labels to 1 digit after the comma
    "mnli": """{sentence1}\nBased on the previous passage, is it true that "{sentence2}"? Yes, no, or maybe?\n{label}\n""",
    "qnli": """Can you answer the question "{sentence1}" based only on the following: {sentence2}? (yes or no)\n{label}\n""",
    "rte": """Does "{sentence1}" imply that "{sentence2}"? Please answer either "yes" or "no".\n{label}\n""",
    "wnli": """{{sentence1}}\n{{sentence2}}\nDoes the first sentence imply the second sentence? (yes or no)\n{label}\n"""
}

TASK_TO_PROMPT_EVAL = {
    "cola": """The following sentence is either "acceptable", meaning it is grammatically correct and makes sense, or "unacceptable". Which is it?\n{sentence1}\n""",
    "sst2": """{sentence1}\nWas that sentence "positive" or "negative"? It was\n""",
    "mrpc": """Does the sentence\n{sentence1}\nparaphrase (that is, mean the same thing as) this sentence? (yes or no)\n{sentence2}\n""",
    "qqp": """Does the question\n{sentence1}\nparaphrase (that is, asking the same thing as) this question? (yes or no)\n{sentence2}\n""",
    # "qqp": """Are the questions "{sentence1}" and "{sentence2}" asking the same thing? (yes or no)\n""",
    "stsb": """Rate on a scale from 0.0 to 5.0 how similar the sentences "{sentence1}" and "{sentence2}" are.\n""",
    # note that you might need to round the labels to 1 digit after the comma
    "mnli": """{sentence1}\nBased on the previous passage, is it true that "{sentence2}"? Yes, no, or maybe?\n""",
    "qnli": """Can you answer the question "{sentence1}" based only on the following: {sentence2}? (yes or no)\n""",
    "rte": """Does "{sentence1}" imply that "{sentence2}"? Please answer either "yes" or "no".\n""",
    "wnli": """{{sentence1}}\n{{sentence2}}\nDoes the first sentence imply the second sentence? (yes or no)\n"""
}

NUM_TO_TEXT = {
    "cola": ["unacceptable", "acceptable"],
    "sst2": ["negative", "positive"],
    "mrpc": ["no", "yes"],
    "qqp": ["no", "yes"],
    "mnli": ["yes", "maybe", "no"],
    "qnli": ["yes", "no"],
    "rte": ["yes", "no"],
    "wnli": ["no", "yes"],
    "stsb": lambda x: str(round(x, 1)),  # not tested yet
}

BIAS_TERMS_DICT = {
    'intermediate': 'intermediate.dense.bias',
    'key': 'attention.self.key.bias',
    'query': 'attention.self.query.bias',
    'value': 'attention.self.value.bias',
    'output': 'output.dense.bias',
    'layernorm': 'LayerNorm',
    'self_attn_layer_norm': 'self_attn_layer_norm',
    'output_layernorm': 'output.LayerNorm.bias',
    'attention_layernorm': 'attention.output.LayerNorm.bias',
    'lora': 'lora',
    'all': 'bias',
}

METRIC_NAME_TO_FUNCTION = {
    "MCC": matthews_corrcoef,
    "Accuracy": accuracy_score,
    "F1": f1_score,
    "Spearman": spearmanr,
    "Pearson": pearsonr,
}


class glue_evaluator:

    def __init__(self, task_name, model_name, device, dtype=None):
        """
        Args:
            task_name (str): task name, e.g. 'rte'.
            model_name (str): model name, e.g. 'bert-base-uncased'.
            device (int): GPU device to run on, if None will run on CPU.
        """
        self.task_name = task_name
        self.model_name = model_name
        self.device = device

        self.dtype = dtype or torch.float32
        if isinstance(self.dtype, str):
            self.dtype = getattr(torch, self.dtype)

        # initialization
        self.is_regression = task_name == 'stsb'
        self.num_labels = None
        self.data_loaders = None
        self.batch_size = None
        self.opt_train_btach = None
        self.model = None
        self.tokenizer = None
        self.model2 = None
        self.scheduler = None
        self.optimizer = None
        self.learning_rate = None
        self.evaluations = None
        self.encoder_trainable = None
        self.masks = None
        self.idx_to_label = None
        self.epochs = None
        self.init_out_ln_params = []

    def preprocess_dataset(self, padding, max_sequence_len, batch_size, sample_size=None, random_seed=42):
        """Preprocess the train and validation datasets.
        Args:
            padding (str): padding method (currently 'max_length' is the suggested method)
            max_sequence_len (int): the maximum sequence length
            batch_size (int): training and evaluating batch size
            train_size (int): clip the train dataset size, if None will use all available samples
        """
        LOGGER.info(f'Downloading dataset: {self.task_name}')
        datasets = load_dataset('glue', self.task_name)

        if sample_size is not None:
            # make random indices and take a sample from the train dataset
            subset_indices = np.random.RandomState(random_seed).permutation(len(datasets['train']))[:sample_size]
            datasets['train'] = datasets['train'].select(subset_indices)

        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if 'opt' in self.model_name:
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.num_labels = 1
        is_regression = self.task_name == 'stsb'
        if not is_regression:
            label_list = datasets['train'].features['label'].names
            self.idx_to_label = {k: v for k, v in enumerate(datasets['train'].features['label'].__dict__['_int2str'])}
            self.num_labels = len(label_list)

        sentence1_key, sentence2_key = TASK_TO_KEYS[self.task_name]

        def _generate_train_prompts(examples):
            prompted_example = (TASK_TO_PROMPT_TRAIN[self.task_name].format(sentence1=examples[sentence1_key],
                                                                            label=NUM_TO_TEXT[self.task_name][examples[
                                                                                'label']])) if sentence2_key is None else \
                (TASK_TO_PROMPT_TRAIN[self.task_name].format(sentence1=examples[sentence1_key],
                                                             sentence2=examples[sentence2_key],
                                                             label=NUM_TO_TEXT[self.task_name][examples['label']])
                 )
            result = self.tokenizer(prompted_example, padding=padding, max_length=max_sequence_len, truncation=True)
            return result

        def _generate_train_prompts_stsb(examples):
            prompted_example = (TASK_TO_PROMPT_TRAIN[self.task_name].format(sentence1=examples[sentence1_key],
                                                                            label=NUM_TO_TEXT[self.task_name][examples[
                                                                                'label']])) if sentence2_key is None else \
                (TASK_TO_PROMPT_TRAIN[self.task_name].format(sentence1=examples[sentence1_key],
                                                             sentence2=examples[sentence2_key],
                                                             label=NUM_TO_TEXT[self.task_name](examples['label']))
                 )
            result = self.tokenizer(prompted_example, padding=padding, max_length=max_sequence_len, truncation=True)
            return result

        def _generate_eval_prompts(examples):
            prompted_example = (
                TASK_TO_PROMPT_EVAL[self.task_name].format(
                    sentence1=examples[sentence1_key])) if sentence2_key is None else (
                TASK_TO_PROMPT_EVAL[self.task_name].format(sentence1=examples[sentence1_key],
                                                           sentence2=examples[sentence2_key])
            )
            result = self.tokenizer(prompted_example, padding=padding, max_length=max_sequence_len, truncation=True)
            return result

        def _preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (
                    examples[sentence1_key], examples[sentence2_key])
            )
            result = self.tokenizer(*args, padding=padding, max_length=max_sequence_len, truncation=True)
            return result

        if 'opt' in self.model_name:
            if self.task_name == 'stsb':
                datasets['train'] = datasets['train'].map(_generate_train_prompts_stsb, load_from_cache_file=False)
                datasets['validation'] = datasets['validation'].map(_generate_eval_prompts, load_from_cache_file=False)
                datasets['test'] = datasets['test'].map(_generate_eval_prompts, load_from_cache_file=False)

            elif self.task_name == 'mnli':
                datasets['train'] = datasets['train'].map(_generate_train_prompts, load_from_cache_file=False)
                datasets['validation_matched'] = datasets['validation_matched'].map(_generate_eval_prompts,
                                                                                    load_from_cache_file=False)
                datasets['validation_mismatched'] = datasets['validation_mismatched'].map(_generate_eval_prompts,
                                                                                          load_from_cache_file=False)
                datasets['test_matched'] = datasets['test_matched'].map(_generate_eval_prompts,
                                                                        load_from_cache_file=False)
                datasets['test_mismatched'] = datasets['test_mismatched'].map(_generate_eval_prompts,
                                                                              load_from_cache_file=False)
            else:
                # datasets['train'] = load_dataset('glue', self.task_name, split='train[:16]')
                datasets['train'] = datasets['train'].map(_generate_train_prompts, load_from_cache_file=False)
                datasets['validation'] = datasets['validation'].map(_generate_eval_prompts, load_from_cache_file=False)
                datasets['test'] = datasets['test'].map(_generate_eval_prompts, load_from_cache_file=False)

        else:
            datasets = datasets.map(_preprocess_function, batched=True, load_from_cache_file=False)

        self.data_loaders = dict()

        self.data_loaders['train'] = datasets['train']

        if self.task_name == 'mnli':
            self.data_loaders['validation_matched'] = datasets['validation_matched']
            self.data_loaders['validation_mismatched'] = datasets['validation_mismatched']
            self.data_loaders['test_matched'] = datasets['test_matched']
            self.data_loaders['test_mismatched'] = datasets['test_mismatched']
        else:
            self.data_loaders['validation'] = datasets['validation']
            self.data_loaders['test'] = datasets['test']

        for dataset_name, dataset in self.data_loaders.items():
            self.data_loaders[dataset_name] = self._convert_dataset_to_data_loader(dataset=dataset,
                                                                                   model_name=self.model_name,
                                                                                   batch_size=self.batch_size,
                                                                                   random_sampler=dataset_name == 'train',
                                                                                   test='test' in dataset_name)
        # print(self.data_loaders.items())
        # exit()

    def training_preparation(self, learning_rate, encoder_trainable, weight_decay, lora_alpha=None, lora_r=None,
                             ft_type=None, apply_lora=False, trainable_components=None, verbose=True):
        """Performs training preparation.
        Perform training preparation including: model initialization, optimizer initialization, relevant
        gradients deactivation and plotting a list of all trainable params (if verbose is True).
        Args:
            learning_rate (float): learning_rate to train with.
            optimizer(str): optimizer to perform the training with, currently adam and adamw are supported.
            encoder_trainable (bool): if True will perform a Full-FT else will perform BitFit training preparation
            trainable_components(Union[List[str], None]): list of trainable component.(subset of `BIAS_TERMS_DICT` keys)
        """
        if self.model:
            raise Exception('Training preparation was already completed.')

        if encoder_trainable and trainable_components:
            raise Exception(
                f"If encoder_trainable is True, you shouldn't supply trainable_components. "
                f"Got trainable_components: {trainable_components}")

        self.encoder_trainable = encoder_trainable
        # model declaration
        if 'opt' in self.model_name:
            assert self.model_name != 'opt'
            config = OPTConfig.from_pretrained(
                self.model_name,
                apply_lora=apply_lora,
                lora_alpha=lora_alpha if apply_lora else None,
                lora_r=lora_r if apply_lora else None,
            )

            device_map = None
            if self.model_name == 'facebook/opt-30b':
                # doesn't work when we add parameters to the model (e.g. lora)
                # needs modification in the .from_pretrained() method
                device_map = 'auto'

            self.model = OPTForCausalLM.from_pretrained(
                self.model_name,
                config=config,
                device_map=device_map,
                torch_dtype=self.dtype,
            )
        else:
            if apply_lora:
                config = BertConfig.from_pretrained(self.model_name, num_labels=self.num_labels, apply_lora=True,
                                                    lora_alpha=lora_alpha, lora_r=lora_r, return_dict=True)
            else:
                config = BertConfig.from_pretrained(self.model_name, num_labels=self.num_labels, return_dict=True)
            self.model = BertForSequenceClassification.from_pretrained(self.model_name, config=config)

        if ft_type == 'outlier':
            self.model2 = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config)
            for name, param in self.model2.named_parameters():
                if 'LayerNorm' in name and not any(x in name for x in ['attention', 'embeddings']):
                    param.requires_grad = False
                    self.init_out_ln_params.append(param)

        if not encoder_trainable:
            self._deactivate_relevant_gradients(ft_type, trainable_components)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            correct_bias=True,
            eps=1e-8,
            weight_decay=weight_decay,
        )

        self.learning_rate = learning_rate

        if verbose:
            total_parameters = sum(p.numel() for p in self.model.parameters())
            total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f'\n-----------------------------------------------------------------')
            print(f'Total Parameters              : {total_parameters / 1e6:>10.2f} M')
            print(f'Number of Trainable Parameters: {total_trainable_params / 1e6:>10.2f} M\n')
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name, '  --->  ', param.shape)
            wandb.log({"total_trainable_params": total_trainable_params})
        self.evaluations = {k: {metric_name: [] for metric_name in TASK_TO_METRICS[self.task_name]} for k in
                            self.data_loaders.keys()}

    def train_and_evaluate(self, num_epochs, gradient_accumulation_steps, warmup_ratio, ft_type=None):
        """Trains the encoder model and evaluate it on validation set.
        Learning curves will be saved to the output_path.
        Args:
            num_epochs (int): Number of epochs to perform.
            output_path (str): Directory path to save the learning curves too.
        """

        # validations
        if not self.data_loaders:
            raise Exception('data loaders were not initialized, please run "preprocess_dataset" before training.')

        if not self.model:
            raise Exception('model was not initialized, please run "training_preparation" before training.')

        # moving model to the required device
        if self.model_name != "facebook/opt-30b" and self.device is not None:
            # when working with 30B model we move it to the device in the model initialization via device_map
            self.model.to(device=torch.device(self.device), dtype=self.dtype)

        # train and evaluate
        self.epochs = num_epochs

        _n = len(self.data_loaders['train'].dataset)
        t_total = math.ceil(_n / self.batch_size) * num_epochs
        #print(f'total update steps: {t_total}')
        warmup_steps = math.ceil(t_total * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=t_total,
        )

        if gradient_accumulation_steps > 0:
            print('----------------------------------------')
            print(f'Using gradient accumulation with {gradient_accumulation_steps} steps')
            print(f'Effective batch size is {self.batch_size * gradient_accumulation_steps}')

        criteria = torch.nn.MSELoss() if self.is_regression else torch.nn.CrossEntropyLoss()

        train_dataloader = self.data_loaders['train']
        global_step = -1  # -1 because we increment it before the first step
        update_step = 0  # 0 because it is logged before updating

        # if self.task_name == 'stsb':
        #     raise NotImplementedError(
        #         "best metric tracking is not implemented for STS-B. If it is MSE, you need to take min instead of max.")

        best_results = {dataloader_type: None for dataloader_type in self.data_loaders.keys() if
                        'validation' in dataloader_type}  # we maximize accuracy, in case of CoLA we maximize MCC

        for epoch in range(num_epochs):
            #global_step += 1
            # move to train mode
            self.model.train()

            evaluated_samples = 0
            loss_sum = 0

            progress_bar = tqdm(train_dataloader, desc=f"EPOCH {epoch}")
            for step, batch in enumerate(progress_bar):
                global_step += 1

                # move batch data to gpu
                if self.device is not None:
                    # if you're using model parallel, use --gpu-device 0 and specify GPUs in the environment variable CUDA_VISIBLE_DEVICES like this:
                    # export CUDA_VISIBLE_DEVICES=2,5
                    batch = tuple(obj.cuda(self.device) for obj in batch)

                if 'roberta' in self.model_name or 'opt' in self.model_name:
                    # labels: LongTensor[batch_size,] is a tensor of 0, 1 indices — class ids from GLUE
                    # note that language loss is calculated using input_ids, we use labels for evaluation only
                    input_ids, attention_mask, labels = batch
                    token_type_ids = None
                else:
                    input_ids, attention_mask, token_type_ids, labels = batch

                # forward pass
                # loss calculation
                if 'opt' in self.model_name:
                    targets = input_ids.clone()
                    targets[targets == self.tokenizer.pad_token_id] = -100
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=targets)
                    loss = outputs.loss / gradient_accumulation_steps  # divide by gradient_accumulation_steps to scale loss

                    _predictions = torch.argmax(outputs.logits, dim=-1)
                    _predictions = _predictions[:, :-1]  # double check that _predictions do not end on a token that is a part of the label
                    _targets = targets[:, 1:]
                    accuracy = (_predictions == _targets) & (_targets != -100)
                    accuracy = accuracy.sum()
                    accuracy = accuracy / torch.sum(_targets != -100)

                    wandb.log({
                        "loss": loss * gradient_accumulation_steps,  # report in-batch loss
                        "train_lm_accuracy": accuracy,
                        "epoch": epoch,
                        "update_step": update_step,
                    }, step=global_step)

                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                         token_type_ids=token_type_ids)
                    outputs = outputs.logits
                    labels = labels.view(-1)
                    outputs = outputs.view(-1) if self.is_regression else outputs.view(-1, self.num_labels)
                    loss = criteria(outputs, labels) / gradient_accumulation_steps
                    evaluated_samples += len(labels)

                    # calculate the accuracy in the classification case
                    if not self.is_regression:
                        outputs = outputs.detach().cpu().numpy()
                        labels = labels.cpu().numpy()
                        outputs = np.argmax(outputs, axis=1)
                        # accuracy calculation
                        accuracy = accuracy_score(labels, outputs)

                        wandb.log({
                            "loss": loss,
                            "train_lm_accuracy": accuracy,
                            "epoch": epoch,
                            "update_step": update_step,
                        }, step=global_step)

                    else:
                        wandb.log({
                            "loss": loss,
                            "epoch": epoch,
                            "update_step": update_step,
                        }, step=global_step)

                loss.backward()

                # masking the relevant gradients (if needed)
                if self.masks:
                    if 'roberta' in self.model_name:
                        for name, param in self.model.roberta.named_parameters():
                            param.grad[~self.masks[name]] = 0
                            param.grad[~self.masks[name]] = 0
                    else:
                        for name, param in self.model.bert.named_parameters():
                            param.grad[~self.masks[name]] = 0

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=MAX_GRAD_NORM)

                # update parameters
                if (global_step + 1) % gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    update_step += 1

                    # if ft_type == 'outlier':
                    #     for param in self.model.parameters():
                    #         param.requires_grad = False
                    #     idx = 0
                    #     for name, param in self.model.named_parameters():
                    #         if 'LayerNorm' in name and not any(x in name for x in ['attention', 'embeddings']):
                    #             for dim in range(768):
                    #                 if dim != 308 and dim != 381:
                    #                     param[dim] = self.init_out_ln_params[idx][dim]
                    #             idx += 1
                    #             param.requires_grad = True
                    #     self.optimizer.step()

                    if self.scheduler is not None:
                        self.scheduler.step()

                    self.model.zero_grad()

                # track train loss
                loss_sum += loss.item()

                # printing training progress
                _loss_to_log = round(loss_sum / (step + 1), 3)
                _accuracy_to_log = round(accuracy.item(), 3) if not self.is_regression else None
                progress_bar.set_postfix({'loss': _loss_to_log, 'accuracy': _accuracy_to_log})
            print()

            # evaluation
            for dataloader_type, dataloader in self.data_loaders.items():
                if 'validation' in dataloader_type:
                    results = self._evaluate(dataloader, dataloader_type)
                    wandb.log({f"{dataloader_type}/{k}": v for k, v in results.items()}, step=global_step)

                    if self.task_name == 'cola':
                        _metric_to_maximize = 'MCC'
                    elif self.task_name == 'stsb':
                        _metric_to_maximize = 'Pearson'
                    else:
                        _metric_to_maximize = 'Accuracy'

                    if best_results[dataloader_type] is None:
                        best_results[dataloader_type] = results

                    if results[_metric_to_maximize] >= best_results[dataloader_type][_metric_to_maximize]:
                        best_results[dataloader_type] = results
                        _log_dict = {f"best_{dataloader_type}/{k}": v for k, v in best_results[dataloader_type].items()}
                        wandb.log(_log_dict, step=global_step)

                    for metric_name, result in results.items():
                        self.evaluations[dataloader_type][metric_name].append(result)

            print()
        # end of training loop

    def save(self, output_path):
        """Saves the evaluator to the output_path directory.
        Args:
            output_path (str): Directory to save to model to.
        """
        LOGGER.info(f'Saving the model to: {output_path}')

        self.model.cpu()
        data = {'model_name': self.model_name, 'task_name': self.task_name, 'num_epochs': self.epochs,
                'learning_rate': self.learning_rate, 'evaluations': self.evaluations,
                'batch_size': self.batch_size, 'num_labels': self.num_labels,
                'encoder_trainable': self.encoder_trainable}
        with open(output_path, 'wb') as file:
            pickle.dump(data, file)

    def _update_outlier_params(self, params):
        for param1, param2 in zip(self.init_out_ln_params, params):
            for dim in range(768):
                if dim != 308 and dim != 381:
                    param2[dim] = param1[dim]

    def _deactivate_relevant_gradients(self, ft_type, trainable_components):

        if ft_type == 'bitfit':
            for param in self.model.parameters():
                param.requires_grad = False
            if trainable_components:
                trainable_components = trainable_components + ['pooler.dense.bias']
            trainable_components = trainable_components + ['classifier']
            for name, param in self.model.named_parameters():
                for component in trainable_components:
                    if component in name:
                        param.requires_grad = True
                        break

        if ft_type == 'bitfit_ln':
            for param in self.model.parameters():
                param.requires_grad = False
            trainable_components = trainable_components + ['classifier', 'pooler.dense.bias', 'LayerNorm']
            for name, param in self.model.named_parameters():
                for component in trainable_components:
                    if component in name:
                        param.requires_grad = True
                        break

        if ft_type == 'bitfit_ln' and 'opt' in self.model_name:
            for param in self.model.parameters():
                param.requires_grad = False
            trainable_components = trainable_components + ['final_layer_norm']
            for name, param in self.model.named_parameters():
                for component in trainable_components:
                    if component in name:
                        param.requires_grad = True
                        break

        if ft_type == 'outlier':
            for param in self.model.parameters():
                param.requires_grad = False
            for name, param in self.model.named_parameters():
                if trainable_components[0] in name and not any(x in name for x in ['attention', 'embeddings']):
                    param.requires_grad = True

        if ft_type == 'layernorm':
            for name, param in self.model.named_parameters():
                if not trainable_components[0] in name:
                    param.requires_grad = False

        if ft_type == 'layernorm' and 'opt' in self.model_name:
            for param in self.model.parameters():
                param.requires_grad = False
            trainable_components = trainable_components + ['final_layer_norm']
            for name, param in self.model.named_parameters():
                for component in trainable_components:
                    if component in name:
                        param.requires_grad = True
                        break

        if ft_type == 'lora':
            for name, param in self.model.named_parameters():
                if not trainable_components[0] in name:
                    param.requires_grad = False

        if ft_type == 'lora_ln':
            for param in self.model.parameters():
                param.requires_grad = False
            if trainable_components:
                trainable_components = trainable_components + ['LayerNorm']
            for name, param in self.model.named_parameters():
                for component in trainable_components:
                    if component in name:
                        param.requires_grad = True
                        break

        if ft_type == 'lora_ln' and 'opt' in self.model_name:
            for param in self.model.parameters():
                param.requires_grad = False
            if trainable_components:
                trainable_components = trainable_components + ['final_layer_norm']
            for name, param in self.model.named_parameters():
                for component in trainable_components:
                    if component in name:
                        param.requires_grad = True
                        break

    @staticmethod
    def convert_to_actual_components(components):
        return [BIAS_TERMS_DICT[component] for component in components]

    def _evaluate(self, eval_dataloader, dataloader_type):
        """Evaluates the model on the dataloader
        Args:
            dataloader (torch.utils.data.DataLoader): the data loader we evaluate the model on
            dataloader_type (str): the dataloader type (train/validation)
        Returns:
            (Dict[str, float]): dictionary that maps between metric_name and the metric result
        """
        # move to eval mode
        self.model.eval()

        evaluated_samples = 0
        accuracy_sum = 0
        all_predictions, all_labels = [], []

        for batch in eval_dataloader:
            prompt_preds, true_labels = [], []
            # move batch data to gpu
            if self.device is not None:
                batch = tuple(obj.cuda(self.device) for obj in batch)

            if 'roberta' in self.model_name or 'opt' in self.model_name:
                input_ids, attention_mask, labels = batch
                token_type_ids = None
            else:
                input_ids, attention_mask, token_type_ids, labels = batch

            # forward pass
            with torch.no_grad():
                if 'opt' in self.model_name:
                    outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=1)

                    for idx in range(len(input_ids)):
                        n_input_tokens = len(input_ids[idx])
                        response_ids = outputs[idx][n_input_tokens:]
                        pred = self.tokenizer.decode(response_ids,
                                                     skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)
                        # if self.is_regression:
                        #     pred = float(pred)
                        if self.is_regression:
                            try:
                                pred = float(pred)
                            except:
                                pred = pred
                            prompt_preds.append(pred)
                        else:
                            true_label = NUM_TO_TEXT[self.task_name][labels[idx]]
                            prompt_preds.append(pred)
                            true_labels.append(true_label)

                    labels = labels.view(-1)
                    evaluated_samples += len(labels)

                    # calculate the accuracy in the classification case
                    if not self.is_regression:
                        # accuracy calculation
                        accuracy_sum += accuracy_score(true_labels, prompt_preds) * len(labels)
                        print(f'{dataloader_type} ACC: {round(accuracy_sum / evaluated_samples, 5)}\r', end='')

                    # aggregate predictions and labels
                    all_predictions.extend(prompt_preds)
                    if not self.is_regression:
                        all_labels.extend(true_labels)
                    else:
                        labels = labels.cpu().numpy()
                        all_labels.extend(list(labels))
                    # print(all_predictions)
                    # print(all_labels)

                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                         token_type_ids=token_type_ids)
                    outputs = outputs.logits
                    # reshaping
                    labels = labels.view(-1)
                    outputs = outputs.view(-1) if self.is_regression else outputs.view(-1, self.num_labels)

                    # moving tensor to cpu and detaching for aggregation
                    outputs = outputs.detach().cpu().numpy()
                    labels = labels.cpu().numpy()
                    evaluated_samples += len(labels)

                    # calculate the accuracy in the classification case
                    if not self.is_regression:
                        outputs = np.argmax(outputs, axis=1)
                        # accuracy calculation
                        accuracy_sum += accuracy_score(labels, outputs) * len(labels)
                        print(f'{dataloader_type} ACC: {round(accuracy_sum / evaluated_samples, 5)}\r', end='')

                    # aggregate predictions and labels
                    all_predictions.extend(list(outputs))
                    all_labels.extend(list(labels))
        print()

        # calculate the required metrics
        results = {}
        for metric_name in TASK_TO_METRICS[self.task_name]:
            if 'F1' in metric_name and 'opt' in self.model_name:
                metric = METRIC_NAME_TO_FUNCTION[metric_name]
                result = metric(all_labels, all_predictions, average='macro')
            else:
                metric = METRIC_NAME_TO_FUNCTION[metric_name]
                if not self.is_regression:
                    result = metric(all_labels, all_predictions)
                else:
                    try:
                        result = metric(all_labels, all_predictions)[0]
                    except:
                        result = 0
            # result = result[0] if self.is_regression else result
            results[metric_name] = result

        return results

    # @staticmethod
    def _convert_dataset_to_data_loader(self, dataset, model_name, batch_size, random_sampler, test=False):
        """converts a datasets.arrow_dataset.Dataset to torch.utils.data.DataLoader.
        Args:
            dataset (datasets.arrow_dataset.Dataset): the Dataset to convert to DataLoader.
            model_name (str): model name (e.g. bert-base-uncased).
            batch_size (int): batch size for training and evaluation.
            random_sampler (bool): if True, DataLoader will sample randomly else sequentially.
            test (bool): if True, dataset contains test samples.
        Returns:
            (torch.utils.data.DataLoader): the data loader
        """
        if test:
            keys = ['input_ids', 'attention_mask', 'token_type_ids']
        else:
            keys = ['input_ids', 'attention_mask', 'token_type_ids', 'label']

        if 'roberta' in model_name or 'opt' in model_name:
            keys.remove('token_type_ids')

        batch_size = batch_size

        data = {key: list() for key in keys}
        for sample in dataset:
            for key in keys:
                data[key].append(sample[key])

        for k, v in data.items():
            data[k] = torch.tensor(v)

        tensor_dataset = TensorDataset(*[data[key] for key in keys])
        data_sampler = RandomSampler(tensor_dataset) if random_sampler else SequentialSampler(tensor_dataset)
        return DataLoader(tensor_dataset, sampler=data_sampler, batch_size=batch_size)
