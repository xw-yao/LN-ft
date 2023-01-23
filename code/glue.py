from scipy.stats import spearmanr, pearsonr
import math
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
import numpy as np
import pickle
import logging
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from datasets import load_dataset
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, OPTForCausalLM,  GPT2Tokenizer
from utils import setup_logging
from datasets.arrow_dataset import Dataset
import torch.nn as nn
import wandb


setup_logging()
LOGGER = logging.getLogger(__file__)
wandb.init(project="opt350m-full_ft-sst2", entity="xwynlp")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

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
    "sst2": """{sentence1}\nWas that sentence "positive" or "negative"? It was {label}\n""",
    "mrpc": """Does the sentence\n{sentence1}\nparaphrase (that is, mean the same thing as) this sentence? (yes or no)\n{sentence2}\n{label}\n""",
    "qqp": """Are the questions "{sentence1}" and "{sentence2}" asking the same thing? (yes or no)\n{label}\n""",
    "stsb": """Rate on a scale from 0.0 to 5.0 how similar the sentences "{sentence1}" and "{sentence2}" are.\n{label}\n""",  # note that you might need to round the labels to 1 digit after the comma
}

TASK_TO_PROMPT_EVAL = {
    "cola": """The following sentence is either "acceptable", meaning it is grammatically correct and makes sense, or "unacceptable". Which is it?\n{sentence1}\n""",
    "sst2": """{sentence1}\nWas that sentence "positive" or "negative"? It was """,
    "mrpc": """Does the sentence\n{sentence1}\nparaphrase (that is, mean the same thing as) this sentence? (yes or no)\n{sentence2}\n""",
    "qqp": """Are the questions "{sentence1}" and "{sentence2}" asking the same thing? (yes or no)\n""",
    "stsb": """Rate on a scale from 0.0 to 5.0 how similar the sentences "{sentence1}" and "{sentence2}" are.\n""",  # note that you might need to round the labels to 1 digit after the comma
}

NUM_TO_TEXT = {
    "cola": ["unacceptable", "acceptable"],
    "sst2": ["negative", "positive"],
    "mrpc": ["no", "yes"],
    "qqp": ["no", "yes"],
}

BIAS_TERMS_DICT = {
    'intermediate': 'intermediate.dense.bias',
    'key': 'attention.self.key.bias',
    'query': 'attention.self.query.bias',
    'value': 'attention.self.value.bias',
    'output': 'output.dense.bias',
    'layernorm': 'LayerNorm',
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

    def __init__(self, task_name, model_name, device):
        """
        Args:
            task_name (str): task name, e.g. 'rte'.
            model_name (str): model name, e.g. 'bert-base-uncased'.
            device (int): GPU device to run on, if None will run on CPU.
        """
        self.task_name = task_name
        self.model_name = model_name
        self.device = device

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

    def preprocess_dataset(self, padding, max_sequence_len, batch_size):
        """Preprocess the train and validation datasets.
        Args:
            padding (str): padding method (currently 'max_length' is the suggested method)
            max_sequence_len (int): the maximum sequence length
            batch_size (int): training and evaluating batch size
            train_size (int): clip the train dataset size, if None will use all available samples
        """
        LOGGER.info(f'Downloading dataset: {self.task_name}')
        datasets = load_dataset('glue', self.task_name)

        self.batch_size = batch_size
        if self.model_name == 'opt':
            self.tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        is_regression = self.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            self.idx_to_label = {k: v for k, v in enumerate(datasets['train'].features['label'].__dict__['_int2str'])}
            self.num_labels = len(label_list)
        else:
            self.num_labels = 1

        sentence1_key, sentence2_key = TASK_TO_KEYS[self.task_name]

        def _generate_train_prompts(examples):
            prompted_example = (TASK_TO_PROMPT_TRAIN[self.task_name].format(sentence1=examples[sentence1_key],
                                                                            label=NUM_TO_TEXT[self.task_name][examples['label']])) if sentence2_key is None else \
                (TASK_TO_PROMPT_TRAIN[self.task_name].format(sentence1=examples[sentence1_key],
                                                             sentence2=examples[sentence2_key],
                                                             label=NUM_TO_TEXT[self.task_name][examples['label']])
            )
            result = self.tokenizer(prompted_example, padding=padding, max_length=max_sequence_len, truncation=True)
            return result

        def _generate_eval_prompts(examples):
            prompted_example = (
                TASK_TO_PROMPT_EVAL[self.task_name].format(sentence1=examples[sentence1_key])) if sentence2_key is None else (
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

        if self.model_name == 'opt':
            datasets['train'] = load_dataset('glue', self.task_name, split='train[:16]')
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
                                                                                   train='train' in dataset_name,
                                                                                   test='test' in dataset_name)
        # print(self.data_loaders.items())
        # exit()

    def training_preparation(self, learning_rate, encoder_trainable, weight_decay, lora_alpha=None, lora_r=None, ft_type=None, apply_lora=False, trainable_components=None, verbose=True):
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
        if self.model_name == 'opt':
            self.model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        else:
            if apply_lora:
                config = AutoConfig.from_pretrained(self.model_name, num_labels=self.num_labels, apply_lora=True, lora_alpha=lora_alpha, lora_r=lora_r, return_dict=True)
            else:
                config = AutoConfig.from_pretrained(self.model_name, num_labels=self.num_labels, return_dict=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config)

        if ft_type == 'outlier':
            self.model2 = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config)
            for name, param in self.model2.named_parameters():
                if 'LayerNorm' in name and not any(x in name for x in ['attention', 'embeddings']):
                    param.requires_grad = False
                    self.init_out_ln_params.append(param)

        if not encoder_trainable:
            self._deactivate_relevant_gradients(ft_type, trainable_components)

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, correct_bias=True, eps=1e-8, weight_decay=weight_decay)

        self.learning_rate = learning_rate

        if verbose:
            print('\n\nTrainable Components:\n----------------------------------------\n')
            total_trainable_params = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name, '  --->  ', param.shape)
                    total_trainable_params += param.shape[0] if len(param.shape) == 1 else param.shape[0] * param.shape[
                        1]
            print(
                f'\n----------------------------------------\nNumber of Trainable Parameters: {total_trainable_params}\n')

        self.evaluations = {metric_name: [] for metric_name in TASK_TO_METRICS[self.task_name]}

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
        if self.device is not None:
            self.model.cuda(self.device)

        # train and evaluate
        self.epochs = num_epochs
        # n = len(self.data_loaders['train'].dataset)
        # t_total = n // gradient_accumulation_steps * num_epochs
        # warmup_steps = math.ceil(t_total * warmup_ratio)
        # self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,
        #                                                  num_training_steps=t_total)
        for epoch in range(num_epochs):
            # training for a single epoch
            #print(f'init_param: {self.init_out_ln_params[0]}')
            self._train(self.data_loaders['train'], epoch, ft_type=ft_type)
            # print(f'init_param: {self.init_out_ln_params[0]}')

            # evaluation
            results = self._evaluate(self.data_loaders['validation'])
            for metric_name, result in results.items():
                self.evaluations[metric_name].append(result)

            # for dataloader_type, dataloader in self.data_loaders.items():
            #     if not ('test' in dataloader_type):
            #         results = self._evaluate(dataloader, dataloader_type.upper())
            #         for metric_name, result in results.items():
            #             self.evaluations[dataloader_type][metric_name].append(result)
            print('')
        #print(time_log)

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

    @staticmethod
    def load(path, gpu_device):
        """Loads the evaluator from `path`.
        Args:
            path (str): Directory to load to model from.
            gpu_device (int): GPU device ID.
        Returns:
            (GLUEvaluator): the GLUEvaluator instance we loaded
        """
        with open(path, 'rb') as file:
            data = pickle.load(file)
        evaluator = GLUEvaluator(data['task_name'], data['model_name'], gpu_device)
        evaluator.num_labels = data['num_labels']
        evaluator.batch_size = data['batch_size']
        evaluator.model = data['model']
        evaluator.learning_rate = data['learning_rate']
        evaluator.evaluations = data['evaluations']
        evaluator.encoder_trainable = data.get('encoder_trainable', None)

        return evaluator

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

        if ft_type == 'bitfit' and 'opt' in self.model_name:
            for param in self.model.parameters():
                param.requires_grad = False
            for name, param in self.model.named_parameters():
                if trainable_components[0] in name:
                    param.requires_grad = True

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

    def _train(self, train_dataloader, epoch, ft_type=None, max_grad_norm=1.0):
        """Trains the model for a single epoch
        Args:
            train_dataloader (torch.utils.data.DataLoader): the train data loader
            epoch (int): the epoch number (for logging)
            max_grad_norm (float): the maximum gradient norm we allow. The norm is computed over all gradients together,
            as if they were concatenated into a single vector.
        """

        if 'opt' in self.model_name:
            wandb.config = {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.opt_train_btach}
        else:
            wandb.config = {
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size}
        # move to train mode
        self.model.train()

        # loss initialization
        criteria = torch.nn.MSELoss() if self.is_regression else torch.nn.CrossEntropyLoss()

        n = len(train_dataloader.dataset)

        trained_samples = loss_sum = 0
        for step, batch in enumerate(train_dataloader):

            # move batch data to gpu
            if self.device is not None:
                batch = tuple(obj.cuda(self.device) for obj in batch)

            if 'roberta' in self.model_name or 'opt' in self.model_name:
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
                loss = outputs.loss
                wandb.log({"loss": loss})
                labels = labels.view(-1)
            else:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                outputs = outputs.logits
                labels = labels.view(-1)
                outputs = outputs.view(-1) if self.is_regression else outputs.view(-1, self.num_labels)
                loss = criteria(outputs, labels)
                wandb.log({"loss": loss})

            # backward pass (gradients calculation)
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
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)

            # update parameters
            self.optimizer.step()
            #print(f'param: {self.init_out_ln_params[0]}')

            if ft_type == 'outlier':
                for param in self.model.parameters():
                    param.requires_grad = False
                idx = 0
                for name, param in self.model.named_parameters():
                    if 'LayerNorm' in name and not any(x in name for x in ['attention', 'embeddings']):
                        for dim in range(768):
                            if dim != 308 and dim != 381:
                                param[dim] = self.init_out_ln_params[idx][dim]
                        idx += 1
                        param.requires_grad = True
                self.optimizer.step()
            #print(f'param2: {self.init_out_ln_params[0]}')

            #self.scheduler.step()
            self.model.zero_grad()

            # track train loss
            loss_sum += loss.item()
            trained_samples += len(labels)

            # printing training progress
            print(f'EPOCH: {epoch}   TRAIN: {trained_samples}/{n}   LOSS: {round(loss_sum / (step + 1), 3)}\r', end='')
        print('')

    def _evaluate(self, eval_dataloader):
        """Evaluates the model on the dataloader
        Args:
            dataloader (torch.utils.data.DataLoader): the data loader we evaluate the model on
            dataloader_type (str): the dataloader type (train/validation)
        Returns:
            (Dict[str, float]): dictionary that maps between metric_name and the metric result
        """
        # move to eval mode
        self.model.eval()

        evaluated_samples = accuracy_sum = 0
        all_predictions, all_labels = [], []
        for step, batch in enumerate(eval_dataloader):
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
                        # print(f'n_input_tokens {n_input_tokens}')
                        # print(f'output length: {len(outputs[idx])}')
                        response_ids = outputs[idx][n_input_tokens:]
                        # print(f'response_ids: {response_ids}')
                        # exit()
                        pred = self.tokenizer.decode(response_ids,
                                                    skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)
                        true_label = NUM_TO_TEXT[self.task_name][labels[idx]]
                        prompt_preds.append(pred)
                        true_labels.append(true_label)

                    labels = labels.view(-1)
                    labels = labels.cpu().numpy()

                    evaluated_samples += len(labels)
                    # print(f'pred labels: {prompt_preds}')
                    # print(f'true labels: {true_labels}')

                    # calculate the accuracy in the classification case
                    if not self.is_regression:
                        # accuracy calculation
                        accuracy_sum += accuracy_score(true_labels, prompt_preds) * len(labels)
                        print(f'VALID ACC: {round(accuracy_sum / evaluated_samples, 5)}\r', end='')

                    # aggregate predictions and labels
                    all_predictions.extend(prompt_preds)
                    all_labels.extend(true_labels)


                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
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
                        print(f'VALID ACC: {round(accuracy_sum / evaluated_samples, 5)}\r', end='')

                    # aggregate predictions and labels
                    all_predictions.extend(list(outputs))
                    all_labels.extend(list(labels))

        print('')

        # calculate the required metrics
        results = {}
        for metric_name in TASK_TO_METRICS[self.task_name]:
            if 'F1' in metric_name and 'opt' in self.model_name:
                metric = METRIC_NAME_TO_FUNCTION[metric_name]
                result = metric(all_labels, all_predictions, average='micro')
            else:
                metric = METRIC_NAME_TO_FUNCTION[metric_name]
                result = metric(all_labels, all_predictions)
            result = result[0] if self.is_regression else result
            results[metric_name] = result

        return results

    # @staticmethod
    def _convert_dataset_to_data_loader(self, dataset, model_name, batch_size, random_sampler, train=False, test=False):
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

        if 'opt' in model_name and train:
            batch_size = self.opt_train_btach = 1
        else:
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