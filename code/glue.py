from scipy.stats import spearmanr, pearsonr
from time import time
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score
import numpy as np
import pickle
import logging
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from datasets import load_dataset
from transformers.optimization import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from utils import setup_logging


setup_logging()
LOGGER = logging.getLogger(__file__)

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

BIAS_TERMS_DICT = {
    'intermediate': 'intermediate.dense.bias',
    'key': 'attention.self.key.bias',
    'query': 'attention.self.query.bias',
    'value': 'attention.self.value.bias',
    'output': 'output.dense.bias',
    'layernorm': 'LayerNorm',
    'output_layernorm': 'output.LayerNorm.bias',
    'attention_layernorm': 'attention.output.LayerNorm.bias',
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
        self.model = None
        self.optimizer = None
        self.learning_rate = None
        self.evaluations = None
        self.encoder_trainable = None
        self.masks = None
        self.idx_to_label = None

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
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        is_regression = self.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            self.idx_to_label = {k: v for k, v in enumerate(datasets['train'].features['label'].__dict__['_int2str'])}
            self.num_labels = len(label_list)
        else:
            self.num_labels = 1

        sentence1_key, sentence2_key = TASK_TO_KEYS[self.task_name]

        def _preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (
                    examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*args, padding=padding, max_length=max_sequence_len, truncation=True)
            return result

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

    def training_preparation(self, learning_rate, encoder_trainable, ft_type=None, trainable_components=None, verbose=True):
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
        if apply_lora:
            config = AutoConfig.from_pretrained(self.model_name, num_labels=self.num_labels, apply_lora=True, lora_alpha=lora_alpha, lora_r=lora_r, return_dict=True)
        else:
            config = AutoConfig.from_pretrained(self.model_name, num_labels=self.num_labels, return_dict=True)

        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config)

        if not encoder_trainable:
            self._deactivate_relevant_gradients(ft_type, trainable_components)

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, correct_bias=True)

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

        self.evaluations = {k: {metric_name: [] for metric_name in TASK_TO_METRICS[self.task_name]} for k in
                            self.data_loaders.keys()}

    def train_and_evaluate(self, num_epochs, evaluation_frequency=1):
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
        time_log = []
        for epoch in range(num_epochs):
            start_time = time()
            # training for a single epoch
            self._train(self.data_loaders['train'], epoch)
            end_time = time()
            time_log.append((end_time - start_time))

            # evaluation
            if not epoch % evaluation_frequency:
                for dataloader_type, dataloader in self.data_loaders.items():
                    if not ('test' in dataloader_type):
                        results = self._evaluate(dataloader, dataloader_type.upper())
                        for metric_name, result in results.items():
                            self.evaluations[dataloader_type][metric_name].append(result)
            print('')
        print(time_log)

    def save(self, output_path):
        """Saves the evaluator to the output_path directory.
        Args:
            output_path (str): Directory to save to model to.
        """
        LOGGER.info(f'Saving the model to: {output_path}')

        self.model.cpu()
        data = {'model': self.model, 'model_name': self.model_name, 'task_name': self.task_name,
                'learning_rate': self.learning_rate, 'evaluations': self.evaluations,
                'batch_size': self.batch_size, 'num_labels': self.num_labels,
                'encoder_trainable': self.encoder_trainable}
        with open(output_path, 'wb') as file:
            pickle.dump(data, file)

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

        if ft_type == 'outlier':
            for param in self.model.parameters():
                param.requires_grad = False

            for name, param in self.model.named_parameters():
                if all(x in name for x in [trainable_components[0], 'weight']) and not any(
                        x in name for x in ['attention', 'embeddings']):
                    for dim in range(768):
                        if dim != 308 and dim != 381:
                            param[dim] = 0
                    param.requires_grad = True

                if all(x in name for x in [trainable_components[0], 'bias']) and not any(
                        x in name for x in ['attention', 'embeddings']):
                    for dim in range(768):
                        if dim != 308 and dim != 381:
                            param[dim] = 0
                    param.requires_grad = True

        if ft_type == 'layernorm':
            for name, param in self.model.named_parameters():
                if not trainable_components[0] in name:
                    param.requires_grad = False

        if ft_type == 'bitfit_ln':
            for param in self.model.parameters():
                param.requires_grad = False
            trainable_components = trainable_components + ['classifier', 'pooler.dense.bias', 'LayerNorm']
            for name, param in self.model.named_parameters():
                for component in trainable_components:
                    if component in name:
                        param.requires_grad = True
                        break

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

        if ft_type == 'lora':
            for name, param in self.model.named_parameters():
                if not trainable_components[0] in name:
                    param.requires_grad = False


    @staticmethod
    def convert_to_actual_components(components):
        return [BIAS_TERMS_DICT[component] for component in components]

    def _train(self, train_dataloader, epoch, max_grad_norm=1.0):
        """Trains the model for a single epoch
        Args:
            train_dataloader (torch.utils.data.DataLoader): the train data loader
            epoch (int): the epoch number (for logging)
            max_grad_norm (float): the maximum gradient norm we allow. The norm is computed over all gradients together,
            as if they were concatenated into a single vector.
        """
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

            if 'roberta' in self.model_name:
                input_ids, attention_mask, labels = batch
                token_type_ids = None
            else:
                input_ids, attention_mask, token_type_ids, labels = batch

            # forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            outputs = outputs.logits

            # loss calculation
            labels = labels.view(-1)
            outputs = outputs.view(-1) if self.is_regression else outputs.view(-1, self.num_labels)

            loss = criteria(outputs, labels)

            # backward pass (gradients calculation)
            loss.backward()

            # masking the relevant gradients (if needed)
            if self.masks:
                if 'roberta' in self.model_name:
                    for name, param in self.model.roberta.named_parameters():
                        param.grad[~self.masks[name]] = 0
                else:
                    for name, param in self.model.bert.named_parameters():
                        param.grad[~self.masks[name]] = 0

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_grad_norm)

            # update parameters
            self.optimizer.step()
            self.model.zero_grad()

            # track train loss
            loss_sum += loss.item()
            trained_samples += len(labels)

            # printing training progress
            print(f'EPOCH: {epoch}   TRAIN: {trained_samples}/{n}   LOSS: {round(loss_sum / (step + 1), 3)}\r', end='')
        print('')

    def _evaluate(self, dataloader, dataloader_type):
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
        for step, batch in enumerate(dataloader):
            # move batch data to gpu
            if self.device is not None:
                batch = tuple(obj.cuda(self.device) for obj in batch)

            if 'roberta' in self.model_name:
                input_ids, attention_mask, labels = batch
                token_type_ids = None
            else:
                input_ids, attention_mask, token_type_ids, labels = batch

            # forward pass
            with torch.no_grad():
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
                print(f'{dataloader_type} ACC: {round(accuracy_sum / evaluated_samples, 5)}\r', end='')


            # aggregate predictions and labels
            all_predictions.extend(list(outputs))
            all_labels.extend(list(labels))
        print('')

        # calculate the required metrics
        results = {}
        for metric_name in TASK_TO_METRICS[self.task_name]:
            metric = METRIC_NAME_TO_FUNCTION[metric_name]
            result = metric(all_labels, all_predictions)
            result = result[0] if self.is_regression else result
            results[metric_name] = result

        return results

    @staticmethod
    def _convert_dataset_to_data_loader(dataset, model_name, batch_size, random_sampler, test=False):
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

        if 'roberta' in model_name:
            keys.remove('token_type_ids')

        data = {key: list() for key in keys}
        for sample in dataset:
            for key in keys:
                data[key].append(sample[key])

        for k, v in data.items():
            data[k] = torch.tensor(v)

        tensor_dataset = TensorDataset(*[data[key] for key in keys])
        data_sampler = RandomSampler(tensor_dataset) if random_sampler else SequentialSampler(tensor_dataset)
        return DataLoader(tensor_dataset, sampler=data_sampler, batch_size=batch_size)