import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from collections import defaultdict

import torch
from transformers import Trainer, EarlyStoppingCallback, TrainingArguments

from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss
from transformers import BertTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BERT_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC,
    BertModel, _CHECKPOINT_FOR_DOC,
)

from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
)


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def get_number_of_labels(dataset):
    return max([label.item() for label in dataset["labels"]]) + 1


def convert_epoch_to_step_count(dataset_size, batch_size):
    """
    Helper method for converting an epoch to the amount of corresponding steps given the dataset and batch size.

    Params:
        dataset_size (int): The amount of samples in the dataset.
        batch_size (int): The amount of samples processed in a single batch.

    Returns:
        step_count (int): The amount of steps corresponding to a fourth epoch, given the input parameters.
    """
    step_count = dataset_size / batch_size

    return int(step_count / 4)


def compute_label_weights(dataset):
    """
    Helper method for calculating relative weights for each label.
    Can be used to improving the loss function in training,
    so it takes into account if a class has less instances than another (and therefore has a weigher weight).

    Params:
        dataset (Dataset): The dataset which contains all training instances.

    Returns:
        label_weights (List[int]): A list of length num_labels, with each entry representing the weight of a class.
    """

    label_counts = defaultdict(int)  # init new empty dict
    for label in dataset['labels']:
        label_counts[label.item()] += 1
    num_labels = max(label_counts) + 1
    total_count = sum(label_counts.values())
    weights = [(1 / num_labels) / (label_counts[label] / total_count)
               if label in label_counts else 0 for label in range(num_labels)]
    return torch.tensor(weights, dtype=torch.float)


class MultilabelTrainer(Trainer):
    """
    The MultilabelTrainer implements a custom loss function,
    which allows the weighting of labels when calculating the total loss.

    This is done since the classes in our dataset are not uniformly distributed.
    """

    # def __init__(self, *args, **kwargs):
    def __init__(self, label_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_weights = label_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.label_weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels),
                        labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class BertForSequenceClassificationTrainingFramework:
    """
    The BertForSequenceClassificationTrainingFramework is a wrapper around the
    `TrainingArguments` and `MultilabelTrainer` (derived from `Trainer`) classes provided by huggingface.

    It provides methods for initializing and using a new training framework instance either for training, or for evaluation.
    """

    def __init__(self, trainer):
        """
        Constructor. Saves the trainer which is wrapped by this class.

        Params:
            trainer (Trainer): the trainer provided by the huggingface library.
        """
        self.trainer = trainer

    @classmethod
    # pylint: disable=too-many-arguments
    def for_training(cls, pretrained_model_name_or_path, run_name, warmup_steps,
                     early_stopping_patience, evaluation_strategy, learning_rate, train_batch_size,
                     eval_batch_size, weight_decay, train_dataset, eval_dataset, compute_metrics, num_train_epochs):
        """
        Initializes the training framework for training.

        Params:
            pretrained_model_name_or_path (str): the name or path of the pre-trained model.
            run_name (str): the name of the run for mlflow.
            logging_epoch_step (float): the epoch interval for logging artifacts/metrics.
            warmup_epoch_step (float): the amount of epochs before the learning_rate is using the maximum value.
            early_stopping_patience (int): for how many epochs the loss can get worse before training is stopped.
            learning_rate (float): the learning rate of the trainer.
            epochs (float): the amount of epochs the trainer will train for.
            train_batch_size (int): the amount of instances passed to the model per training step.
            eval_batch_size (int): the amount of instances passed to the model per evaluation step.
            weight_decay (float): the weight decay to apply to the model.
            train_dataset (Dataset): the dataset used for training the model.
            eval_dataset (Dataset): the dataset used for evaluating the model.
            compute_metrics (function): the function for calculating the metrics.
            optimize (bool): If hyperparameter optimization should be done.
            optimize_trials (int): The amount of trials, if hyperparameter optimization is enabled.
        """
        eval_steps = convert_epoch_to_step_count(len(train_dataset), train_batch_size)
        device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
        # Memo: use label_weights in alternative setting only
        label_weights = compute_label_weights(train_dataset).to(device)
        num_labels = max(get_number_of_labels(train_dataset), get_number_of_labels(eval_dataset))

        def model_init():
            return BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path,
                                                                 num_labels=num_labels).to(device)

        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False)
        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)

        training_args = TrainingArguments(output_dir='output/' + run_name,
                                          run_name=run_name,
                                          logging_dir=None,
                                          learning_rate=learning_rate,
                                          eval_steps=eval_steps,
                                          num_train_epochs=num_train_epochs,
                                          per_device_train_batch_size=train_batch_size,
                                          per_device_eval_batch_size=eval_batch_size,
                                          warmup_steps=warmup_steps,
                                          weight_decay=weight_decay,
                                          evaluation_strategy=evaluation_strategy,
                                          logging_steps=eval_steps,
                                          load_best_model_at_end=True,
                                          save_total_limit=1,
                                          )

        trainer = MultilabelTrainer(
            model_init=model_init,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            callbacks=[early_stopping_callback],
            label_weights=label_weights
        )

        return cls(trainer)

    def train(self, *args, **kwargs):
        """
        Train the model. Wrapper for the trainer's train method.
        """
        return self.trainer.train(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        """
        Evaluate the model. Wrapper for the trainer's evaluate method.
        """
        return self.trainer.evaluate(*args, **kwargs)

    def predict(self, *args, **kwargs):
        """
        Predict labels using the model. Wrapper for the trainer's predict method.
        """
        return self.trainer.predict(*args, **kwargs)

    @classmethod
    def for_evaluation(cls, pretrained_model_name_or_path, eval_batch_size, evaluation_strategy, metric=None):
        """
        Initializes the training framework for evaluation.

        Params:
            pretrained_model_name_or_path (str): the path to the trained model.
            eval_batch_size (int): the amount of instances passed to the model per evaluation step.
            metric (Optional[function]): an addtional function for calculating the metrics.
        """

        training_args = TrainingArguments(output_dir='output/evaluation/',
                                          per_device_eval_batch_size=eval_batch_size,
                                          evaluation_strategy=evaluation_strategy)
        device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
        model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path).to(device)

        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            compute_metrics=metric
        )

        return cls(trainer)

    @classmethod
    def for_testing(cls, pretrained_model_name_or_path, test_batch_size, test_dataset, evaluation_strategy,
                    metric=None):
        """
        Initializes the training framework for evaluation.

        Params:
            pretrained_model_name_or_path (str): the path to the trained model.
            eval_batch_size (int): the amount of instances passed to the model per evaluation step.
            metric (Optional[function]): an addtional function for calculating the metrics.
        """

        training_args = TrainingArguments(output_dir='output/validation/',
                                          per_device_eval_batch_size=test_batch_size,
                                          evaluation_strategy=evaluation_strategy)
        device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
        model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path).to(device)

        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False)

        trainer = MultilabelTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            compute_metrics=metric,
            test_dataset=test_dataset,
            label_weights=0
        )

        return cls(trainer)


#The MIT License (MIT)
#Copyright (c) 2021, Team Orange

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


