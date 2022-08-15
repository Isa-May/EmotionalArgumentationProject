import numpy as np
import torch
from datasets import load_dataset

from transformers import AutoTokenizer


class EmotionDetectionDataLoader:
    OUTPUT_DATASET_FOLDER = '../../data_preprocessing/processed'

    """
    The EmotionDetectionDataLoader is a class, which simplifies the process of data_preprocessing preprocessing.
    It provides methods for creating datasets from a csv file, and avoids a lot of boilerplate code.
    """

    def __init__(self, pretrained_model_name_or_path):
        """
        Constructor. Initializes the tokenizer for a pretrained model.

        Params:
            pretrained_model_name_or_path (str)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False)

    def create_train_eval_test_dataset(self, csv, percentile):
        dataset = self.create_dataset(csv, percentile)

        train_dataset = dataset.filter(lambda x: x['Set'] == 'train')
        print(train_dataset)
        print(train_dataset)

        eval_dataset = dataset.filter(lambda x: x['Set'] == 'dev')
        test_dataset = dataset.filter(lambda x: x['Set'] == 'test')

        train_dataset.set_format('torch', columns=['input_ids', 'labels', 'attention_mask'])
        eval_dataset.set_format('torch', columns=['input_ids', 'labels', 'attention_mask'])
        test_dataset.set_format('torch', columns=['input_ids', 'labels', 'attention_mask'])

        return train_dataset, eval_dataset, test_dataset

    def create_formatted_test_dataset(self, csv, percentile):
        dataset = self.create_dataset(csv, percentile)
        test_dataset = dataset.filter(lambda x: x['Set'] == 'test')
        return test_dataset

    def create_formatted_dev_dataset(self, csv, percentile):
        dev_dataset = self.create_dataset(csv, percentile)
        return dev_dataset

    def create_dataset(self, csv, percentile):
        """
        Converts the given csv file into a dataset consisting of tensors.

        Params:
            csv (str): The path to the CSV file to be converted.
            percentile (int): The percentile for the maximum sentence length. Example:
                              99.5 == The maximum sentence length is larger than the length 99.5% of all sentences.
        Returns:
            dataset (Dataset): The dataset, which was extracted from the csv file.
        """
        dataset = load_dataset('csv', data_files=[csv])["train"]
        print(dataset)

        def get_max_length(column):
            def get_length(batch):
                return self.tokenizer(batch, return_length=True, padding=False)

            lengths = dataset.map(get_length, input_columns=column, batched=True, batch_size=32)['length']
            return int(np.percentile(lengths, percentile)) + 1

        max_len = get_max_length('Sentence')

        def format_dataset_row(row):
            text = row['Sentence']
            label = row['Emotion']

            encoding = self.tokenizer.encode_plus(
                text,
                truncation=True,
                add_special_tokens=True,
                max_length=max_len,
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
            )

            formatted = {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor([label], dtype=torch.long)
            }
            return formatted

        dataset = dataset.map(format_dataset_row)

        return dataset

#The MIT License (MIT)
#Copyright (c) 2021, Team Orange

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


