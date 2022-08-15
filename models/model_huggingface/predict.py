import argparse
import transformers

from models.features.argument_mining_data_loader import ArgumentMiningDataLoader
from models.features.format_predictions import format_predictions
from models.model_huggingface.bert_classification_trainer import BertForSequenceClassificationTrainingFramework
from models.model_huggingface.basic_metric import basic_metric_for_classes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The argument mining prediction.')

    # pylint: disable=duplicate-code
    parser.add_argument('--classes', nargs='+', help='the classes of the model output.',
                        default=['Non-emotional', 'Emotional'])
    parser.add_argument('--eval_batch_size', type=int, help='amount of inputs per evaluation step.', default=32)
    parser.add_argument('--max_len_percentile', type=int,
                        help='Maximum length percentile of input sentences to the model.', default=99.5)
    parser.add_argument('--trained_model_path', type=str, required=True, help='the path of the trained model.')
    parser.add_argument('--csv', type=str, required=True, help='the path to the Argument Mining CSV-file.')
    parser.add_argument('--predictions_path', type=str, required=True,
                        help='file and path to save the predictions')
    parser.add_argument('--evaluation_strategy', type=transformers.trainer_utils.IntervalStrategy,
                        help='Whether to use epoch or step count as evaluation strategy.', default='steps')

    args = parser.parse_args()

    data_loader = ArgumentMiningDataLoader(args.trained_model_path)
    dataset_test = data_loader.create_dataset(csv=args.csv, percentile=args.max_len_percentile)

    metric = basic_metric_for_classes(args.classes)

    prediction_framework = BertForSequenceClassificationTrainingFramework.for_evaluation(
        args.trained_model_path, args.eval_batch_size, args.evaluation_strategy, metric=metric)

    print(prediction_framework.predict(dataset_test))

    raw_predictions = prediction_framework.predict(dataset_test)

    # dataset with predictions added in the .csv
    final_dataset = format_predictions(raw_predictions, args.csv)
    final_dataset.to_csv(args.predictions_path)

#The MIT License (MIT)
#Copyright (c) 2021, Team Orange

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


