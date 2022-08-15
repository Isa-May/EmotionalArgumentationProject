import argparse
import transformers
from models.features.emotion_detection_data_loader import EmotionDetectionDataLoader
from models.model_huggingface.basic_metric import basic_metric_for_classes
from models.model_huggingface.bert_classification_trainer import BertForSequenceClassificationTrainingFramework

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The emotion detection trainer.')
    parser.add_argument('--learning_rate', type=float, help='the learning rate of the model.', default=5e-5)
    parser.add_argument('--run_name', type=str, help='name of the run on mlflow', default="")
    parser.add_argument('--weight_decay', type=float, help='the rate of weight decay.', default=0.1)
    parser.add_argument('--num_train_epochs', type=int,
                        help='the number of complete passes through the training dataset.',
                        default=5)
    parser.add_argument('--model_name', type=str, help='the name of the model.', default='bert-base-uncased')
    parser.add_argument('--classes', nargs='+', help='the classes of the model.',
                        default=['Non-emotional', 'Emotional'])
    parser.add_argument('--eval_batch_size', type=int, help='amount of inputs per evaluation step.', default=32)
    parser.add_argument('--train_batch_size', type=int, help='amount of inputs per training step.', default=32)
    parser.add_argument('--max_len_percentile', type=float,
                        help='Maximum length percentile of input sentences to the model.', default=99.5)
    parser.add_argument('--csv', type=str, required=True, help='the path of CSV-file.')
    parser.add_argument('--warmup_steps', type=float,
                        help='After how many epochs the model should be fully warmed up '
                             'in regards to the learning rate.', default=0)
    parser.add_argument('--early_stopping_patience', type=int,
                        help='How many evaluation steps can be worse than the best one, '
                             'before the model aborts training.', default=12)
    parser.add_argument('--optimize', type=bool,
                        help='If the model should use hyperparameter optimization (optuna).', default=False)
    parser.add_argument('--evaluation_strategy', type=transformers.trainer_utils.IntervalStrategy,
                        help='Whether to use epoch or step count as evaluation strategy.', default='steps')
    args = parser.parse_args()

    # creates our formatted input data_preprocessing in the form of input ids, attention masks..
    data_loader = EmotionDetectionDataLoader(args.model_name)

    train_dataset, eval_dataset, _ = data_loader.create_train_eval_test_dataset(
        args.csv, args.max_len_percentile)

    metric = basic_metric_for_classes(args.classes)

    train_framework = BertForSequenceClassificationTrainingFramework.for_training(
        pretrained_model_name_or_path=args.model_name,
        run_name=args.run_name,
        warmup_steps=args.warmup_steps,
        early_stopping_patience=args.early_stopping_patience,
        evaluation_strategy=args.evaluation_strategy,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        weight_decay=args.weight_decay,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metric,
        num_train_epochs=args.num_train_epochs
    )

    train_framework.train()
    print(train_framework.evaluate())

#The MIT License (MIT)
#Copyright (c) 2021, Team Orange

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


