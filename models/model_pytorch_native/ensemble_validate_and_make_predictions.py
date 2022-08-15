import argparse
import numpy as np
import pandas
import torch
from sklearn.metrics import f1_score
from transformers import AdamW

from models.features.emotion_detection_data_loader import EmotionDetectionDataLoader
from models.model_pytorch_native.ensemble_model import BertForSequenceClassificationEnsemble


def validate_model(test_csv, model_name, predictions_path):
    device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')

    """
    Create the test files
    """
    """
    create the test dataset: one dataset and two dataloaders are needed
    """
    data_loader = EmotionDetectionDataLoader('bert-base-uncased')
    _, _, test_dataloader_1 = data_loader.create_train_eval_test_dataset(test_csv, 99.5)
    test_dataloader_2 = test_dataloader_1

    """
    Test the model
    """
    """
    Load the best model from above
    """
    """ we need the raw predictions for the file with the predictions; the raw_predictions are the logits"""
    raw_predictions = []
    model = BertForSequenceClassificationEnsemble().to(device)
    checkpoint = torch.load('output/' + model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = AdamW(params=model.parameters(), lr=5e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_ = checkpoint['epoch']
    print('Testing if early stopping is working via printing the epoch of model saved at early stopping epoch')
    print(epoch_)

    test_f1 = 0
    epochs = 5
    model.eval()
    for epoch in range(0, epochs):
        test_predictions, test_ground_truth = [], []
        # iterate the test_dataloader_1 and the test_dataloader_2 inputs simultaneously
        for step, combined_batch in enumerate(zip(test_dataloader_1, test_dataloader_2)):
            # only forward pass so no dropout
            model.eval()
            batch_1, batch_2 = combined_batch

            with torch.no_grad():
                outputs = model(torch.unsqueeze(batch_1.get('input_ids'), 0).to(device),
                                torch.unsqueeze(batch_2.get('input_ids'), 0).to(device),
                                torch.unsqueeze(batch_1.get('attention_mask'), 0).to(device),
                                torch.unsqueeze(batch_2.get('attention_mask'), 0).to(device),
                                labels=torch.unsqueeze(batch_1.get('labels'), 0).to(device))

                tmp_eval_loss, logits = outputs[:2]
                logits = logits.detach().cpu().numpy()
                outputs = np.argmax(logits, axis=1)
                label_ids = batch_1.get('labels').to('cpu').numpy()
                raw_predictions.extend(logits)

            test_predictions.extend(outputs)
            test_ground_truth.extend(label_ids)
            test_f1 = f1_score(test_ground_truth, test_predictions, average='macro')

    print('======== Test Macro F1 predictions {:}'.format(test_f1))

    """
    test sentences
    """
    test_sentences = pandas.read_csv(test_csv)
    test_sentences_ = []
    for row in test_sentences['Sentence']:
        test_sentences_.append(row)

    print('======== Test predictions {:} / Test ground truth {:} ========'.format(test_predictions, test_ground_truth))

    """
    MAKE PREDICTIONS FILE
    """
    apply_softmax_dim_one = torch.nn.Softmax(dim=1)
    predictions_tensor = torch.tensor(raw_predictions)
    predictions_softmax = apply_softmax_dim_one(predictions_tensor).tolist()
    predictions = np.argmax(raw_predictions, axis=1)
    predictions_confidence = np.max(predictions_softmax, axis=1)
    predictions = pandas.DataFrame(predictions, columns=["label"])
    predictions_confidence = pandas.DataFrame(predictions_confidence, columns=["confidence"])
    dataset = pandas.read_csv(test_csv, header=0)
    dataset['label'] = predictions
    dataset['confidence'] = predictions_confidence

    final_dataset = dataset
    final_dataset.pop('Id')
    final_dataset = final_dataset.reset_index(drop=True)
    final_dataset.index.name = 'Id'
    final_dataset.to_csv(predictions_path)


def make_predictions():
    parser = argparse.ArgumentParser(description='The arg parser for validating the ensemble model.')
    parser.add_argument('--learning_rate', type=float, help='the learning rate of the model.', default=5e-5)
    parser.add_argument('--model_name', type=str, required=True, help='the name of the pretrained model.')
    parser.add_argument('--test_csv', type=str, required=True, help='the path of test CSV-file.')
    parser.add_argument('--predictions_path', type=str, required=True, help='the path to the predictions-file.')
    args = parser.parse_args()
    validate_model(args.test_csv, args.model_name, args.predictions_path)


if __name__ == '__main__':
    make_predictions()
