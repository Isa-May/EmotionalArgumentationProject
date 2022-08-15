import argparse
import numpy as np
import pandas
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from transformers import AdamW, get_linear_schedule_with_warmup
from models.features.emotion_detection_data_loader import EmotionDetectionDataLoader
from models.features.pytorchtools import EarlyStopping
from models.model_pytorch_native.ensemble_model import BertForSequenceClassificationEnsemble

device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')

"""
Parse arguments needed for training, evaluation (dev_csv) and validation (test_csv)
"""

parser = argparse.ArgumentParser(description='The emotion detection trainer.')
parser.add_argument('--learning_rate', type=float, help='the learning rate of the model.', default=5e-5)
parser.add_argument('--run_name', type=str, help='name of the run on mlflow', default="")
parser.add_argument('--model_name', type=str, help='the name of the model.', default='bert-base-uncased')
parser.add_argument('--train_csv', type=str, required=True, help='the path of the train CSV-file.')
parser.add_argument('--dev_csv', type=str, required=True, help='the path of dev CSV-file.')
parser.add_argument('--predictions_path', type=str, required=True, help='the path to the predictions-file.')
args = parser.parse_args()


def train_model():
    """
    create the train dataset: one dataset and two dataloaders are needed
    """
    data_loader = EmotionDetectionDataLoader('bert-base-uncased')
    train_dataloader_1, _, _ = data_loader.create_train_eval_test_dataset(args.train_csv, 99.5)
    train_dataloader_2 = train_dataloader_1

    """
    create the dev dataset: one dataset and two dataloaders are needed
    """
    data_loader = EmotionDetectionDataLoader('bert-base-uncased')
    _, dev_dataloader_1, _ = data_loader.create_train_eval_test_dataset(args.dev_csv, 99.5)
    dev_dataloader_2 = dev_dataloader_1



    """
    initialize the model and make sure it's on the correct device
    """
    model = BertForSequenceClassificationEnsemble().to(device)


    """
    info on the model
    """
    params = list(model.named_parameters())
    modelDescription = str(model)
    print(modelDescription)

    """
    set the amount of training epochs
    """
    epochs = 5

    """
    set the optimizer
    """
    optimizer = AdamW(params=model.parameters(), lr=5e-5)

    """
    train sentences
    """
    train_dataset = pandas.read_csv(args.train_csv)
    train_sentences = []
    for row in train_dataset['Sentence']:
        train_sentences.append(row)

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(train_dataset) * epochs)

    """
    train the model
    """
    training_stats = []
    train_losses = []
    eval_losses = []

    early_stopping = EarlyStopping(patience=3, verbose=True)

    for epoch in range(0, epochs):
        # iterate over the two inputs simultaneously
        for step, combined_batch in enumerate(zip(train_dataloader_1, train_dataloader_2)):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch, epochs))
            print('Training...')

            train_batch_1, train_batch_2 = combined_batch
            model.train()
            """
            clear gradients before training
            """
            model.zero_grad()

            """
            pass input_ids, attention_masks and labels to the forward loop
            and receive results here
            """
            outputs = model(torch.unsqueeze(train_batch_1.get('input_ids'), 0).to(device), torch.unsqueeze(train_batch_2.get('input_ids'), 0).to(device),
                            torch.unsqueeze(train_batch_1.get('attention_mask'), 0).to(device), torch.unsqueeze(train_batch_1.get('attention_mask'),0).to(device),
                            torch.unsqueeze(train_batch_1.get('labels'),0).to(device))

            loss = outputs[0]
            train_losses.append(loss)
            loss.backward()

            # Clip the norm of the gradients to 1.0 to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            print(f"epoch:{epoch}, loss:{loss}")

            optimizer.step()
            scheduler.step()
        print("\n")


        """
        evaluate: after each epoch measure current model's performance on dev set
        """
        eval_predictions, eval_ground_truth = [], []
        model.eval()

        # iterate the test_dataloader_1 and the test_dataloader_2 inputs simultaneously
        for step, combined_batch in enumerate(zip(dev_dataloader_1, dev_dataloader_2)):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
            print('Evaluating...')

            batch_1, batch_2 = combined_batch

            with torch.no_grad():
                outputs = model(torch.unsqueeze(batch_1.get('input_ids'), 0).to(device),
                                torch.unsqueeze(batch_2.get('input_ids'), 0).to(device),
                                torch.unsqueeze(batch_1.get('attention_mask'), 0).to(device),
                                torch.unsqueeze(batch_2.get('attention_mask'), 0).to(device),
                                torch.unsqueeze(batch_1.get('labels'), 0).to(device))

                eval_loss = outputs[0]

                eval_losses.append(eval_loss)

                tmp_eval_loss, logits = outputs[:2]
                logits = logits.detach().cpu().numpy()
                outputs = np.argmax(logits, axis=1)
                label_ids = batch_1.get('labels').to('cpu').numpy()

                eval_predictions.extend(outputs)
                eval_ground_truth.extend(label_ids)

                eval_f1 = f1_score(eval_ground_truth, eval_predictions, average='macro')

                print('======== Eval Macro F1 predictions {:}'.format(eval_f1))
        print('======== Eval predictions {:} / Eval ground truth {:} ========'.format(eval_predictions,
                                                                                      eval_ground_truth))

        train_losses_cpu = []
        for val in train_losses:
            cpu_val = val.detach().cpu().numpy()
            train_losses_cpu.append(cpu_val)

        eval_losses_cpu = []
        for val in eval_losses:
            cpu_val_ = val.detach().cpu().numpy()
            eval_losses_cpu.append(cpu_val_)

        train_losses = train_losses_cpu
        eval_losses = eval_losses_cpu

        train_loss_ = np.average(train_losses)
        eval_loss_ = np.average(eval_losses)

        training_stats.append(
            {
                'epoch': epoch,
                'Training loss': train_loss_,
                'Evaluation loss': eval_loss_,
                'Eval macro f1': eval_f1
            }
        )

        epoch_len = len(str(epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss_:.5f} ' +
                     f'eval_loss: {eval_loss_:.5f}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        eval_losses = []

        """
        implement early stopping
        """
        early_stopping(eval_loss_, model)

        """checks if validation loss has decreased and if yes, saves model"""
        output_dir = 'output/' + args.run_name
        if early_stopping.early_stop:
            print("Early stopping")
            print("Saving model to %s" % output_dir)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': eval_loss,
            }, output_dir)
            break

        # display floats with two decimal places.
        pandas.set_option('precision', 2)

        # create a dataframe from the training statistics
        df_stats = pandas.DataFrame(data=training_stats)

        # use 'epoch' as the row index
        df_stats = df_stats.set_index('epoch')

        # style the plot
        sns.set(style='darkgrid')

        # increase plot and font size
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12, 6)

        # plot the training
        plt.plot(df_stats['Training loss'], 'b-o', label="Train loss")
        plt.plot(df_stats['Evaluation loss'], 'g-o', label="Eval loss")
        plt.plot(df_stats['Eval macro f1'], 'r-o', label="Eval Macro F1")

        # label the plot
        plt.title("Training & Evaluation Loss & Macro F1")
        plt.xlabel("Epoch")
        plt.ylabel("Loss and eval macro f1")
        plt.legend()
        plt.xticks([0, 1, 2, 3])

        plt.show()



if __name__ == '__main__':
    train_model()
