import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel

device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')


class BertForSequenceClassificationEnsemble(nn.Module):
    def __init__(self):
        super(BertForSequenceClassificationEnsemble, self).__init__()

        self.num_labels = 2

        """
        arg-bert ensemble
        """
        self.bert1 = BertModel.from_pretrained('output/ArgBERTSeed1/checkpoint-20', num_labels=2,
                                               output_attentions=False, output_hidden_states=False)
        self.bert2 = BertModel.from_pretrained('output/ArgBERTSeed2/checkpoint-40', num_labels=2,
                                               output_attentions=False, output_hidden_states=False)

        """
        arg-bert-emo-init ensemble
        """
        # self.bert1 = BertModel.from_pretrained('output/EmoBERTSeed1/checkpoint-1824',  num_labels = 2, output_attentions = False, output_hidden_states = False)
        # self.bert2 = BertModel.from_pretrained('output/EmoBERTSeed3/checkpoint-1216', num_labels = 2, output_attentions = False, output_hidden_states = False)

        """
        arg-bert PLUS arg-bert-emo-init ensemble
        """
        # self.bert1 = BertModel.from_pretrained('output/EmoBERTSeed1/checkpoint-1824',  num_labels = 2, output_attentions = False, output_hidden_states = False)
        # self.bert2 = BertModel.from_pretrained('bert-base-uncased',  num_labels = 2, output_attentions = False, output_hidden_states = False)

        """
        Bert's dropout is 0.1
        """
        self.dropout = nn.Dropout(0.1)

        """
        Bert base's hidden size
        """

        self.hidden_size = 768
        self.classifier = nn.Linear(self.hidden_size * 2, 2)

    def forward(self, input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, labels=None):
        outputs1 = self.bert1(input_ids_1,
                              attention_mask=attention_mask_1)
        outputs2 = self.bert1(input_ids_2,
                              attention_mask=attention_mask_2)

        pooled_output1 = outputs1.pooler_output
        pooled_output2 = outputs2.pooler_output

        """
        shared classifier
        """
        concat = torch.cat((pooled_output1, pooled_output2), axis=1)
        logits = self.classifier(concat)

        outputs = logits

        """
        weights in order to tackle class imbalance
        """
        weights = torch.tensor([0.8792, 1.1593]).to(device)

        if labels is not None:
            loss_fct = CrossEntropyLoss(weights)
            loss = loss_fct(outputs.view(-1, 2), labels.view(-1))
            return loss, logits
        else:
            return logits
