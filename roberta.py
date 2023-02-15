from typing import Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import RobertaForSequenceClassification, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput


class RobertaClassificationHeadCustomSimple(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        self.dense_1 = nn.Linear(hidden_size, 2 * hidden_size)
        self.dense_2 = nn.Linear(2 * hidden_size, hidden_size)
        self.dense_3 = nn.Linear(hidden_size, 2 * hidden_size)
        self.dense_4 = nn.Linear(2 * hidden_size, hidden_size)
        self.dense_5 = nn.Linear(hidden_size, 2 * hidden_size)
        self.dense_6 = nn.Linear(2 * hidden_size, hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])

        x = self.dense_1(x)
        x = torch.rrelu(x)
        x = self.dropout(x)

        x = self.dense_2(x)
        x = torch.rrelu(x)
        x = self.dropout(x)
        
        x = self.dense_3(x)
        x = torch.rrelu(x)
        x = self.dropout(x)

        x = self.dense_4(x)
        x = torch.rrelu(x)
        x = self.dropout(x)

        x = self.dense_5(x)
        x = torch.rrelu(x)
        x = self.dropout(x)

        x = self.dense_6(x)
        x = torch.rrelu(x)
        x = self.dropout(x)

        x = self.out_proj(x)
        return x

class RobertaForSequenceClassificationCustomSimple(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHeadCustomSimple(config)

        self.init_weights()
