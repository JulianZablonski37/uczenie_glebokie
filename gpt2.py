import logging
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import GPT2Model, GPT2ForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


logger = logging.getLogger(__name__)


# Simple version #

class GPT2ClassificationHeadCustomSimple(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.n_embd
        self.dense_1 = nn.Linear(hidden_size, 2 * hidden_size)
        self.dense_2 = nn.Linear(2 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.out_proj = nn.Linear(hidden_size, config.num_labels, bias=False)

    def forward(self, x):
        x = self.dense_1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.dense_2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.out_proj(x)
        return x


class GPT2ForSequenceClassificationCustomSimple(GPT2ForSequenceClassification):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.score = GPT2ClassificationHeadCustomSimple(config)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()


# Version with custom forward 1 #

class GPT2ClassificationHeadCustom(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.n_embd
        self.dense_1_input = nn.Linear(hidden_size, 2 * hidden_size)
        self.dense_1_hidden = nn.Linear(hidden_size, 2 * hidden_size)
        self.dense_2 = nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.out_proj = nn.Linear(hidden_size, config.num_labels, bias=False)

    def forward(self, x, **kwargs):
        if 'hidden_states' in kwargs and kwargs['hidden_states'] is not None:
            # Get hidden states from second from the end
            hidden = kwargs['hidden_states'][-2]
        else:
            hidden = torch.zeros(x.size(), dtype=x.dtype, device=x.device)

        x = self.dense_1_input(x)
        x = torch.nn.LeakyReLU(x)
        x = self.dropout(x)

        hidden = self.dense_1_hidden(hidden)
        hidden = torch.nn.LeakyReLU(hidden)
        hidden = self.dropout(hidden)

        x = torch.cat((x, hidden), dim=2)
        x = self.dense_2(x)
        x = torch.nn.LeakyReLU(x)
        x = self.dropout(x)

        x = self.out_proj(x)
        return x


class GPT2ForSequenceClassificationCustom(GPT2ForSequenceClassification):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.score = GPT2ClassificationHeadCustom(config)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states or self.config.use_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states, hidden_states=transformer_outputs.hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
