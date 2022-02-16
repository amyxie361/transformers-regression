import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert import BertPreTrainedModel, BertModel
from transformers.models.distilbert import DistilBertModel

class BertForDistillSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, teacher_model=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.teacher_model = teacher_model
        #self.teacher_model = self.teacher_model.eval()

        #self.bert = BertModel(config)
        if config.model_flag == "distil":
            self.bert = DistilBertModel(config)
            #classifier_dropout = config.hidden_dropout_prob
        else:
            self.bert = BertModel(config)
        
            #classifier_dropout = (
            #    config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
            #)
        #self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        #self.gate = nn.Sequential(
        #    nn.Dropout(classifier_dropout), 
        #    nn.Linear(config.hidden_size, 1),
        #    nn.Sigmoid(),
        #)

        self.init_weights()
    """
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    """
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
        #if not self.gate_require_grad:
        #    for param in self.gate.parameters():
        #        param.requires_grad = False

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
        logits = self.classifier(pooled_output)
        self.teacher_model.eval()
        tea_logits = self.teacher_model(
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
        tea_logits = tea_logits.logits.detach()

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
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

            softmax = nn.Softmax(dim=1)
            distill_loss_fct = nn.KLDivLoss(reduction='none')
            device = outputs[1].get_device()

            labels_ = labels.reshape(labels.size()[0], 1)
            labels_one_hot = (labels_ == torch.arange(self.num_labels).reshape(1, self.num_labels).to(device)).float()

            loss_label_std = distill_loss_fct(
                F.log_softmax(logits, dim=-1),
                labels_one_hot
            )
            loss_label_tea = distill_loss_fct(
                F.log_softmax(tea_logits, dim=-1),
                labels_one_hot
            )
            loss_tea_std = distill_loss_fct(
                F.log_softmax(logits / self.config.tempreture, dim=-1),
                F.log_softmax(tea_logits / self.config.tempreture, dim=-1)
            ) * (self.config.tempreture ** 2)
            mask = torch.gt(torch.sum(loss_label_std, 1), torch.sum(loss_label_tea, 1))
            index_label = torch.zeros(tea_logits.size()[0], 2).to(device).scatter_(1, labels_, 1)
            teacher_conf = torch.sum(torch.mul(softmax(tea_logits), index_label), dim=1)
            mask = torch.mul(mask, teacher_conf)
            loss_logit = torch.mean(mask * torch.sum(loss_tea_std, 1))

            loss += self.config.alpha * loss_logit + loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
