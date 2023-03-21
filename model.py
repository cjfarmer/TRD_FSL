import torch
import torch.nn as nn
from transformers import ElectraForPreTraining


class ElectraForPrompt(nn.Module):
    def __init__(self, model_name, ):
        super().__init__()
        self.electara = ElectraForPreTraining.from_pretrained(model_name)

    # return the original possibilities of label description words, shape=(batch, seq, class)
    def forward(self, batch):
        # get the output of ELECTRA
        electra_output = self.electara(
                        input_ids=batch['input_ids'].cuda(), 
                        attention_mask=batch['input_mask'].cuda(), 
                        token_type_ids=batch['segment_ids'].cuda(),
                        labels=batch['full_labels'].cuda(),
                        )
        electra_logits = electra_output.logits # (batch, seq)
        loss = electra_output.loss # loss

        # Process the [original] probability of each label description word
        batch_size = electra_logits.shape[0]
        class_logits = electra_logits[batch['full_labels_mask_local'].bool()].reshape(batch_size, -1)
        # class_prob = 1 - torch.softmax(class_logits, -1)
        class_prob = 1 - torch.sigmoid(class_logits)

        return loss, class_prob






