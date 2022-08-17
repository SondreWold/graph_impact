
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class SequenceModel(nn.Module):

    def __init__(self, model_name: str, device):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, return_dict=True, output_hidden_states=True, output_attentions=True)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(self.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        batch_size, n_choices, seq_length = input_ids.size()[0], input_ids.size()[1], input_ids.size()[-1]
        pass