import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

class ExplaGraphs(Dataset):
    def __init__(self, model_name, split="train", use_graphs=False):
        print(f"Use graph explanations = {use_graphs}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        df = pd.read_csv(f"./data/{split}.tsv", sep="\t", header=0, index_col=0)
        premises, arguments, self.labels, explanations = df.to_numpy().T
        self.label_converter = {"counter": 0, "support": 1}
        self.label_inverter = {0: "counter", 1: "support"}
        explanations = [self.clean_string(x) for x in explanations]
        if use_graphs == True:
            features = [prem + " [SEP] " + arg + " [SEP] " + exp for prem,arg,exp in zip(premises, arguments, explanations)]
        else:
            features = [prem + " [SEP] " + arg for prem,arg in zip(premises, arguments)]

        encodings = self.tokenizer(features, truncation=True, padding=True)
        self.input_ids, self.attention_masks = encodings["input_ids"], encodings["attention_mask"]
        
    def clean_string(self, x):
        x = x.replace(")(", ", ")
        return x.replace("(", "").replace(")","").replace(";", "")
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return torch.LongTensor(self.input_ids[idx]), torch.BoolTensor(self.attention_masks[idx]), self.label_converter[self.labels[idx]]