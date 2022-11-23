import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class ExplaGraphs(Dataset):
    def __init__(self, model_name, split="train", use_graphs=False, use_pg=False, use_rg=False, generate_pg=False):
        print(f"Use graph explanations = {use_graphs}, use path generator = {use_pg}, use random generator = {use_rg}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        df = pd.read_csv(f"./data/explagraphs/{split}.tsv", sep="\t", header=0, index_col=0)
        self.premises = df["belief"].to_numpy()
        self.arguments = df["argument"].to_numpy()
        self.labels = df["label"].to_numpy()
        self.explanations = df["gold_graph"].to_numpy()
        self.generated_explanations = df["generated_graph"].to_numpy()
        self.random_explanations = df["retrieved_graph"]
        self.random_explanations = self.random_explanations.fillna('').to_numpy() #replace no path found with empty path
        self.r2t = None
        with open('relation2text.json') as json_file:
            r2t = json.load(json_file)
            self.r2t =  {k.lower(): v for k, v in r2t.items()}
        self.label_converter = {"counter": 0, "support": 1}
        self.label_inverter = {0: "counter", 1: "support"}
        self.skipped_examples = 0

        if use_pg == True:
            self.explanations = self.generated_explanations
        if use_rg == True:
            self.explanations = self.random_explanations
        if use_graphs == True:
            self.features = [prem + " " + self.tokenizer.sep_token + " " + arg + " " + self.tokenizer.sep_token + " " + self.clean_string(exp) for prem,arg,exp in zip(self.premises, self.arguments, self.explanations)]
        else:
            self.features = [prem + " " + self.tokenizer.sep_token + " " + arg for prem,arg in zip(self.premises, self.arguments)]

        encodings = self.tokenizer(self.features, truncation=True, padding=True)
        self.input_ids, self.attention_masks = encodings["input_ids"], encodings["attention_mask"]

        print(f"Skipped examples: {self.skipped_examples}")

    def get_decoded_sample(self, idx):
        return self.tokenizer.decode(self.input_ids[idx])

        
    def clean_string(self, x):
        res = eval(x)
        if isinstance(res, int):
            self.skipped_examples += 1
            return ""
        flat_list = [item.replace("_", "") for sublist in list(res) for item in sublist]
        out = []
        for ent in flat_list:
            if ent.lower() in self.r2t.keys():
                out.append(self.r2t[ent.lower()])
            else:
                out.append(ent)
        path = " ".join(out)
        return path
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return torch.LongTensor(self.input_ids[idx]), torch.BoolTensor(self.attention_masks[idx]), self.label_converter[self.labels[idx]]