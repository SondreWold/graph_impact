import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from datasets import Dataset as Dset
import json
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class ExplaGraphs(Dataset):
    def __init__(self, model_name, split="train", use_graphs=False, use_pg=False, use_rg=False, generate_pg=False):
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

        print(f"Examples skipped due to no graph explanation found: {self.skipped_examples}")

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



class CopaDataset(Dataset):
    def __init__(self, model_name, split="train", use_graphs=False, use_pg=False, use_rg=False, generate_pg=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        options = [1, 2]
        df = pd.read_csv(f"./data/copa/{split}.tsv", sep="\t", header=0, index_col=0)
        dataset = Dset.from_pandas(df)
        self.graph_type = "gold_graph"
        self.use_graphs = use_graphs
        self.skipped_examples = 0


        with open('relation2text.json') as json_file:
            r2t = json.load(json_file)
            self.r2t =  {k.lower(): v for k, v in r2t.items()}

        if use_rg:
            self.graph_type= "retrieved_graph"
        if use_pg:
            self.graph_type= "generated_graph"

        self.dataset = self.preprocess_dataset(dataset)

        self.label_indexer = {v:k for k,v in enumerate(options)}
        self.label_inverter = {k:v for k,v in enumerate(options)}
        self.labels = [self.label_indexer[x] for x in self.dataset["most-plausible-alternative"]]
        #self.labels = self.dataset["most-plausible-alternative"]


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


    
    def get_decoded_sample(self, idx):
        return "Alt 1: " + self.tokenizer.decode(self.dataset[idx]['input_ids'][0]) + " Alt 2: " + self.tokenizer.decode(self.dataset[idx]['input_ids'][1]) + " Correct answer: " + str(self.labels[idx])


    def encode_batch(self, examples):
        all_encoded = {"input_ids": [], "attention_mask": []}
        # Iterate through all examples in this batch
        for premise, choice1, choice2, graph in zip(examples["p"], examples["a1"], examples["a2"], examples[self.graph_type]):
            if self.use_graphs:
                sentences_a = [premise + " " + self.clean_string(graph) for _ in range(2)]
            else:
                sentences_a = [premise for _ in range(2)]
            # Both answer choices are passed in an array according to the format needed for the multiple-choice prediction head
            sentences_b = [choice1, choice2]
            encoded = self.tokenizer(
                sentences_a,
                sentences_b,
                max_length=64,
                truncation=True,
                padding="max_length",
            )
            all_encoded["input_ids"].append(encoded["input_ids"])
            all_encoded["attention_mask"].append(encoded["attention_mask"])
        return all_encoded

    def preprocess_dataset(self, dataset):
        # Encode the input data
        dataset = dataset.map(self.encode_batch, batched=True)
        # The transformers model expects the target class column to be named "labels"
        return dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_ids = torch.LongTensor(self.dataset[idx]["input_ids"]).squeeze(0)
        attention_masks = torch.BoolTensor(self.dataset[idx]["attention_mask"]).squeeze(0)
        y = int(self.labels[idx])
        return input_ids, attention_masks, y
