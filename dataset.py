import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
from transformers import GPT2Config, GPT2Tokenizer, GPT2Model
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# Define the generator model from Wang et al (2020)
class Generator(nn.Module):
    def __init__(self, gpt, config, max_len=31):
        super(Generator, self).__init__()
        self.gpt = gpt
        self.config = config
        self.max_len = max_len
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, inputs):
        # input: [batch, seq]
        context_len = inputs.size(1)
        generated = inputs
        next_token = inputs
        past = None
        with torch.no_grad():
            for step in range(self.max_len):
                outputs = self.gpt(next_token, past_key_values=past)
                hidden = outputs[0][:, -1]
                past = outputs[1]
                next_token_logits = self.lm_head(hidden)
                next_logits, next_token = next_token_logits.topk(k=1, dim=1)
                generated = torch.cat((generated, next_token), dim=1)
        return generated

class PathGenerator():
    def __init__(self):
        print("Load Path Generator..")
        lm_type = 'gpt2'
        config = GPT2Config.from_pretrained(lm_type)
        self.tokenizer = GPT2Tokenizer.from_pretrained(lm_type)
        self.tokenizer.add_tokens(['<PAD>'])
        self.tokenizer.add_tokens(['<SEP>'])
        self.tokenizer.add_tokens(['<END>'])
        gpt = GPT2Model.from_pretrained(lm_type)
        config.vocab_size = len(self.tokenizer)
        gpt.resize_token_embeddings(len(self.tokenizer))
        pretrain_generator_ckpt = "./pg/commonsense-path-generator.ckpt"
        self.generator = Generator(gpt, config)
        self.generator.load_state_dict(torch.load(pretrain_generator_ckpt, map_location=torch.device(device)), strict=False)

    def prepare_input(self, head_entity, tail_entity, input_len=16):
        head_entity = head_entity.replace('_', ' ')
        tail_entity = tail_entity.replace('_', ' ')
        input_token = tail_entity + '<SEP>' + head_entity
        input_id = self.tokenizer.encode(input_token, add_special_tokens=False)[:input_len]
        input_id += [self.tokenizer.convert_tokens_to_ids('<PAD>')] * (input_len - len(input_id))
        return torch.tensor([input_id], dtype=torch.long)

    def connect_entities(self, head_entity, tail_entity):
        gen_input = self.prepare_input(head_entity, tail_entity)
        gen_output = self.generator(gen_input)
        path = self.tokenizer.decode(gen_output[0].tolist(), skip_special_tokens=True)
        path = ' '.join(path.replace('<PAD>', '').split())
        return path[path.index('<SEP>')+6:]


class ExplaGraphs(Dataset):
    def __init__(self, model_name, split="train", use_graphs=False, use_pg=False, generate_pg=False):
        print(f"Use graph explanations = {use_graphs}, use path generator = {use_pg}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        df = pd.read_csv(f"./data/{split}.tsv", sep="\t", header=0, index_col=0)
        self.premises, self.arguments, self.labels, self.explanations, self.generated_explanations = df.to_numpy().T
        self.label_converter = {"counter": 0, "support": 1}
        self.label_inverter = {0: "counter", 1: "support"}

        ''' If you have the original data files and need to generate the PG paths. 
        if generate_pg:
            print("Generating paths...")
            #self.explanations = [self.get_path(x) for x in self.explanations]
            for i, exp in enumerate(tqdm(self.explanations)):
                self.explanations[i] = self.get_path(exp)
        else:
            self.explanations = [self.clean_string(x) for x in self.explanations]
        '''

        if use_pg == True:
            self.PG = PathGenerator()
            self.explanations = self.generated_explanations
            
        if use_graphs == True:
            features = [prem + " [SEP] " + arg + " [SEP] " + exp for prem,arg,exp in zip(self.premises, self.arguments, self.explanations)]
        else:
            features = [prem + " [SEP] " + arg for prem,arg in zip(self.premises, self.arguments)]

        encodings = self.tokenizer(features, truncation=True, padding=True)
        self.input_ids, self.attention_masks = encodings["input_ids"], encodings["attention_mask"]

    def get_path(self, x):
        original_explanation_graph = x.split(";")
        head = self.clean_string(original_explanation_graph[0])
        tail = self.clean_string(original_explanation_graph[-1])
        #print(f"Explanation was: {original_explanation_graph}, head is now {head}, tail is {tail}")
        path = self.PG.connect_entities(head, tail)
        return path

        
    def clean_string(self, x):
        x = x.replace(")(", ", ")
        return x.replace("(", "").replace(")","").replace(";", "")

    def use_path_generator(self, graph_to_replace):
        pass
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return torch.LongTensor(self.input_ids[idx]), torch.BoolTensor(self.attention_masks[idx]), self.label_converter[self.labels[idx]]