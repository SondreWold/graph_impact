from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
import numpy as np
import os
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from sentence_transformers import SentenceTransformer, util
data_dir = "./data/conceptnet/"


def load_vocab(data_dir):
    rel_path = os.path.join(data_dir, 'relation_vocab.pkl')
    ent_path = os.path.join(data_dir, 'entity_vocab.pkl')

    with open(rel_path, 'rb') as handle:
        rel_vocab = pickle.load(handle)

    with open(ent_path, 'rb') as handle:
        ent_vocab = pickle.load(handle)

    return rel_vocab['i2r'], rel_vocab['r2i'], ent_vocab['i2e'], ent_vocab['e2i']

def get_path(source, target):
        try:
            source_entity = e2i[source]
            target_entity = e2i[target]
        except KeyError as e:
            return -1    
        try:
            path = nx.shortest_path(kg_full, source=source_entity, target=target_entity)
        except nx.exception.NetworkXNoPath as e:
            return -1
        return path

if __name__ == '__main__':
    with open('./expla_matched.jsonl', 'r') as json_file:
        json_list = list(json_file)
    
    print("Loading conceptnet...")
    data_path = os.path.join("./data/conceptnet/", 'conceptnet_graph.nx')
    kg_full = nx.read_gpickle(data_path)
    i2r, r2i, i2e, e2i = load_vocab(data_dir)
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    k = 3

    no_paths = 0
    with open('expla_grounded.jsonl', 'a') as the_file:
        for counter, line in enumerate(tqdm(json_list)):
            NODES = []
            paths = []
            entity = json.loads(line)
            context = f"{entity['sent']} {entity['ans']}"
            #print(f" ORIGINAL CONTEXT: {context}")
            qc = entity['qc']
            ac = entity['ac']

            for q_ent in qc:
                NODES.append(q_ent)
                for a_ent in ac:
                    NODES.append(a_ent)
                    path = get_path(q_ent, a_ent)
                    if path != -1:
                        paths.append(path)
                        if len(path) <= 4:
                            for element in path:
                                NODES.append(i2e[element])

            rels = []
            for path in paths:
                rel_local = []
                for i in range(len(path)-1):
                    local_path = kg_full[path[i]][path[i+1]]
                    rel_list = list(set([local_path[item]['rel'] for item in local_path]))
                    rel_local.append(rel_list[0])
                rels.append(rel_local)

            NODES = list(set(NODES))
            
            triples = []
            finals = []
            for p, r in zip(paths, rels):
                s = ""
                fs = []
                for i in range(len(r)):
                    f = [i2e[p[i]], i2r[r[i]], i2e[p[i+1]]]
                    s += i2e[p[i]] + " " + i2r[r[i]] + " " + i2e[p[i+1]] + " "
                    fs.append(f)
                triples.append(s)
                finals.append(fs)
            
            if len(triples) == 0:
                no_paths += 1
                continue

            scores = []
            embeddings1 = model.encode(context, convert_to_tensor=True)
            for idx, triple in enumerate(triples):
                embeddings2 = model.encode(triple, convert_to_tensor=True)
                out = util.cos_sim(embeddings1, embeddings2)
                scores.append(out.item())

            #print(f"TOP TRIPLE WAS: {triples[np.argmax(scores)]}")
            the_file.write(json.dumps({'id': entity['id'], 'path': finals[np.argmax(scores)]}) +  "\n")