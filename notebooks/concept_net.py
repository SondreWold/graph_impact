
import networkx as nx
import itertools
import math
import random
import json
from tqdm import tqdm
import sys
import time
import timeit
import nltk
import json
import pickle
import os
import argparse
import numpy as np
import logging
from random import sample
import matplotlib.pyplot as plt

# Partially building on the work by https://github.com/wangpf3/Commonsense-Path-Generator/blob/main/learning-generator/sample_path_rw.py

def parse_args():
    parser = argparse.ArgumentParser(
        description="Find shortest path")

    parser.add_argument(
    "--data_dir",
    type=str,
    default="../data/conceptnet/",
    help="The folder of the concept_net graph",
    )
    
    parser.add_argument(
    "--output_dir",
    type=str,
    default="",
    help="The output folder",
    )

    args = parser.parse_args()

    return args


def load_kg(data_dir):
    print("Loading conceptnet...")
    data_path = os.path.join(data_dir, 'conceptnet_graph.nx')
    kg_full = nx.read_gpickle(data_path)
    '''
    kg_simple = nx.DiGraph()
    for u, v, data in kg_full.edges(data=True):
        kg_simple.add_edge(u, v)
    '''
    return kg_full

def load_vocab(data_dir):
    rel_path = os.path.join(data_dir, 'relation_vocab.pkl')
    ent_path = os.path.join(data_dir, 'entity_vocab.pkl')

    with open(rel_path, 'rb') as handle:
        rel_vocab = pickle.load(handle)

    with open(ent_path, 'rb') as handle:
        ent_vocab = pickle.load(handle)

    return rel_vocab['i2r'], rel_vocab['r2i'], ent_vocab['i2e'], ent_vocab['e2i']


class PathRetriever():
    def __init__(self, data_dir):
        logging.info("Loading full kg")
        self.kg = load_kg(data_dir)
        logging.info("Loading vocabs")
        self.i2r, self.r2i, self.i2e, self.e2i = load_vocab(data_dir)
        self.r2t = None
        with open('relation2text.json') as json_file:
            r2t = json.load(json_file)
            self.r2t =  {k.lower(): v for k, v in r2t.items()}


    def is_entity(self, entity):
        try:
            self.e2i[entity]
        except KeyError as e:
            return False
        return True

    def get_path(self, source, target):
        try:
            source_entity = self.e2i[source]
            target_entity = self.e2i[target]
        except KeyError as e:
            return -1
        
        try:
            path = nx.shortest_path(self.kg, source=source_entity, target=target_entity)
        except nx.exception.NetworkXNoPath as e:
            return -1
            


        rels = []
        for i in range(len(path)-1):
            local_path = self.kg[path[i]][path[i+1]]
            rel_list = list(set([local_path[item]['rel'] for item in local_path]))
            rel_candidate = rel_list[0]
                
            rels.append(rel_candidate)

        str_path = []
        path = [self.i2e[x] for x in path]
        rels = [self.i2r[x] for x in rels]
        final_path = []
        for i, p in enumerate(path[0:-1]):
            head = p
            pred = rels[i]
            tail = path[i+1]
            final_path.append([head,pred,tail])
            
        
        '''
        for i, rel in enumerate(rels):
            rel = rel.lower()
            has_u = False
            if "_" in rel:
                has_u = True
                rel = rel[1:]
            try:
                rel = self.r2t[rel]
            except KeyError as e:
                return -1

            #rel = "_" + rel # How to handle the reverse relation type?
            rels[i] = rel
        out = ""
        for i in range(len(path)):
            if i > 0:
                out += " "
            out += path[i] + " "
            if i < len(rels):
                out += rels[i]
        '''

        return final_path




def main(args):
    path_finder = PathRetriever(args.data_dir)
    #logging.info(path_finder.e2i["national_assembly"])
    path = path_finder.get_path("oslo", "snow")
    logging.info(path)


    '''
    logging.info("Loading full kg")
    kg_full = load_kg(args.data_dir)
    x = e2i["sklopapps"]
    print(x)
    
    logging.info(f'Num of entities: {len(i2e)}')
    logging.info(f'Num of relations: {len(i2r)}')
    logging.info(f"Number of nodes in kg: {len(kg_full.nodes())}")

    s, t = e2i["cave"], e2i["china"]

    #random_nodes = sample(list(kg_full.nodes()), 3)
    #logging.info(russia, country)
    path = nx.shortest_path(kg_full, source=s, target=t)
    
    rels = []
    for i in range(len(path)-1):
        local_path = kg_full[path[i]][path[i+1]]
        rel_list = list(set([local_path[item]['rel'] for item in local_path]))
        rels.append(rel_list[0])

    str_path = []
    path = [i2e[x] for x in path]
    rels = [i2r[x] for x in rels]

    draw_graph(path, rels)
    '''


def draw_graph(path, rels):
    edges = []
    for i in range(len(path)-1):
        one_hop = []
        one_hop.append(path[i])
        one_hop.append(path[i+1])
        edges.append(one_hop)

    print(edges)
    
    G = nx.Graph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    plt.figure()
    nx.draw(
        G, pos, edge_color='black', width=1, linewidths=1,
        node_size=500, node_color='pink', alpha=0.9,
        labels={node: node for node in G.nodes()}
    )
    edge_labels_dict = {}
    for i in range(len(edges)):
        v = rels[i]
        edge_labels_dict[(edges[i][0], edges[i][1])] = v
    
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels_dict,
        font_color='red'
    )
    plt.axis('off')
    plt.show()




if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    main(args)





