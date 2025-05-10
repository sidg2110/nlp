from models.utils.taxonomy_graph import TaxonomyGraph
from typing import List
import networkx as nx
import numpy as np
import random
from networkx.algorithms.similarity import graph_edit_distance

class SubgraphManager:
    def __init__(self, word_embeddings):
        self.word_embeddings = word_embeddings

    def get_subgraph_size(self, node: str, taxonomy: TaxonomyGraph) -> int:
        return int(10*np.exp(-0.2*taxonomy.out_degree(node)))
        pass

    def create_positive_subgraph(self, queries: List[str], gt_parents: List[str], taxonomy: TaxonomyGraph, exclude_node = []) -> List[TaxonomyGraph]:
        """
        For each query, returns a DiGraph
        """
        if len(exclude_node) == 0:
            exclude_node = ["NULLNULL"]*len(gt_parents)
        similarity_threshold = 0.1
        graphs = []
        for i in range(len(queries)):
            size = self.get_subgraph_size(gt_parents[i], taxonomy)
            node_queue = [gt_parents[i], "NULL"]
            node_similarity = [1.0, -1.0]
            node_parent = [gt_parents[i], "NULL"]
            reverse_bool = [0.0, -1.0]
            iter = 0
            queue_level = 1
            subgraph = TaxonomyGraph([], taxonomy.node_embeddings)
            node_set = set()
            node_set.add(str(gt_parents[i]))
            while size != 0 and iter < len(node_queue):
                print(size)
                if(node_queue[iter] == "NULL"):
                    if iter == len(node_queue)-1:
                        break
                    node_queue.append("NULL")
                    node_parent.append("NONE")
                    node_similarity.append(-1.0)
                    reverse_bool.append(-1.0)
                    queue_level += 1
                    iter += 1
                    continue
                if node_similarity[iter] >= queue_level*similarity_threshold and node_queue[iter] != exclude_node[i]:
                    if str(node_queue[iter]) != str(node_parent[iter]):
                        if reverse_bool[iter] == 0.0:
                            subgraph.add_edge(str(node_queue[iter]), str(node_parent[iter]))
                        else:
                            subgraph.add_edge(str(node_parent[iter]), str(node_queue[iter]))
                        node_set.add(str(node_queue[iter]))
                        size -= 1
                    for node in taxonomy.successors(node_queue[iter]):
                        if(str(node)) in node_set:
                            continue
                        node_queue.append(node)
                        temp1 = taxonomy.node_embeddings[str(node)]
                        temp2 = taxonomy.node_embeddings[str(node_queue[iter])]
                        node_similarity.append(np.dot(temp1, temp2)/(np.linalg.norm(temp1)*np.linalg.norm(temp2)))
                        node_parent.append(node_queue[iter])
                        reverse_bool.append(0.0)
                    for node in taxonomy.predecessors(node_queue[iter]):
                        if(str(node)) in node_set:
                            continue
                        node_queue.append(node)
                        temp1 = taxonomy.node_embeddings[str(node)]
                        temp2 = taxonomy.node_embeddings[str(node_queue[iter])]
                        node_similarity.append(np.dot(temp1, temp2)/(np.linalg.norm(temp1)*np.linalg.norm(temp2)))
                        node_parent.append(node_queue[iter])
                        reverse_bool.append(1.0)
                iter += 1
            graphs.append(subgraph)
        return graphs

    def create_negative_subgraph(self, queries: List[str], gt_parents: List[str], taxonomy: TaxonomyGraph) -> List[TaxonomyGraph]:
        exclude_node = []
        for i in range(len(gt_parents)):
            random_node = random.choice(list(taxonomy.nodes))
            exclude_node.append(random_node)
        return self.create_positive_subgraph(queries, gt_parents, taxonomy, exclude_node)
    
    def subgraph_similarity(self, taxonomy1: TaxonomyGraph, taxonomy2: TaxonomyGraph) -> float:
        alpha = 0.6
        nodes1 = set(taxonomy1.nodes)
        nodes2 = set(taxonomy2.nodes)
        jaccard_nodes = len(nodes1 & nodes2) / len(nodes1 | nodes2)
        edit_distance = graph_edit_distance(taxonomy1, taxonomy2)
        similarity_score = alpha * jaccard_nodes + (1-alpha) * (1 / (1 + edit_distance))
        return similarity_score
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTaxonomyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x= x.to(torch.float32)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x))  # logits

from torch.utils.data import Dataset, DataLoader
from numpy import dot
from numpy.linalg import norm

class TaxonomyDataset(Dataset):
    def __init__(self, data, seed_taxonomy: TaxonomyGraph, query_vectorizer, word_embeddings):
        self.data = data  # list of (query, taxonomy_graph)
        # self.seed_nodes = list(seed_taxonomy.nodes())
        self.seed_taxonomy = seed_taxonomy
        self.vectorizer = query_vectorizer
        self.word_embeddings = word_embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query, taxonomy_graph = self.data[idx]
        query_vec = self.vectorizer(query, self.word_embeddings,self.seed_taxonomy)

        # Make binary vector for the taxonomy nodes
        # label_vec = torch.zeros(len(self.seed_nodes))
        # present_nodes = set(taxonomy_graph.nodes())
        # for i, node in enumerate(self.seed_nodes):
            # if node in present_nodes:
            #     label_vec[i] = 1.0

        return query_vec

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

# Example: dummy vectorizer that turns string into random vector
def vectorizer(query, word_embeddings, seed_taxonomy):
    split_concept = query.split()
    embedding = np.zeros(300)
    for sub_concept in split_concept:
        if sub_concept in word_embeddings:
            embedding += word_embeddings[sub_concept]
    word_embed = np.concatenate([embedding, np.zeros(768 - 300)])
    final_vector = 0
    for i in range(len(seed_taxonomy.level_embed)):
        final_vector += cosine_similarity(word_embed, seed_taxonomy.level_embed[i])*seed_taxonomy.level_embed[i]
    return final_vector
