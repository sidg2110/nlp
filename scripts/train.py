import hydra
import torch
import numpy as np
import json, sys, os
from typing import List
from omegaconf import DictConfig

from models.utils.taxonomy_graph import TaxonomyGraph
from models.utils.embeddings import Embeddings
from models.utils.evaluator import Evaluator
from models.utils.subgraph import SimpleTaxonomyModel, TaxonomyDataset, SubgraphManager, cosine_similarity, vectorizer
# from models.model_v1 import TaxoExpan
# from models.model_v2 import TaxoExpan
from models.model_v3 import TaxoExpan
from trainer import Trainer
from utils import process_train_data, strip_and_lowercase, process_definitions, get_torch_taxonomy_graph


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(args: DictConfig):
    TRAIN_FILE = os.path.join(args.base_path, f"data/flame/{args.dataset}/{args.dataset}_train.taxo")
    EVAL_FILE = os.path.join(args.base_path, f"data/flame/{args.dataset}/{args.dataset}_eval.terms")
    EVAL_GT = os.path.join(args.base_path, f"data/flame/{args.dataset}/{args.dataset}_eval.gt")
    DICT_FILE = os.path.join(args.base_path, f"data/flame/{args.dataset}/dic.json")

    with open(TRAIN_FILE, 'r') as file:
        edges = file.readlines()
    with open(EVAL_FILE, 'r') as file:
        eval_concepts = file.readlines()
    with open(EVAL_GT, 'r') as file:
        eval_gt = file.readlines()
    with open(DICT_FILE, 'r') as file:
        definitions = json.load(file)
    
    definitions = process_definitions(definitions)

    edges = process_train_data(edges)
    eval_concepts = strip_and_lowercase(eval_concepts)
    eval_gt = strip_and_lowercase(eval_gt)

    print(f"Pre-processing complete")

    embeddings_manager = Embeddings()
    taxonomy_graph = TaxonomyGraph(edges)
    
    print(f"Taxonomy created")

    concepts = []
    concept2index = {}
    for (parent, child) in edges:
        if parent not in concept2index:
            concept2index[parent] = len(concepts)
            concepts.append(parent)
        if child not in concept2index:
            concept2index[child] = len(concepts)
            concepts.append(child)
    
    from torch.utils.data import Dataset, DataLoader
    from numpy import dot
    from numpy.linalg import norm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    queries = ...
    gt_parents = ...
    seed_taxonomy = taxonomy_graph
    Manager = SubgraphManager()
    positive_subgraphs = Manager.create_positive_subgraph(queries, gt_parents, seed_taxonomy)
    negative_subgraphs = Manager.create_negative_subgraph(queries, gt_parents, seed_taxonomy)

    data = ...
    dataset = TaxonomyDataset(data, seed_taxonomy.nodes, vectorizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = SimpleTaxonomyModel(input_dim=300, hidden_dim=128, output_dim=len(seed_taxonomy.nodes))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(10):
        iter = 0
        for query_vec, label_vec in dataloader:
            preds = model(query_vec)
            max_index = np.argmax(preds)
            subgraph = Manager.create_positive_subgraph([concepts[max_index]], [concepts[max_index]], seed_taxonomy)
            loss = Manager.subgraph_similarity(subgraph, negative_subgraphs[iter])-Manager.subgraph_similarity(subgraph, positive_subgraphs[iter])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter += 1

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

if __name__=="__main__":
    main()