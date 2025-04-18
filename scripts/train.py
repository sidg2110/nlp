import hydra
import torch
import numpy as np
import json, sys, os
from typing import List
from omegaconf import DictConfig

from models.utils.taxonomy_graph import TaxonomyGraph
from models.utils.embeddings import Embeddings
from models.utils.evaluator import Evaluator
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
    
    device = 'cuda'

    # For model v1
    # print(f"Model v1")
    # train_embeddings = embeddings_manager.get_term_embeddings(concepts)
    # model = TaxoExpan(train_embeddings, device=device)
    # eval_embeddings = embeddings_manager.get_term_embeddings(eval_concepts)
    # print(f"Predicting...")
    # predictions = model.predict_parents(eval_embeddings, concepts)

    # For model v2
    # print(f"Model v2")
    # print(f"Computing train embeddings...")
    # train_embeddings = embeddings_manager.get_concept_embeddings(concepts, definitions)
    # model = TaxoExpan(train_embeddings, device=device)
    # print(f"Computing eval embeddings...")
    # eval_embeddings = embeddings_manager.get_concept_embeddings(eval_concepts, definitions)
    # print(f"Predicting...")
    # predictions = model.predict_parents(eval_embeddings, concepts)

    # For model v3
    print(f"Model v3")
    print(f"Computing train embeddings...")
    train_embeddings = embeddings_manager.get_concept_embeddings(concepts, definitions)
    train_edges = []
    for edge in edges:
        train_edges.append([concept2index[edge[0]], concept2index[edge[1]]])
    train_edges = torch.tensor(train_edges)
    torch_taxonomy_graph = get_torch_taxonomy_graph(train_embeddings, train_edges)
    model = TaxoExpan(graph=torch_taxonomy_graph,
                      in_dim=train_embeddings.shape[-1],
                      hidden_dim=train_embeddings.shape[-1],
                      out_dim=train_embeddings.shape[-1]
                    )
    print(f"Computing eval embeddings...")
    eval_embeddings = embeddings_manager.get_concept_embeddings(eval_concepts, definitions)
    trainer = Trainer(model, torch_taxonomy_graph, device, lr=1e-3)
    trainer.train(eval_embeddings, eval_gt, epochs=1)

    # General to all models
    # print(f"Computing Metrics...")
    # accuracy = Evaluator.compute_accuracy(predictions, eval_gt)
    # wu_palmer = Evaluator.compute_wu_palmer(predictions, eval_gt, taxonomy_graph)
    # print(f"Accuracy: {accuracy}, Wu-Palmer: {wu_palmer}")

if __name__=="__main__":
    main()