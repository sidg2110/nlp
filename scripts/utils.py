import torch
from torch_geometric.data import Data
from typing import List
from importlib import import_module

def process_train_data(train_data: List[str]) -> List[tuple[str, str]]:
    """
    Returns a list of (parent, child) tuples
    """
    taxonomy_list = set()
    for relationship in train_data:
        nodes = relationship.split('\t')
        nodes[0] = nodes[0].strip().lower()
        nodes[1] = nodes[1].strip().lower()
        taxonomy_list.add((nodes[0], nodes[1]))
    return list(taxonomy_list)

def strip_and_lowercase(data: List[str]) -> List[str]:
    stripped_data = []
    for element in data:
        stripped_data.append(element.strip().lower())
    return stripped_data

def process_definitions(definitions: dict[str, List[str]]) -> dict[str, str]:
    processed_definitions = {}
    for key, value in definitions.items():
        processed_definitions[key.strip().lower()] = value[0].strip().lower()
    return processed_definitions

def get_torch_taxonomy_graph(node_embeddings: torch.Tensor, edges: torch.Tensor):
    return Data(x=node_embeddings, edge_index=edges)


def get_model(module_path):
    module_path = module_path
    module = import_module(module_path)
    taxonomy = getattr(module, "Taxonomy")
    taxo_expan = getattr(module, "TaxoExpan")
    return taxonomy, taxo_expan