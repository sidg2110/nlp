import torch
import numpy as np
import json, sys, os
from typing import List

from models.utils.taxonomy_graph import TaxonomyGraph
from models.utils.embeddings import Embeddings
from models.utils.evaluator import Evaluator
from models.utils.subgraph import SimpleTaxonomyModel, TaxonomyDataset, SubgraphManager, cosine_similarity, vectorizer
from models.utils.subgraph import SubgraphManager
from models.model import TaxonomyExpander
from models.utils.subgraph import SimpleTaxonomyModel, TaxonomyDataset, SubgraphManager, cosine_similarity, vectorizer
from utils import process_train_data, strip_and_lowercase, process_definitions

from torch.utils.data import Dataset, DataLoader
from numpy import dot
from numpy.linalg import norm
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from gensim.models import KeyedVectors

WORD2VEC = 'C:/Users/ee121/Desktop/nlp/word2vec-google-news-300'

def main():
    word_embeddings = KeyedVectors.load_word2vec_format(WORD2VEC, binary=True)
    sys.path.append(r"C:/Users/ee121/Desktop/nlp")
    TRAIN_FILE = "C:/Users/ee121/Desktop/nlp/data/environment/environment_train.taxo"
    EVAL_FILE = "C:/Users/ee121/Desktop/nlp/data/environment/environment_eval.terms"
    EVAL_GT = "C:/Users/ee121/Desktop/nlp/data/environment/environment_eval.gt"
    DICT_FILE = "C:/Users/ee121/Desktop/nlp/data/environment/dic.json"

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

    
    concepts = []
    concept2index = {}
    for (parent, child) in edges:
        if parent not in concept2index:
            concept2index[parent] = len(concepts)
            concepts.append(parent)
        if child not in concept2index:
            concept2index[child] = len(concepts)
            concepts.append(child)
    
    embeddings_manager = Embeddings()
    concept_embeddings = embeddings_manager.get_concept_embeddings(concepts, definitions)
    # print(f"Edges: {len(edges)}")
    taxonomy_graph = TaxonomyGraph(edges, concept_embeddings)
    taxonomy_graph.level_embeddings()
    print(f"Taxonomy created")

    training_dataset = []
    for node in taxonomy_graph.nodes():
        if taxonomy_graph.out_degree(node) == 0:  # leaf
            training_dataset.append((node, next(taxonomy_graph.predecessors(node))))
    
    training_dataset = random.sample(training_dataset, k=int(0.1 * len(training_dataset)))

    queries = [query for query, _ in training_dataset]
    gt_parents = [parent for _, parent in training_dataset]
    Manager = SubgraphManager(word_embeddings)
    positive_subgraphs = Manager.create_positive_subgraph(queries, gt_parents, taxonomy_graph)
    negative_subgraphs = Manager.create_negative_subgraph(queries, gt_parents, taxonomy_graph)

    print("New dataset created")

    data = [(query, None) for query in queries]
    dataset = TaxonomyDataset(data, taxonomy_graph, vectorizer, word_embeddings)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    subgraph_predictor = SimpleTaxonomyModel(input_dim=768, hidden_dim=128, output_dim=len(taxonomy_graph.nodes))
    optimizer = torch.optim.Adam(subgraph_predictor.parameters(), lr=1e-3)

    print("subgraph predictor training started...")
    n_epochs = 10
    for epoch in range(n_epochs):
        iter = 0
        for query_vec in dataloader:
            preds = subgraph_predictor(query_vec)
            max_index = torch.argmax(preds).item()
            subgraph = Manager.create_positive_subgraph([concepts[max_index]], [concepts[max_index]], taxonomy_graph)[0]
            loss = torch.tensor(Manager.subgraph_similarity(subgraph, negative_subgraphs[iter]) - Manager.subgraph_similarity(subgraph, positive_subgraphs[iter]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter += 1
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    print("finetuning data")
    finetuning_data = []
    for query, parent in zip(queries, gt_parents):
        finetuning_data.append({
            "node": query,
            "parent": parent,
            "description": definitions[query]
        })

    model = TaxonomyExpander(subgraph_predictor)
    print(f"finetuning started")
    model.finetune(concepts, taxonomy_graph, finetuning_data)

    print("evaluating")
    predictions = []
    for query in eval_concepts:
        prompt = model.extract_subgraph_prompt(concepts, query, definitions[query])
        prediction = model.predict_parent(prompt)
        predictions.append(prediction)

    accuracy = Evaluator.compute_accuracy(predictions, eval_gt)
    wu_palmer = Evaluator.compute_wu_palmer(predictions, eval_gt, taxonomy_graph)
    print(f"Accuracy: {accuracy}, Wu-Palmer: {wu_palmer}")

if __name__=="__main__":
    main()