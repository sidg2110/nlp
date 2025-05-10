# Taxonomy Expansion with Subgraph-Level Embeddings

This project focuses on the task of **taxonomy expansion**, which involves identifying appropriate parent nodes for new concepts to be integrated into an existing taxonomy. The goal is to ensure ontological consistency while scaling structured knowledge.

## Key Features

- Constructs subgraphs from seed taxonomies and candidate parent nodes.
- Uses a trained model to predict the most relevant subgraph (parent node) for a given query concept.
- Computes subgraph-level embeddings using node-level representations.
- Custom loss function based on cosine similarity to align predictions with ground truth.

## Components

- **Data Preparation**: Reads a seed taxonomy and generates subgraphs for training.
- **Embedding Module**: BERT and Word2Vec-based concept encoders.
- **Model**: Subgraph predictor that selects the most relevant parent from candidate subgraphs.
- **Training Script**: Includes loss computation and backpropagation over cosine similarity of graph-level embeddings.

## Usage

```bash
python scripts/train.py
