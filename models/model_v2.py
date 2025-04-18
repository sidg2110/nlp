import networkx as nx
import torch
import torch.nn.functional as F
from typing import List

class TaxoExpan():
    """
    Returns the concept with maximum cosine similarity with the query concept
    """

    def __init__(self, concept_embeddings: torch.Tensor, device='cpu'):
        self.device = device
        self.concept_embeddings = concept_embeddings.to(device)
        
    def predict_parents(self, eval_concepts: torch.Tensor, concepts: List[str]) -> List[str]:
        """
        eval_concept: (number_of_eval_samples, embedding_dim)
        concepts: List of concepts in seed taxonomy. concepts[i] corresponds to embedding self.term_embeddings[i]
        """
        eval_concepts = eval_concepts.to(self.device)
        # Convert to unit vectors so that cosine similarity is equivalent to matrix multiplication later on
        normalized_terms = F.normalize(self.concept_embeddings, p=2, dim=1)
        normalized_batch = F.normalize(eval_concepts, p=2, dim=1)

        # Compute cosine similarity: (N, D) @ (D, M) → (N, M)
        similarity = torch.matmul(normalized_batch, normalized_terms.T)
        parent_indices = torch.argmax(similarity, dim=1)        
        return [concepts[i] for i in parent_indices.tolist()]