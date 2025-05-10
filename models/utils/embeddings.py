import torch
import numpy as np
from typing import List
from transformers import BertTokenizer, BertModel
from gensim.models import KeyedVectors
from tqdm import tqdm

WORD2VEC = 'C:/Users/ee121/Desktop/nlp/word2vec-google-news-300'

class Embeddings:
    def __init__(self, word_embedding="word2vec", tokenizer="bert", embedding_model="bert", device="cpu"):
        self.device = device
        self.word_embeddings = KeyedVectors.load_word2vec_format(WORD2VEC, binary=True)
        self.term_embedding_dim = 300
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.embedding_model = BertModel.from_pretrained("bert-base-uncased").to(device)
        self.embedding_model.eval()
        if self.device == "cuda":
            self.embedding_model.cuda()

    def get_term_embeddings(self, concepts: List[str]) -> torch.Tensor:
        """
        Returns word2vec embeddings of concepts
        """
        embeddings = []
        for concept in concepts:
            split_concept = concept.split()
            embedding = np.zeros(self.term_embedding_dim)
            for sub_concept in split_concept:
                if sub_concept in self.word_embeddings:
                    embedding += self.word_embeddings[sub_concept]
            embeddings.append(embedding)
        return torch.tensor(np.array(embeddings))

    # def get_concept_embeddings(self, concepts: List[str], definitions: dict[str, str]) -> torch.Tensor:
    def get_concept_embeddings(self, concepts: List[str], definitions: dict[str, str]) -> dict[str, np.array]:
        """
        Returns a tensor encoding the term and its definition using BERT
        """
        # concept_embeddings = []
        concept_embeddings = {}
        default_definition = "This concept has no definition"
        for concept in tqdm(concepts):
            definition = definitions.get(concept.lower(), default_definition)
            encoded_input = self.tokenizer(concept, definition, return_tensors='pt').to(self.device)
            with torch.no_grad():
                embedding = self.embedding_model(**encoded_input)
            # concept_embeddings.append(embedding["last_hidden_state"][0][0].detach().cpu().numpy())
            concept_embeddings[concept] = embedding["last_hidden_state"][0][0].detach().cpu().numpy()
        return concept_embeddings
        # return torch.tensor(np.array(concept_embeddings))