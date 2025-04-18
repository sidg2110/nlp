import torch
import torch.nn.functional as F
from torch.optim import Adam

class Trainer:
    def __init__(self, model, graph, device='cpu', lr=1e-3):
        """
        model: The model to be trained.
        graph: The graph (torch_geometric.data.Data) to be used for training.
        device: The device to run the model on ('cpu' or 'cuda').
        lr: Learning rate for optimizer.
        """
        self.model = model.to(device)
        self.graph = graph.to(device)
        self.device = device
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def train(self, query_embeddings, target_nodes, epochs=100):
        """
        Train the model for a specified number of epochs.

        query_embeddings: List of query embeddings for training.
        target_nodes: List of target node embeddings (ground truth).
        epochs: Number of training epochs.
        """
        for epoch in range(epochs):
            total_loss = 0
            for query_embedding, target_node in zip(query_embeddings, target_nodes):
                loss = self.train_step(query_embedding, target_node)
                total_loss += loss
            
            avg_loss = total_loss / len(query_embeddings)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def train_step(self, query_embedding, target_node):
        """
        Perform one training step (forward + backward).

        query_embedding: The query embedding (bs, d) to be processed by the model.
        target_node: The target node embedding for computing loss.
        :return: The computed loss.
        """
        self.model.train()
        self.optimizer.zero_grad()
        predicted_node = self.model(query_embedding, self.graph)
        loss = self.compute_loss(predicted_node, target_node)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def compute_loss(self, predicted_node, target_node):
        """
        Compute the loss between predicted node and target node.
        
        predicted_node: The node predicted by the model.
        target_node: The ground truth target node.
        :return: The computed loss (L2 loss in this case).
        """
        # TODO: Implement this
        # return F.mse_loss(predicted_node, target_node)
    
    # def evaluate(self, query_embedding, target_node):
    #     """
    #     Evaluate the model's performance on a specific query and target node.

    #     query_embedding: The query embedding (bs, d).
    #     target_node: The ground truth target node.
    #     :return: The computed similarity score (cosine similarity).
    #     """
    #     self.model.eval()  # Set model to evaluation mode
        
    #     predicted_node = self.model(query_embedding, self.graph)
        
    #     # Compute cosine similarity between the predicted node and target node
    #     similarity = F.cosine_similarity(predicted_node, target_node, dim=-1)
        
    #     return similarity.item()
