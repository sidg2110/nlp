from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
import torch.nn.functional as F
import torch

class TaxoExpan(torch.nn.Module):
    def __init__(self, graph, in_dim, hidden_dim, out_dim):
        """
        graph_data: a torch_geometric.data.Data object
        """
        super().__init__()
        self.x = graph.x
        self.edges = graph.edge_index

        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, out_dim)

    def forward(self, query_embeddings):
        """
        query_embeddings: [batch_size, out_dim]
        """
        # Step 1: GCN forward
        x = F.relu(self.gcn1(self.x, self.edges))
        x = self.gcn2(x, self.edges)  # [num_nodes, out_dim]

        # Step 2–3: Compute avg of node and its neighbors
        row, col = self.edges
        all_neighbors = torch.cat([row, col], dim=0)
        all_centers = torch.cat([col, row], dim=0)

        self_loop_edges = torch.arange(x.size(0), device=x.device)
        all_centers = torch.cat([all_centers, self_loop_edges])
        all_neighbors = torch.cat([all_neighbors, self_loop_edges])

        avg_embed = scatter_mean(x[all_neighbors], all_centers, dim=0, dim_size=x.size(0))

        # Step 4: Similarity
        similarity = torch.matmul(query_embeddings, avg_embed.T)  # [B, N]

        # Step 5: Top-1 node
        best_nodes = torch.argmax(similarity, dim=1)

        return best_nodes, similarity
