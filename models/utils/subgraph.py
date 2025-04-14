from models.utils.taxonomy_graph import TaxonomyGraph
from typing import List
import networkx as nx


# Rough signature. Can change if needed
class SubgraphManager:
    def __init__(self):
        pass

    def create_subgraph_data(self, queries: List[str], gt_parents: List[str], taxonomy: TaxonomyGraph) -> List[nx.DiGraph]:
        """
        For each query, returns a DiGraph
        """
        pass

    def get_subgraph_size(self, node: str, taxonomy: TaxonomyGraph) -> int:
        pass
    