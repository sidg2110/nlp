from models.utils.taxonomy_graph import TaxonomyGraph
from typing import List
import networkx as nx
import numpy as np

# Rough signature. Can change if needed
class SubgraphManager:
    def __init__(self):
        pass

    def get_subgraph_size(self, node: str, taxonomy: TaxonomyGraph) -> int:
        return 10
        pass

    def create_subgraph_data(self, queries: List[str], gt_parents: List[str], taxonomy: TaxonomyGraph) -> List[TaxonomyGraph]:
        """
        For each query, returns a DiGraph
        """
        similarity_threshold = 0.1
        graphs = []
        for i in range(len(queries)):
            size = self.get_subgraph_size(queries[i], taxonomy)
            node_queue = [gt_parents[i], "NULL"]
            node_similarity = [1.0, -1.0]
            node_parent = [gt_parents[i], "NULL"]
            reverse_bool = [0.0, -1.0]
            iter = 0
            queue_level = 1
            subgraph = TaxonomyGraph([])
            node_set = set()
            node_set.add(str(gt_parents[i]))
            while size != 0 and iter < len(node_queue):
                if(node_queue[iter] == "NULL"):
                    node_queue.append("NULL")
                    node_parent.append("NONE")
                    node_similarity.append(-1.0)
                    reverse_bool.append(-1.0)
                    queue_level += 1
                    iter += 1
                    continue
                if node_similarity[iter] >= queue_level*similarity_threshold:
                    if str(node_queue[iter]) != str(node_parent[iter]):
                        if reverse_bool[iter] == 0.0:
                            subgraph.add_edge(str(node_queue[iter]), str(node_parent[iter]))
                        else:
                            subgraph.add_edge(str(node_parent[iter]), str(node_queue[iter]))
                        node_set.add(str(node_queue[iter]))
                        size -= 1
                    for node in taxonomy.successors(node_queue[iter]):
                        if(str(node)) in node_set:
                            continue
                        node_queue.append(node)
                        temp1 = taxonomy.node_embeddings[str(node)]
                        temp2 = taxonomy.node_embeddings[str(node_queue[iter])]
                        node_similarity.append(np.dot(temp1, temp2)/(np.linalg.norm(temp1)*np.linalg.norm(temp2)))
                        node_parent.append(node_queue[iter])
                        reverse_bool.append(0.0)
                    for node in taxonomy.predecessors(node_queue[iter]):
                        if(str(node)) in node_set:
                            continue
                        node_queue.append(node)
                        temp1 = taxonomy.node_embeddings[str(node)]
                        temp2 = taxonomy.node_embeddings[str(node_queue[iter])]
                        node_similarity.append(np.dot(temp1, temp2)/(np.linalg.norm(temp1)*np.linalg.norm(temp2)))
                        node_parent.append(node_queue[iter])
                        reverse_bool.append(1.0)
                iter += 1
            graphs.append(subgraph)
        return graphs
        pass
    