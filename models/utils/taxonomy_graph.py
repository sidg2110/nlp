import networkx as nx
from typing import List
import matplotlib.pyplot as plt
import numpy as np

class TaxonomyGraph(nx.DiGraph):
    def __init__(self, edges, node_embeddings={}):
        super().__init__(edges)
        self.root = None
        self.node_embeddings = node_embeddings
        self.level_embed = None
        self._remove_transitive_edges()
        self._set_root()

    def _remove_transitive_edges(self):
        """
        Removes all transitive edges from the graph.
        If node A has two predecessors B and C
        such that, B is reachable from C
        then, B to A edge is redundant and is removed 
        """
        bad_edges = []
        for node in self.nodes:
            if len(self.pred[node]) <= 1:
                continue
            for predecessor_1 in self.predecessors(node):
                for predecessor_2 in self.predecessors(node):
                    if predecessor_2 != predecessor_1:
                        if nx.has_path(self, predecessor_1, predecessor_2):
                            bad_edges.append((predecessor_1, node))
        self.remove_edges_from(bad_edges)

    def _set_root(self):
        """
        Sets the root of the taxonomy to the concept with no predecessor
        """
        # print(f"Nodes: {len(self.nodes)}")
        for node in self.nodes:
            if self.in_degree(node) == 0:
                self.root = node
                break
        # assert self.root != None

    def get_lca(self, node_1: str, node_2: str) -> str:
        """
        Returns the lowest common ancestor of concepts represented by node_1 and node_2
        """
        return nx.lowest_common_ancestor(self, node_1, node_2)

    def get_shortest_path(self, node_1: str, node_2: str) -> int:
        """
        Returns the length of shortest path node_1 -> node_2
        Assumes unweighted di-graph 
        """
        return nx.shortest_path_length(self, node_1, node_2)
    
    def level_embeddings(self):
        root = str(self.root)
        # print(f"Root: {root}")
        levels = nx.single_source_shortest_path_length(self, root)
        max_level = 0
        for node, level in levels.items():
        for node, level in levels.items():
            max_level = max(max_level, level)
        final_embed = [np.zeros(768)]*(max_level+1)
        for node, level in levels.items():
            final_embed[level] = final_embed[level]+self.node_embeddings[str(node)]
        # for i in range(len(final_embed)):
        #     final_embed[i] = ...
        self.level_embed = final_embed

    # def depth_embeddings(self):
    #     root = str(self.root)
    #     self.depth_embed = ()
    #     for node in self.nodes:
    #         if self.out_degree(node) != 0:
    #             continue
    #         path = nx.shortest_path(self, source=root, target=str(node))
    #         self.depth_embed[str(node)] = np.array([])
    #         for path_node in path:
    #             self.depth_embed[str(path_node)] = np.concat(self.depth_embed[str(path_node)], self.node_embeddings[str(path_node)])
    #         self.depth_embed[str(node)] = ...

