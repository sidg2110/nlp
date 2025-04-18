from typing import List
from models.utils.taxonomy_graph import TaxonomyGraph

class Evaluator:
    def __init__(self):
        pass
    
    @staticmethod
    def compute_accuracy(predictions: List, ground_truths: List) -> float:
        """
        Returns the accuracy of predictions
        """
        assert len(predictions) == len(ground_truths), "Length of predictions list and ground truth lists should be same"
        N = len(predictions)
        accuracy = sum([1.0 if prediction == ground_truth else 0.0 for prediction, ground_truth in zip(predictions, ground_truths)])
        accuracy = accuracy/N
        assert accuracy <= 1, "Accuracy cannot be greater than 1"
        return accuracy
    
    @staticmethod
    def compute_wu_palmer(predictions: List, ground_truths: List, taxonomy: TaxonomyGraph) -> float:
        """
        Returns the Wu-Palmer Metric of predictions
        """
        assert len(predictions) == len(ground_truths), "Length of predictions list and ground truth lists should be same"
        N = len(predictions)
        wu_palmer = 0.
        for prediction, ground_truth in zip(predictions, ground_truths):
            wu_palmer += Evaluator._compute_wu_palmer_per_sample(prediction, ground_truth, taxonomy)
        wu_palmer = wu_palmer/N
        assert wu_palmer <= 1, "Wu Palmer cannot be greater than 1"
        return wu_palmer
    
    @staticmethod
    def _compute_wu_palmer_per_sample(prediction: str, ground_truth: str, taxonomy: TaxonomyGraph) -> float:
        if prediction == ground_truth == taxonomy.root:
            return 1.0
        lca = taxonomy.get_lca(prediction, ground_truth)
        depth_lca = taxonomy.get_shortest_path(taxonomy.root, lca)
        depth_prediction = taxonomy.get_shortest_path(taxonomy.root, prediction)
        depth_ground_truth = taxonomy.get_shortest_path(taxonomy.root, ground_truth)
        wu_palmer = float(2 * depth_lca) / float(depth_prediction + depth_ground_truth)
        assert wu_palmer <= 1, "Wu Palmer cannot be greater than 1"
        return wu_palmer