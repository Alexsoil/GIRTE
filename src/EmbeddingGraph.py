import torch

class Vertex:

    neighbours = []

    def __init__(self, tensor: torch.tensor):
        self.tensor = tensor
        # Self-edge creation
        self.neighbours.append(self)
    
    def add_neighbour(self, vertex):
        self.neighbours.append(vertex)

class EmbeddingGraph:

    adjacency_list = []