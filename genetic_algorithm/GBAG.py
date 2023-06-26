from typing import List
import torch
import torch.nn as nn
from genetic_algorithm.SparseAlgo import *
from genetic_algorithm.utils.graph_visualizations import remove_connections

import sparselinear as sl

def create_random_connectivity_matrix(in_size, out_size, num_connections):
    """Creates a random connectivity matrix."""
    col = torch.randint(low=0, high=in_size, size=(num_connections,)).view(1, -1).long()
    row = torch.randint(low=0, high=out_size, size=(num_connections,)).view(1, -1).long()
    connections = torch.cat((row, col), dim=0)
    return connections


def flatten_connectivity_matrix(connectivity_matrix, in_dim, out_dim) -> torch.Tensor:
    """Flattens the connectivity matrix (concatenation of the rows)."""
    nnz = connectivity_matrix.shape[1]
    adj = torch.sparse.FloatTensor(
        connectivity_matrix, torch.ones(nnz), torch.Size([out_dim, in_dim])
    ).to_dense()
    adj = adj.type(torch.DoubleTensor)
    adj = torch.where(adj <= 1, adj, 1.0)  # remove redundant connections
    return torch.flatten(adj)

def decode(chromosome, shape):
    """Converts the genotype back to the phenotype.
    Transforms the bit string/chromosome representation back to the tensor representation.
    First, chromosome has to reshaped (unflattened), before the dense adjacency matrix has
    to be converted to sparse adjacency matrix of shape (m,n).
    """
    chrom = torch.reshape(chromosome, shape)
    assert chrom.dim() == 2
    ind = chrom.nonzero(as_tuple=False).t().contiguous()
    # if no connections, create random one
    if ind.size()[1] == 0:
        ind = torch.zeros(size=(2, 1))
        ind[0] = torch.randint(low=0, high=shape[0] - 1, size=(1,))
        ind[1] = torch.randint(low=0, high=shape[1] - 1, size=(1,))
        ind = torch.tensor(ind, dtype=torch.long)
    return torch.stack((ind[1], ind[0]), dim=0)

class GBAG(SparseAlgo):
    """
    Implementation of a Gradual Bipolar Argumentation Graph / edge-weighted QBAF as a sparse multi-layer perceptron
    """
    def __init__(self, input_size, hidden_size, output_size, connections1, connections2):
        super().__init__()

        self.sparse_linear1 = SparseLinearEnhanced(input_size, hidden_size, connectivity=connections1)
        self.activation1 = nn.Sigmoid()
        self.sparse_linear2 = SparseLinearEnhanced(hidden_size, output_size, connectivity=connections2)
        self.output_layer = nn.Softmax()

    def forward(self, x):
        x = self.sparse_linear1.forward(x)
        x = self.activation1(x)
        x = self.sparse_linear2.forward(x)
        return self.output_layer(x)

    @classmethod
    def random_connectivity_init(cls, params):
        """
        Initializes the GBAG with random connectivity matrices
        """
        input_size = params['input_size']
        hidden_size = params['hidden_size']
        output_size = params['output_size']
        num_connections1 = params['number_connections1']
        num_connections2 = params['number_connections2']
        connections1 = create_random_connectivity_matrix(input_size, hidden_size, num_connections1)
        connections2 = create_random_connectivity_matrix(hidden_size, output_size, num_connections2)
        return GBAG(input_size, hidden_size, output_size, connections1, connections2)

    def get_mask_matrix_encoding(self) -> List[torch.Tensor]:
        """Encodes the structure of the graph as a bit string for genetic algorithm.

        The rows of the connectivity matrix are concatenated.
        """
        connectivity1 = self.sparse_linear1.connectivity
        connectivity2 = self.sparse_linear2.connectivity
        matrix1 = flatten_connectivity_matrix(connectivity1, 
                    self.sparse_linear1.in_features, 
                    self.sparse_linear1.out_features)
        matrix2 = flatten_connectivity_matrix(connectivity2, 
                    self.sparse_linear2.in_features, 
                    self.sparse_linear2.out_features)
        return [matrix1, matrix2]

    @classmethod
    def from_mask_matrix_encoding(cls, params, mask_matrix_encoding):
        """
        Converts the genotype back to the phenotype.
        Transforms the bit string/chromosome representation back to the tensor representation.
        First, chromosome has to reshaped (unflattened), before the dense adjacency matrix has
        to be converted to sparse adjacency matrix of shape (m,n).
        """
        input_size = params['input_size']
        hidden_size = params['hidden_size']
        output_size = params['output_size']
        matrix1 = mask_matrix_encoding[0]
        matrix2 = mask_matrix_encoding[1]
        connectivity1 = decode(matrix1, (input_size, hidden_size))
        connectivity2 = decode(matrix2, (hidden_size, output_size))
        return GBAG(input_size, hidden_size, output_size, connectivity1, connectivity2)

    def total_num_conn(self):
        """
        remove not meaningful connections
        returns the number of connections after removing
        """
        return self.sparse_linear1.connectivity.shape[1] + self.sparse_linear2.connectivity.shape[1]

    def total_max_num_conn(self):
        """
        Returns the MAXIMUM POSSIBLE number of connections in the network.
        """
        return self.sparse_linear1.in_features * self.sparse_linear1.out_features + \
                self.sparse_linear2.in_features * self.sparse_linear2.out_features

    def reduced_num_conn(self):
        """
        remove not meaningful connections
        returns the number of connections after removing
        """        
        classifier, ind = remove_connections(self)
        return classifier.total_num_conn()

    def describe(self):
        return {
            "sparse_linear1": self.sparse_linear1.describe(),
            "sparse_linear2": self.sparse_linear2.describe(),
        }



