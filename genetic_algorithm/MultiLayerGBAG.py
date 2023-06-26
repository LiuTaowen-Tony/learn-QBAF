from typing import List
import torch
from torch import nn
from genetic_algorithm.SparseAlgo import *
class MultiGBAG(SparseAlgo):
    """
    QBAF model with 2 hidden layers. One more hidden layer than the baseline model
    """
    def __init__(self,
                 input_size,
                hidden_size1,
                hidden_size2,
                output_size,
                connections1,
                connections2,
                connections3,):
        super().__init__()
        self.sparse_linear1 = SparseLinearEnhanced(input_size, hidden_size1, connectivity=connections1)
        self.sparse_linear2 = SparseLinearEnhanced(hidden_size1, hidden_size2, connectivity=connections2)
        self.sparse_linear3 = SparseLinearEnhanced(hidden_size2, output_size, connectivity=connections3)
        self.activation1 = nn.Sigmoid()
        self.activation2 = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.sparse_linear1(x)
        x = self.activation1(x)
        x = self.sparse_linear2(x)
        x = self.activation2(x)
        x = self.sparse_linear3(x)
        return self.softmax(x)

    def get_connectivities(self):
        return [
            self.sparse_linear1.get_connectivities(),
            self.sparse_linear2.get_connectivities(),
            self.sparse_linear3.get_connectivities(),
        ]
    
    def get_mask_matrix_encoding(self) -> List[torch.Tensor]:
        """Encodes the structure of the graph as a bit string for genetic algorithm.

        The rows of the connectivity matrix are concatenated.
        """
        return [
            self.sparse_linear1.get_mask_matrix_encoding(),
            self.sparse_linear2.get_mask_matrix_encoding(),
            self.sparse_linear3.get_mask_matrix_encoding(),
        ]
    
    @classmethod
    def random_connectivity_init(self, params):
        """
        Initializes the MULTIGBAG with random connectivity. 
        """
        input_size = params["input_size"]
        hidden_size1 = params["hidden_size1"]
        hidden_size2 = params["hidden_size2"]
        output_size = params["output_size"]
        n_connections1 = params["number_connections1"]
        n_connections2 = params["number_connections2"]
        n_connections3 = params["number_connections3"]

        connectivities1 = connectivity_list_random(in_size=input_size, out_size=hidden_size1, num_connections=n_connections1)
        connectivities2 = connectivity_list_random(in_size=hidden_size1, out_size=hidden_size2, num_connections=n_connections2)
        connectivities3 = connectivity_list_random(in_size=hidden_size2, out_size=output_size, num_connections=n_connections3)
        return MultiGBAG(
            input_size=input_size,
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            output_size=output_size,
            connections1=connectivities1,
            connections2=connectivities2,
            connections3=connectivities3,
        )

    
    @classmethod
    def from_mask_matrix_encoding_to_connectivity(cls, params, mask_matrix_encoding):
        """
        Converts the genotype back to the phenotype.
        Transforms the bit string/chromosome representation back to the tensor representation.
        First, chromosome has to reshaped (unflattened), before the dense adjacency matrix has
        to be converted to sparse adjacency matrix of shape (m,n).
        """
        input_size = params["input_size"]
        hidden_size1 = params["hidden_size1"]
        hidden_size2 = params["hidden_size2"]
        output_size = params["output_size"]

        connectivities1 = SparseLinearEnhanced.from_mask_matrix_encoding_to_connectivity(
            input_size=input_size,
            output_size=hidden_size1,
            mask_matrix_encoding=mask_matrix_encoding[0],
        )
        connectivities2 = SparseLinearEnhanced.from_mask_matrix_encoding_to_connectivity(
            input_size=hidden_size1,
            output_size=hidden_size2,
            mask_matrix_encoding=mask_matrix_encoding[1],
        )
        connectivities3 = SparseLinearEnhanced.from_mask_matrix_encoding_to_connectivity(
            input_size=hidden_size2,
            output_size=output_size,
            mask_matrix_encoding=mask_matrix_encoding[2],
        )
        return [
            connectivities1,
            connectivities2,
            connectivities3,
        ]

    @classmethod
    def from_mask_matrix_encoding(cls, params, mask_matrix_encoding):
        """
        Converts the genotype back to the phenotype.
        Transforms the bit string/chromosome representation back to the tensor representation.
        First, chromosome has to reshaped (unflattened), before the dense adjacency matrix has
        to be converted to sparse adjacency matrix of shape (m,n).
        """
        connectivities = MultiGBAG.from_mask_matrix_encoding_to_connectivity(params, mask_matrix_encoding)
        input_size = params["input_size"]
        hidden_size1 = params["hidden_size1"]
        hidden_size2 = params["hidden_size2"]
        output_size = params["output_size"]

        return MultiGBAG(
            input_size=input_size,
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            output_size=output_size,
            connections1=connectivities[0],
            connections2=connectivities[1],
            connections3=connectivities[2],
        )

    def describe(self):
        return {
            'sparse_linear1': self.sparse_linear1.describe(),
            'sparse_linear2': self.sparse_linear2.describe(),
            'sparse_linear3': self.sparse_linear3.describe(),
        }
    
    def reduced_num_conn(self):
        """
        remove not meaningful connections
        returns the number of connections after removing
        """
        sl1_before = self.sparse_linear1.connectivity
        sl2_before = self.sparse_linear2.connectivity
        sl3_before = self.sparse_linear3.connectivity

        sl3_metioned_hidden2 = set()
        for i in sl3_before.T:
            to = i[0].item()
            from_ = i[1].item()
            sl3_metioned_hidden2.add(from_)

        sl2_after1 = set()
        sl2_metioned_hidden1 = set()
        for i in sl2_before.T:
            to = i[0].item()
            from_ = i[1].item()
            if to in sl3_metioned_hidden2: 
                sl2_metioned_hidden1.add(from_)
                sl2_after1.add((to, from_))

        sl1_mentioned_hidden1 = set()
        sl1_after = set()
        for i in sl1_before.T:
            to = i[0].item()
            from_ = i[1].item()
            if to in sl2_metioned_hidden1:
                sl1_mentioned_hidden1.add(to)
                sl1_after.add((to, from_))

        sl2_after2 = set()
        sl2_mentioned_hidden2 = set()
        for (to, from_) in sl2_after1:
            if from_ in sl1_mentioned_hidden1:
                sl2_after2.add((to, from_))
                sl2_mentioned_hidden2.add(to)

        sl3_after = set()
        for i in sl3_before.T:
            to = i[0].item()
            from_ = i[1].item()
            if from_ in sl2_mentioned_hidden2:
                sl3_after.add((to, from_))
        
        return len(sl1_after) + len(sl2_after2) + len(sl3_after)
    
    def total_num_conn(self):
        """
        Returns the total number of connections in the network.
        """
        return (self.sparse_linear1.total_num_conn() + 
                self.sparse_linear2.total_num_conn() + 
                self.sparse_linear3.total_num_conn())
    
    def total_max_num_conn(self):
        """
        Returns the MAXIMUM POSSIBLE number of connections in the network.
        """
        return (self.sparse_linear1.total_max_num_conn() +
                self.sparse_linear2.total_max_num_conn() +
                self.sparse_linear3.total_max_num_conn())
    

    
    