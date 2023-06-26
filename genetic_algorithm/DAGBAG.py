import torch
import torch.nn as nn
import sparselinear as sl
from typing import List
from genetic_algorithm.SparseAlgo import *

class DAGBAG(SparseABC):
    """
    QBAF with direct attack and support. The baseline model uses a layered QBAF. 
    The relationship between input and output arguments is indirect. 
    Input arguments affect output arguments through hidden arguments. 
    However, while analyzing the graphical structure of the baseline QBAF 
    classifier, it is often observed that an input argument solely influences 
    a single hidden argument, subsequently affecting only one output argument. 
    A more general acylic QBAF could capture the direct relation between 
    the input and output arguments. 

    Our extension adds a direct attack and support term

    o = \sigma(\sum wi hi + \sum wj xj + b)

    hi represents the strength value of i-th hidden argument,
    xj represents the strength value of j-th input argument. Now 
    both input and hidden arguments can affect output arguments' strength 
    value. This improves the expressiveness of the baseline model and 
    removes the restriction of a "layered" structure.
    """
    def __init__(self, input_size, hidden_size, output_size,
                 connections1, connections2, skip_connections):
        """
        :param input_size: number of input features
        :param hidden_size: number of hidden units
        :param output_size: number of output units
        :param connections1: list of tuples (to, from) for the first layer
        :param connections2: list of tuples (to, from) for the second layer
        :param skip_connections: list of tuples (to, from) for the skip connection
        """
        super().__init__()
        self.sparse_linear1 = SparseLinearEnhanced(input_size, hidden_size, connectivity=connections1)
        self.sparse_linear2 = SparseLinearEnhanced(hidden_size, output_size, connectivity=connections2)
        self.sparse_linear_skip = SparseLinearEnhanced(input_size, output_size, connectivity=skip_connections, bias=False)
        self.activation = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        tmp = self.sparse_linear_skip(x)
        x = self.sparse_linear1(x)
        x = self.activation(x)
        x = self.sparse_linear2(x)
        x = x + tmp
        return self.softmax(x)

    def get_connectivities(self):
        return [
            self.sparse_linear1.get_connectivities(),
            self.sparse_linear2.get_connectivities(),
            self.sparse_linear_skip.get_connectivities()
        ]

    def get_mask_matrix_encoding(self) -> List[torch.Tensor]:
        """Encodes the structure of the graph as a bit string for genetic algorithm.
        The rows of the connectivity matrix are concatenated.
        """
        return [
            self.sparse_linear1.get_mask_matrix_encoding(),
            self.sparse_linear2.get_mask_matrix_encoding(),
            self.sparse_linear_skip.get_mask_matrix_encoding()
        ]

    def describe(self):
        return {
            'sparse_linear1': self.sparse_linear1.describe(),
            'sparse_linear2': self.sparse_linear2.describe(),
            'sparse_linear_skip': self.sparse_linear_skip.describe()
        }

    def reduced_num_conn(self):
        """
        Remove redundant connections.
        return the number of connections after removing redundant connections.
        """
        sl1_before = self.sparse_linear1.connectivity
        sl2_before = self.sparse_linear2.connectivity
        sl3_before = self.sparse_linear_skip.connectivity

        sl2_mentioned_hidden = set()
        for i in sl2_before.T:
            to = i[0].item()
            from_ = i[1].item()
            sl2_mentioned_hidden.add(from_)

        sl1_after = set()
        print("sl2_mentioned_hidden")
        print(sl2_mentioned_hidden)
        sl1_metioned_hidden = set()
        for i in sl1_before.T:
            to = i[0].item()
            from_ = i[1].item()
            if to in sl2_mentioned_hidden:
                sl1_after.add((to, from_))
                sl1_metioned_hidden.add(to)
        print("sl1_after")
        print(sl1_after)
        print("sl1_metioned_hidden")
        print(sl1_metioned_hidden)
        sl2_after = set()
        for i in sl2_before.T:
            to = i[0].item()
            from_ = i[1].item()
            if from_ in sl1_metioned_hidden:
                sl2_after.add((to, from_))

        return len(sl1_after) + len(sl2_after) + sl3_before.shape[1]

    @classmethod
    def from_mask_matrix_encoding_to_connectivity(cls,
            *,
            input_size: int,
            hidden_size: int,
            output_size: int,
            mask_matrix_encoding: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Converts the genotype back to the phenotype.
        Transforms the bit string/chromosome representation back to the tensor representation.
        First, chromosome has to reshaped (unflattened), before the dense adjacency matrix has
        to be converted to sparse adjacency matrix of shape (m,n).
        """
        connectivity1 = SparseLinearEnhanced.from_mask_matrix_encoding_to_connectivity(
            input_size=input_size,
            output_size=hidden_size,
            mask_matrix_encoding=mask_matrix_encoding[0]
        )
        connectivity2 = SparseLinearEnhanced.from_mask_matrix_encoding_to_connectivity(
            input_size=hidden_size,
            output_size=output_size,
            mask_matrix_encoding=mask_matrix_encoding[1]
        )
        skip_connectivity = SparseLinearEnhanced.from_mask_matrix_encoding_to_connectivity(
            input_size=input_size,
            output_size=output_size,
            mask_matrix_encoding=mask_matrix_encoding[2]
        )
        return [connectivity1, connectivity2, skip_connectivity]
    

    @classmethod
    def random_connectivity_init(cls, params) -> "DAGBAG":
        """
        Initializes the DAGBAG with random connectivity. 
        """
        input_size = params["input_size"]
        hidden_size = params["hidden_size"]
        output_size = params["output_size"]
        n_connections_input_hidden = params["number_connections1"]
        n_connections_hidden_output = params["number_connections2"]
        n_connections_skip = params["number_connections_skip"]
        connectivity1 = connectivity_list_random(input_size, hidden_size, n_connections_input_hidden)
        connectivity2 = connectivity_list_random(hidden_size, output_size, n_connections_hidden_output)
        skip_connectivity = connectivity_list_random(input_size, output_size, n_connections_skip)
        return DAGBAG(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            connections1=connectivity1,
            connections2=connectivity2,
            skip_connections=skip_connectivity
        )


    def find_sparsity(self) -> float:
        return 1 - self.total_num_conn() / self.total_max_num_conn()

    @classmethod
    def from_mask_matrix_encoding(self, 
            params,
            mask_matrix_encoding: List[torch.Tensor]):
        """
        Converts the genotype back to the phenotype.
        Transforms the bit string/chromosome representation back to the tensor representation.
        First, chromosome has to reshaped (unflattened), before the dense adjacency matrix has
        to be converted to sparse adjacency matrix of shape (m,n).
        """
        input_size = params["input_size"]
        hidden_size = params["hidden_size"]
        output_size = params["output_size"]
        connectivities = self.from_mask_matrix_encoding_to_connectivity(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            mask_matrix_encoding=mask_matrix_encoding
        )
        return DAGBAG(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            connections1=connectivities[0],
            connections2=connectivities[1],
            skip_connections=connectivities[2]
        )

    def total_max_num_conn(self):
        """
        Returns the MAXIMUM POSSIBLE number of connections in the network.
        """
        return self.sparse_linear1.total_max_num_conn() + \
                self.sparse_linear2.total_max_num_conn() + \
                self.sparse_linear_skip.total_max_num_conn()

    def total_num_conn(self):
        """
        Returns the total number of connections in the network.
        """
        return self.sparse_linear1.total_num_conn() + \
                self.sparse_linear2.total_num_conn() + \
                self.sparse_linear_skip.total_num_conn()
