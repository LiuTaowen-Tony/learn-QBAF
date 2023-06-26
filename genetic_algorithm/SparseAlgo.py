import torch
import torch.nn as nn
from typing import List
from abc import ABC, abstractmethod
from sparselinear import SparseLinear

def connectivity_list_random(in_size: int, out_size: int, num_connections: int):
    col = torch.randint(low=0, high=in_size, size=(num_connections,)).view(1, -1).long()
    row = torch.randint(low=0, high=out_size, size=(num_connections,)).view(1, -1).long()
    connections = torch.cat((row, col), dim=0)
    return connections

def mask_matrix_2_connectivity_list(adjacency_matrix: torch.Tensor):
    adjacency_matrix = adjacency_matrix.round()
    adjacency_matrix = adjacency_matrix.type(torch.long)
    assert adjacency_matrix.unique().tolist() in [[0], [1], [0, 1]]
    adjacency_list = [[], []]
    for i in range(adjacency_matrix.shape[0]):
        row_non_zero = torch.nonzero(adjacency_matrix[i]).squeeze(1).tolist()
        for j in row_non_zero:
            adjacency_list[1].append(i)
            adjacency_list[0].append(j)
    return torch.tensor(adjacency_list)

def adjacency_list_2_mask_matrix(adjacency_list: torch.Tensor, in_size: int, out_size: int):
    # convert type to int
    adjacency_list = adjacency_list.type(torch.int)
    adjacency_matrix = torch.zeros(in_size, out_size, dtype=torch.int)
    for row, col in (adjacency_list.T):
        assert 0 <= col < in_size
        assert 0 <= row < out_size
        adjacency_matrix[col, row] = 1
    return adjacency_matrix


class SparseABC(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_connectivities(self):
        pass

    @abstractmethod
    def get_mask_matrix_encoding(self) -> List[torch.Tensor]:
        """Encodes the structure of the graph as a bit string for genetic algorithm.

        The rows of the connectivity matrix are concatenated.
        """
        pass

    @abstractmethod
    def from_mask_matrix_encoding_to_connectivity(self):
        pass

    def find_sparsity(self) -> float:
        return 1 - self.total_num_conn() / self.total_max_num_conn()

    @abstractmethod
    def from_mask_matrix_encoding(self):
        pass

    @abstractmethod
    def total_max_num_conn(self):
        """
        Returns the MAXIMUM POSSIBLE number of connections in the network.
        """
        pass

    @abstractmethod
    def total_num_conn(self):
        """
        Returns the total number of connections in the network.
        """
        pass

    @abstractmethod
    def describe(self):
        pass

class SparseAlgo(SparseABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def reduced_num_conn(self):
        """
        remove not meaningful connections
        returns the number of connections after removing
        """
        pass

    @abstractmethod
    def random_connectivity_init(self):
        pass

class SparseLinearEnhanced(SparseLinear, SparseABC):
    def __init__(
            self,
            in_features: int, 
            out_features: int, 
            connectivity: torch.Tensor,
            bias=True,
        ):
        """
        :param connectivity: connectivity matrix of shape (in_features, out_features) 
        """
        try:
            super().__init__(
                in_features=in_features, 
                out_features=out_features, 
                connectivity=connectivity.to(torch.long))
        except:
            connectivity = torch.tensor([[0],[0]], dtype=torch.long)
            super().__init__(
                in_features=in_features,
                out_features=out_features,
                connectivity=connectivity.to(torch.long),
                bias=bias)
        self.mask_matrix = adjacency_list_2_mask_matrix(connectivity, in_features, out_features)

    def total_max_num_conn(self):
        """
        Returns the MAXIMUM POSSIBLE number of connections in the network.
        """
        return self.in_features * self.out_features
    
    def total_num_conn(self):
        """
        Returns the total number of connections in the network.
        """
        return self.connectivity.shape[1]

    def get_connectivities(self):
        return self.connectivity

    def get_mask_matrix_encoding(self):
        """Encodes the structure of the graph as a bit string for genetic algorithm.

        The rows of the connectivity matrix are concatenated.
        """
        return adjacency_list_2_mask_matrix(
            self.connectivity,
            self.in_features,
            self.out_features)

    def describe(self):
        from_to_weight = []
        bias = self.bias.tolist()
        for i in self.connectivity.T:
            to = i[0].item()
            from_ = i[1].item()
            weight = self.weight[to, from_].item()
            from_to_weight.append(
                {
                    "from": from_,
                    "to": to,
                    "weight": weight,
                    "bias": bias[to]
                })
        return from_to_weight

    
    @classmethod
    def from_mask_matrix_encoding_to_connectivity(
            cls, 
            *, 
            input_size:int,
            output_size:int,
            mask_matrix_encoding:torch.Tensor)->torch.Tensor:
        """Converts the genotype back to the phenotype.

        Transforms the bit string/chromosome representation back to the tensor representation.
        First, chromosome has to reshaped (unflattened), before the dense adjacency matrix has
        to be converted to sparse adjacency matrix of shape (m,n).
        """
        return mask_matrix_2_connectivity_list(
            mask_matrix_encoding.reshape(input_size, output_size)
        )

    @classmethod
    def from_mask_matrix_encoding(
        cls,
        *,
        input_size:int,
        output_size:int,
        mask_matrix_encoding:torch.Tensor) -> "SparseLinearEnhanced":
        """Converts the genotype back to the phenotype.
        Transforms the bit string/chromosome representation back to the tensor representation.
        First, chromosome has to reshaped (unflattened), before the dense adjacency matrix has
        to be converted to sparse adjacency matrix of shape (m,n).
        """
        return SparseLinearEnhanced(
            in_features=input_size,
            out_features=output_size,
            connectivity=mask_matrix_encoding
        ) 

def test_sparse_linear_enhanced():
    in_size = 10
    out_size = 20
    num_connections = 30
    connectivity = connectivity_list_random(in_size, out_size, num_connections)
    sparse_linear = SparseLinearEnhanced(
        in_features=in_size, 
        out_features=out_size, 
        connectivity=connectivity)
    assert sparse_linear.find_sparsity() == (1 - num_connections / (in_size * out_size))
    assert sparse_linear.mask_matrix.shape == (in_size, out_size)
    assert (sparse_linear.mask_matrix == adjacency_list_2_mask_matrix(connectivity, in_size, out_size)).all()
    assert (sparse_linear.mask_matrix == sparse_linear.get_mask_matrix_encoding()).all()
    assert (sparse_linear.connectivity == sparse_linear.get_connectivites()).all()
    assert (sparse_linear.connectivity == mask_matrix_2_connectivity_list(sparse_linear.mask_matrix)).all()
    assert (sparse_linear.connectivity == sparse_linear.from_mask_matrix_encoding_to_connectivity(
        input_size=in_size, output_size=out_size, mask_matrix_encoding=sparse_linear.get_mask_matrix_encoding())).all()
    assert (sparse_linear.from_mask_matrix_encoding(
        input_size=in_size, output_size=out_size, mask_matrix_encoding=sparse_linear.get_mask_matrix_encoding()).connectivity )