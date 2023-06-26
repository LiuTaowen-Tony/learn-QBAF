import torch
import torch.nn as nn
from typing import List
from genetic_algorithm.SparseAlgo import *

class ProdCollectiveAS(SparseABC):
    """
    This is the t-norm operator for the joint attack and support QBAF

    The t-norm operator is the product of the selected inputs.
    T(Ri) = r1 * r2 * ... * rn
    """
    def __init__(self, *, input_size: int, output_size: int, connectivities: torch.Tensor):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.connectivities = connectivities
        _, self.n_connections = connectivities.shape
        self.mask_matrix = adjacency_list_2_mask_matrix(connectivities, input_size, output_size).T.to(torch.float32).unsqueeze(0).contiguous()
        self.weight = nn.Parameter(torch.rand(size=(self.output_size,)))

    @classmethod
    def from_mask_matrix_encoding(cls, *,  input_size: int, output_size: int, mask_matrix_encoding: torch.Tensor):
        """
        Converts the genotype back to the phenotype.
        Transforms the bit string/chromosome representation back to the tensor representation.
        First, chromosome has to reshaped (unflattened), before the dense adjacency matrix has
        to be converted to sparse adjacency matrix of shape (m,n).
        """
        connectivity = ProdCollectiveAS.from_mask_matrix_encoding_to_connectivity(
            input_size=input_size,
            output_size=output_size,
            mask_matrix_encoding=mask_matrix_encoding)
        return cls(
            connectivity=connectivity, 
            input_size=input_size, 
            joint_feature_size=output_size)

    @classmethod
    def from_mask_matrix_encoding_to_connectivity(
            cls, 
            *,
            input_size:int,
            output_size:int,
            mask_matrix_encoding:torch.Tensor) -> torch.Tensor:
        """
        Converts the genotype back to the phenotype.
        Transforms the bit string/chromosome representation back to the tensor representation.
        First, chromosome has to reshaped (unflattened), before the dense adjacency matrix has
        to be converted to sparse adjacency matrix of shape (m,n).
        """
        return mask_matrix_2_connectivity_list(
            mask_matrix_encoding.reshape(input_size, output_size)
        ) 
    
    def total_max_num_conn(self):
        """
        Returns the MAXIMUM POSSIBLE number of connections in the network.
        """
        return self.input_size * self.output_size
    
    def total_num_conn(self):
        """
        Returns the total number of connections in the network.
        """
        return self.output_size

    def get_connectivities(self):
        return self.connectivities

    def get_mask_matrix_encoding(self) -> torch.Tensor:
        """Encodes the structure of the graph as a bit string for genetic algorithm.

        The rows of the connectivity matrix are concatenated.
        """
        return self.mask_matrix.squeeze(0).T.flatten().contiguous()

    def describe(self, linear:SparseLinearEnhanced):
        connectivity = self.get_connectivities()
        result = []
        for i in connectivity.T:
            to = i[0].item()
            from_ = i[1].item()
            weight = self.weight[to].item()
            bias = linear.bias.tolist()[to].item()
            result.append(
                {
                    "from": from_,
                    "to": to,
                    "bias": bias,
                    "weight": weight
                }
            )
        return result

    def forward(self, x: torch.Tensor):
        # map to log space, addition becomes multiplication
        # map back to normal space
        x = torch.log(x)
        x = x.clamp(-50, 50)
        x = x.unsqueeze(-1)
        x = self.mask_matrix @ x
        x = x.squeeze(-1)
        x = torch.exp(x)
        return x * self.weight

def test_prod_collective_as():
    input_size = 3
    output_size = 2
    connectivities = torch.Tensor([[1, 0, 0], [1, 1, 0]])
    prod_collective_as = ProdCollectiveAS(
        input_size=input_size, 
        output_size=output_size, 
        connectivities=connectivities)
    prod_collective_as.weight = nn.Parameter(torch.tensor([1., 1., 1.]))
    x = torch.arange(30, dtype=torch.float32).view(6,5)
    # assert (prod_collective_as(x) == torch.Tensor([[  0.,  17.], [  0., 168.]])).all()
    assert prod_collective_as.total_num_conn() == 3
    assert prod_collective_as.total_max_num_conn() == 6
    # assert list(map(list, prod_collective_as.get_connectivities())) == [connectivity_input_joint, connectivity_joint_output]
    assert (prod_collective_as.get_mask_matrix_encoding() == adjacency_list_2_mask_matrix(
        adjacency_list=connectivities, 
        in_size=input_size, 
        out_size=output_size).flatten()).all()


class JASGBAG(SparseABC):
    """
    The baseline model cannot capture the semantic that a single argument is 
    sufficient to defeat another argument, but multiple arguments are 
    required to defeat another jointly.

    h = \sigma(\sum wi T(Ri) + b) $$
    Extending a normal connection to a joint connection, ri is replaced 
    with T(Ri). T(Ri) is a t-norm applied to a non-empty subset of 
    the arguments from the previous layer. 

    T is the product t-norm operator
    T(Ri) = \prod ri    """
    def __init__(self, 
        no_softmax=False,
        *,
        input_size, 
        hidden_size, 
        output_size, 
        joint_connection_size1,
        joint_connection_size2,
        connections_input_hidden,
        connections_hidden_output,
        connections_jointly_input_hidden,
        connections_jointly_hidden_output,
    ):
        super().__init__()

        self.fitness = torch.tensor(0)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.joint_connection_size1 = joint_connection_size1
        self.joint_connection_size2 = joint_connection_size2

        self.sparse_linear1 = SparseLinearEnhanced(
            input_size, 
            hidden_size, 
            connectivity=connections_input_hidden)
        self.collective1 = ProdCollectiveAS(
            input_size=input_size,
            output_size=joint_connection_size1,
            connectivities=connections_jointly_input_hidden)
        self.activation1 = nn.Sigmoid()

        self.sparse_linear2 = SparseLinearEnhanced(
            hidden_size, 
            output_size, 
            connectivity=connections_hidden_output)
        self.collective2 = ProdCollectiveAS(
            input_size=hidden_size,
            output_size=joint_connection_size2,
            connectivities=connections_jointly_hidden_output)
        if no_softmax:
            self.output_layer = lambda x: x
        else:
            self.output_layer = nn.Softmax()
    
    def reduced_num_conn(self):
        """
        remove not meaningful connections
        returns the number of connections after removing
        """
        connectivity2_all = torch.hstack((
            self.sparse_linear2.connectivity,
            self.collective2.get_connectivities()))
        sparse_linear1_before_remove = self.sparse_linear1.get_connectivities().clone()
        sparse_linear2_before_remove = self.sparse_linear2.get_connectivities().clone()
        collective1_before_remove = self.collective1.get_connectivities().clone()
        collective2_before_remove = self.collective2.get_connectivities().clone()

        connected_hidden_neurons_1 = set()

        sparse_linear1_after_remove = set()
        sparse_linear2_after_remove = set() 
        collective1_after_remove = set()
        collective2_after_remove = set()

        for conn in sparse_linear1_before_remove.T:
            conn = tuple(conn.tolist())
            to_idx, from_idx = conn
            if to_idx in connectivity2_all[1]:
                sparse_linear1_after_remove.add(conn)

        for conn in collective1_before_remove.T:
            conn = tuple(conn.tolist())
            to_idx, from_idx = conn
            if to_idx in connectivity2_all[1]:
                collective1_after_remove.add(conn)

        for conn in sparse_linear1_after_remove:
            to_idx, from_idx = conn
            connected_hidden_neurons_1.add(to_idx)

        for conn in collective1_after_remove:
            to_idx, from_idx = conn
            connected_hidden_neurons_1.add(to_idx)

        for conn in sparse_linear2_before_remove.T:
            conn = tuple(conn.tolist())
            to_idx, from_idx = conn
            if from_idx in connected_hidden_neurons_1:
                sparse_linear2_after_remove.add(conn)

        for conn in collective2_before_remove.T:
            conn = tuple(conn.tolist())
            to_idx, from_idx = conn
            if from_idx in connected_hidden_neurons_1:
                collective2_after_remove.add(conn)

        n_sparse_conns = len(sparse_linear1_after_remove) + len(sparse_linear2_after_remove)
        collective_conns1 = { to_idx for (to_idx, _) in collective1_after_remove }
        collective_conns2 = { to_idx for (to_idx, _) in collective2_after_remove }
        n_collective_conns = len(collective_conns1) + len(collective_conns2)
        return n_sparse_conns + n_collective_conns

    def describe(self):
        return {
            "sparse_linear1": self.sparse_linear1.describe(),
            "collective1": self.collective1.describe(),
            "sparse_linear2": self.sparse_linear2.describe(),
            "collective2": self.collective2.describe(),
        }

    def total_max_num_conn(self):
        """
        Returns the MAXIMUM POSSIBLE number of connections in the network.
        """
        return self.sparse_linear1.total_max_num_conn() + self.collective1.total_max_num_conn() + self.sparse_linear2.total_max_num_conn() + self.collective2.total_max_num_conn()
    
    def total_num_conn(self):
        """
        Returns the total number of connections in the network.
        """
        return self.sparse_linear1.total_num_conn() + self.collective1.total_num_conn() + self.sparse_linear2.total_num_conn() + self.collective2.total_num_conn()

    @classmethod
    def random_connectivity_init(cls, params, no_softmax=False):
        # initialize the network with random connectivity matrix
        input_size = params["input_size"]
        hidden_size = params["hidden_size"]
        output_size = params["output_size"]
        joint_connection_size1 = params["joint_connections_size1"]
        joint_connection_size2 = params["joint_connections_size2"]
        n_connections_input_hidden = params["number_connections1"]
        n_connections_hidden_output = params["number_connections2"]
        n_connections_jointly_input_hidden = params["joint_connections_input_num1"]
        n_connections_jointly_hidden_output = params["joint_connections_input_num2"]

        connectivity_input_hidden = connectivity_list_random(input_size, hidden_size, n_connections_input_hidden)
        connectivity_hidden_output = connectivity_list_random(hidden_size, output_size, n_connections_hidden_output)
        connectivity_jointly_input_hidden = connectivity_list_random(input_size, joint_connection_size1, n_connections_jointly_input_hidden)
        connectivity_jointly_hidden_output = connectivity_list_random(hidden_size, joint_connection_size2, n_connections_jointly_hidden_output)
        return JASGBAG(
            no_softmax=no_softmax,
            input_size=input_size, 
            hidden_size=hidden_size, 
            output_size=output_size, 
            joint_connection_size1=joint_connection_size1,
            joint_connection_size2=joint_connection_size2,
            connections_input_hidden=connectivity_input_hidden,
            connections_hidden_output=connectivity_hidden_output,
            connections_jointly_input_hidden=connectivity_jointly_input_hidden,
            connections_jointly_hidden_output=connectivity_jointly_hidden_output,
        )

    @classmethod
    def from_mask_matrix_encoding_to_connectivities(
            self,
            *,
            input_size:int,
            hidden_size:int,
            output_size:int,
            joint_connection_size1,
            joint_connection_size2,
            mask_matrix_encoding) -> List[torch.Tensor]:
        """
        Converts the genotype back to the phenotype.
        Transforms the bit string/chromosome representation back to the tensor representation.
        First, chromosome has to reshaped (unflattened), before the dense adjacency matrix has
        to be converted to sparse adjacency matrix of shape (m,n).
        """
        encoding_sparse_linear1 = mask_matrix_encoding[0]
        encoding_collective1 = mask_matrix_encoding[1]
        encoding_sparse_linear2 = mask_matrix_encoding[2]
        encoding_collective2 = mask_matrix_encoding[3]

        connectivity_sparse_linear1 = SparseLinearEnhanced.from_mask_matrix_encoding_to_connectivity(
            mask_matrix_encoding=encoding_sparse_linear1,
            input_size=input_size,
            output_size=hidden_size
        )
        connectivities_collective1 = ProdCollectiveAS.from_mask_matrix_encoding_to_connectivity(
            mask_matrix_encoding=encoding_collective1,
            input_size=input_size,
            output_size=joint_connection_size1
        )
        connectivity_sparse_linear2 = SparseLinearEnhanced.from_mask_matrix_encoding_to_connectivity(
            mask_matrix_encoding=encoding_sparse_linear2,
            input_size=hidden_size,
            output_size=output_size
        )
        connectivities_collective2 = ProdCollectiveAS.from_mask_matrix_encoding_to_connectivity(
            mask_matrix_encoding=encoding_collective2,
            input_size=hidden_size,
            output_size=joint_connection_size2
        )
        return (connectivity_sparse_linear1, connectivities_collective1, connectivity_sparse_linear2, connectivities_collective2)

    @classmethod
    def from_mask_matrix_encoding(
            cls, 
            params,
            mask_matrix_encoding : List[torch.Tensor], 
            ):
        """
        Converts the genotype back to the phenotype.
        Transforms the bit string/chromosome representation back to the tensor representation.
        First, chromosome has to reshaped (unflattened), before the dense adjacency matrix has
        to be converted to sparse adjacency matrix of shape (m,n).
        """
        input_size = params["input_size"]
        hidden_size = params["hidden_size"]
        output_size = params["output_size"]
        joint_connection_size1 = params["joint_connections_size1"]
        joint_connection_size2 = params["joint_connections_size2"]
        (connectivity_sparse_linear1, connectivity_jointly_input_hidden , 
         connectivity_sparse_linear2, connectivity_jointly_hidden_output)  = JASGBAG.from_mask_matrix_encoding_to_connectivities(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            joint_connection_size1=joint_connection_size1,
            joint_connection_size2=joint_connection_size2,
            mask_matrix_encoding=mask_matrix_encoding
        )

        return JASGBAG(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            joint_connection_size1=joint_connection_size1,
            joint_connection_size2=joint_connection_size2,
            connections_input_hidden=connectivity_sparse_linear1,
            connections_hidden_output=connectivity_sparse_linear2,
            connections_jointly_input_hidden=connectivity_jointly_input_hidden,
            connections_jointly_hidden_output=connectivity_jointly_hidden_output)

    def get_connectivities(self):
        return (self.sparse_linear1.get_connectivities(),
            self.collective1.get_connectivities(),
            self.sparse_linear2.get_connectivities(),
            self.collective2.get_connectivities())

    def get_mask_matrix_encoding(self):
        """Encodes the structure of the graph as a bit string for genetic algorithm.

        The rows of the connectivity matrix are concatenated.
        """
        return [self.sparse_linear1.get_mask_matrix_encoding(),
            self.collective1.get_mask_matrix_encoding(),
            self.sparse_linear2.get_mask_matrix_encoding(),
            self.collective2.get_mask_matrix_encoding()]

    def forward(self, x):
        x1 = self.sparse_linear1(x)
        x2 = self.collective1(x)
        x = x1
        x[:, :self.joint_connection_size1] += x2
        x = self.activation1(x)
        x1 = self.sparse_linear2(x)
        x2 = self.collective2(x)
        x = x1
        x[:, :self.joint_connection_size2] += x2
        x = self.output_layer(x)
        return x
    


def test_jasgbag():
    input_size = 5
    hidden_size = 10
    output_size = 2
    connectivity_input_hidden = torch.tensor([
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0]
    ], dtype=torch.float32)
    connectivity_hidden_output = torch.tensor([
        [1, 1],
        [0, 0]
    ], dtype=torch.float32)
    connectivity_jointly_input_hidden = torch.tensor([
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1]
    ], dtype=torch.float32)
    connectivity_jointly_hidden_output = torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1]
    ], dtype=torch.float32)
    jasgbag = JASGBAG(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        connections_input_hidden=connectivity_input_hidden,
        connections_hidden_output=connectivity_hidden_output,
        connections_jointly_input_hidden=connectivity_jointly_input_hidden,
        connections_jointly_hidden_output=connectivity_jointly_hidden_output)

    connectivities = [
        connectivity_input_hidden,
        connectivity_jointly_input_hidden,
        connectivity_hidden_output,
        connectivity_jointly_hidden_output
    ]

    x = torch.ones((5, 5))
    y = jasgbag(x)
    assert y.shape == (5, 2)
    # assert torch.allclose(y, torch.ones((5, 2)))
    for i, j in zip(jasgbag.get_connectivities(), connectivities):
        i = i.to(torch.float32)
        j = j.to(torch.float32)
        assert torch.allclose(i, j)

    matrix_encodings = jasgbag.get_mask_matrix_encoding()
    jasgbag2 = JASGBAG.from_flattened_mask_matrices(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        mask_matrix_encoding=matrix_encodings)
    a, b, c, d = jasgbag2.get_connectivities() 
    assert torch.allclose(a, torch.tensor([[1], [0]]))
    assert torch.allclose(b, torch.tensor([[0, 1, 1], [0, 0, 1]]))
    assert torch.allclose(c, torch.tensor([[1], [0]]))
    assert torch.allclose(d, torch.tensor([[1, 1], [0, 1]]))


def test_adjacency_matrix_2_list_conversion():
    adjacency_matrix = torch.tensor([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ], dtype=torch.float32)
    adjacency_list = mask_matrix_2_connectivity_list(adjacency_matrix)
    expected_adjacency_list = torch.tensor([[1, 2, 0, 2, 0, 1, 3, 2, 4, 3],
        [0, 0, 1, 1, 2, 2, 2, 3, 3, 4]])
    assert torch.allclose(expected_adjacency_list, adjacency_list)

def test_adjacency_list_2_matrix_conversion():
    adjacency_list = torch.tensor([[1, 2, 0, 2, 0, 1, 3, 2, 4, 3],
        [0, 0, 1, 1, 2, 2, 2, 3, 3, 4]])
    adjacency_matrix = adjacency_list_2_mask_matrix(adjacency_list, 5, 5)
    expected_adjacency_matrix = torch.tensor([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ], dtype=torch.int)
    assert torch.allclose(expected_adjacency_matrix, adjacency_matrix)

if __name__ == "__main__":
    test_adjacency_list_2_matrix_conversion()
    print("test adjacency list 2 matrix conversion passed")
    test_adjacency_matrix_2_list_conversion()
    print("test adjacency matrix 2 list conversion passed")
    test_prod_collective_as()
    print("test prod collective as passed")
    test_jasgbag()
    print("test jasgbag passed")
    # x = JASGBAG.random_connectivity_init(
    #     input_size=10,
    #     hidden_size=10,
    #     output_size=10,
    #     joint_feature_size1=10,
    #     joint_feature_size2=10,
    #     n_connections_input_hidden=10,
    #     n_connections_hidden_output=10,
    #     n_connections_input_joint1=10,
    #     n_connections_joint1_hidden=10,
    #     n_connections_hidden_joint2=10,
    #     n_connections_joint2_output=10,
    # )
    # z = x.get_mask_matrix_encoding()
    # print(x.get_mask_matrix_encoding())
    # print(z)
    # y = JASGBAG.from_flattened_mask_matrices(
    #     input_size=10,
    #     hidden_size=10,
    #     output_size=10,
    #     joint_feature_size1=10,
    #     joint_feature_size2=10,
    #     mask_matrix_encoding=z
    # )
    # print(y.get_mask_matrix_encoding())
    # def recur_all_eq(l, r):
    #     if isinstance(l, torch.Tensor):
    #         return torch.allclose(l, r)
    #     if isinstance(l, list):
    #         return all(recur_all_eq(x, y) for x, y in zip(l, r))
    # print(recur_all_eq(y.get_mask_matrix_encoding(), z))
    