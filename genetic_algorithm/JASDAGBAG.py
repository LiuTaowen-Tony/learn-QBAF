from genetic_algorithm.SparseAlgo import *
from genetic_algorithm.JASGBAG import JASGBAG
from torch import nn
from typing import List
import torch
import math

class JASDAGBAG(SparseAlgo):
    """
    QBAF with both joint attack and support and direct attack and support.
    JASDAGBAG is a DAGBAG with a JASGBAG as the main component.
    please see documentation for JASGBAG and DAGBAG for more information.
    """
    def __init__(self,
                 *,
                 jasgbag: JASGBAG,
                 skip_connectivity
                 ):
        super().__init__()
        self.jasgbag = jasgbag
        self.input_size = jasgbag.input_size
        self.hidden_size = jasgbag.hidden_size
        self.output_size = jasgbag.output_size
        self.sparse_linear_skip = SparseLinearEnhanced(self.input_size, self.output_size, connectivity=skip_connectivity)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        tmp = self.sparse_linear_skip(x)
        x = self.jasgbag(x)
        x = x + tmp
        return self.softmax(x)

    def get_connectivities(self):
        return [
            *self.jasgbag.get_connectivities(),
            self.sparse_linear_skip.get_connectivities()
        ]

    def get_mask_matrix_encoding(self) -> List[torch.Tensor]:
        """Encodes the structure of the graph as a bit string for genetic algorithm.

        The rows of the connectivity matrix are concatenated.
        """
        return [
            *self.jasgbag.get_mask_matrix_encoding(),
            self.sparse_linear_skip.get_mask_matrix_encoding()
        ]

    def describe(self):
        return {
            'jasgbag': self.jasgbag.describe(),
            'sparse_linear_skip': self.sparse_linear_skip.describe()
        }
    
    def reduced_num_conn(self):
        """
        remove not meaningful connections
        returns the number of connections after removing
        """
        return self.jasgbag.reduced_num_conn() + self.sparse_linear_skip.total_num_conn()
    

    @classmethod
    def from_mask_matrix_encoding(cls, params, mask_matrix_encoding: List[torch.Tensor]):
        """
        Converts the genotype back to the phenotype.
        Transforms the bit string/chromosome representation back to the tensor representation.
        First, chromosome has to reshaped (unflattened), before the dense adjacency matrix has
        to be converted to sparse adjacency matrix of shape (m,n).
        """
        jas_enc = mask_matrix_encoding[:4]
        jas_conn = JASGBAG.from_mask_matrix_encoding_to_connectivities(
            input_size = params['input_size'],
            hidden_size = params['hidden_size'],
            output_size = params['output_size'],
            joint_connection_size1=params['joint_connections_size1'],
            joint_connection_size2=params['joint_connections_size2'],
            mask_matrix_encoding = jas_enc
        )
        skip_enc = mask_matrix_encoding[4]
        skip_conn = mask_matrix_2_connectivity_list(
            skip_enc.reshape(params['input_size'], params['output_size'])
        )
        return JASDAGBAG(
            jasgbag=JASGBAG(
                no_softmax=True,
                input_size = params['input_size'],
                hidden_size = params['hidden_size'],
                output_size = params['output_size'],
                joint_connection_size1=params['joint_connections_size1'],
                joint_connection_size2=params['joint_connections_size2'],
                connections_input_hidden=jas_conn[0],
                connections_hidden_output=jas_conn[2],
                connections_jointly_input_hidden=jas_conn[1],
                connections_jointly_hidden_output=jas_conn[3],
            ),
            skip_connectivity=skip_conn
        )

    @classmethod
    def random_connectivity_init(cls, params):
        """
        Initializes the JASDAGBAG with random connectivity. 
        """
        jasgbag = JASGBAG.random_connectivity_init(params, no_softmax=True)
        skip_conn = connectivity_list_random(params['input_size'], params['output_size'], params['number_connections_skip'])
        
        return JASDAGBAG(
            jasgbag=jasgbag,
            skip_connectivity=skip_conn
        )

    def total_num_conn(self):
        """
        Returns the total number of connections in the network.
        """
        return self.jasgbag.total_num_conn() + self.sparse_linear_skip.total_num_conn()

    def total_max_num_conn(self):
        """
        Returns the MAXIMUM POSSIBLE number of connections in the network.
        """
        return self.jasgbag.total_max_num_conn() + self.sparse_linear_skip.total_max_num_conn()

    