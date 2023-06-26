import numpy as np

from .mutation import Mutation


class FlipMutation(Mutation):
    def __init__(self, mutation_probability):
        """
        :param mutation_probability: Probability of mutation for each gene.
        """
        super().__init__(mutation_probability)

    # def mutate(self, offspring):
    #     """Applies flip mutation on a binary encoded chromosome.

    #         Each gene whose probability is <= the mutation probability is mutated randomly.
    #     """
    #     mutated_offspring = np.array(offspring)
    #     for offspring_idx in range(offspring.shape[0]):
    #         probs = np.random.random(size=offspring.shape[1])
    #         for gene_idx in range(offspring.shape[1]):
    #             if probs[gene_idx] <= self.mutation_probability:
    #                 target_gene = mutated_offspring[offspring_idx, gene_idx]
    #                 idx = list()
    #                 for max_idx in target_gene.shape:
    #                     idx.append(np.random.randint(max_idx))
    #                 target_gene.__setitem__(tuple(idx), not target_gene.__getitem__(tuple(idx)))
    #     return mutated_offspring

    def mutate(self, offspring):
        """Applies flip mutation on a binary encoded chromosome.

            Each gene whose probability is <= the mutation probability is mutated randomly.
        """
        mutated_offspring = np.array(offspring)
        for offspring_idx in range(offspring.shape[0]):
            probs = np.random.random(size=offspring.shape[1])
            for gene_idx in range(offspring.shape[1]):
                if probs[gene_idx] <= self.mutation_probability:
                    # print("flippped")
                    # print(mutated_offspring[offspring_idx, gene_idx])
                    mutated_offspring[offspring_idx, gene_idx] = (not offspring[offspring_idx, gene_idx])
        return mutated_offspring
