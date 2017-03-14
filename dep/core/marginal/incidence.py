from scipy import sparse
import numpy as np
from functools import reduce

def incidence(marginal_scenaraio):
    """
    Computes the incidence matrix for the given marginal scenario.

    Args:
        marginal_scenaraio: MarginalScenario object

    Returns:
        Incidence matrix as sparse, dtype=boolean, format=coo
    """
    return sparse.vstack((context_incidence(marginal_scenaraio.outcomes, context) for context in marginal_scenaraio.contexts), dtype='bool')

def context_incidence(outcomes, context):
    """
    Computes the incidence matrix for a single context

    Args:
        outcomes: list of the cardinality of the variables
        context:  list of indices of the subset of outcomes

    Returns:
        Incidence matrix as sparse, dtype=boolean, format=coo
    """
    kronecker_elements = tuple(sparse.identity(outcomes[i], format='coo', dtype='bool') if i in context else sparse.coo_matrix(np.ones(outcomes[i]), dtype='bool') for i in range(len(outcomes)))
    return sparse_multi_kronecker(kronecker_elements)


def sparse_multi_kronecker(kronecker_elements):
    """
    Computes the tensor product of multiple matrix elements in the order they appear

    Args:
        kronecker_elements: list of matrix elements

    Returns:
        The tensor product of kronecker_elements
    """
    return reduce(lambda l, r: sparse.kron(l, r, format='coo'), kronecker_elements)
