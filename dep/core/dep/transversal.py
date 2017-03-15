from scipy import sparse
import numpy as np
from ..datastructures.sets import *
from itertools import product

class Nodeset(frozenset):

    def __new__(cls, *args, **kwargs):
        return super(Nodeset,cls).__new__(cls,*args, **kwargs)

capture_methods(Nodeset, frozenset, Foofrozenset.capture_methods)

class NodesetCounter(Counter):

    def __init__(self, *args, **kwargs):
        super(NodesetCounter, self).__init__(*args, **kwargs)

capture_methods(NodesetCounter, Counter, Foocounter.capture_methods)

class Transverer():

    def __init__(self, H, W):
        """
        Computes all of the minimal hypergraph transversals of a hypergraph H
        with respect to edge weights W and using the algorithm of
        Kavvadias, Stavropoulos
        https://www.emis.de/journals/JGAA/accepted/2005/KavvadiasStavropoulos2005.9.2.pdf

        Args:
            H:  The hypergraph to be transversed
                sparse array csr format (n x m)
            W:  The weights of the edges to be hit (typical hypergraphs
                have weight 1 for all edges)
                numpy array (1 x m)
        """

        # assert H.format == 'csr', "Hypergraph H must have format 'csr' not {}".format(H.format)

        self.H_r = H.tocsr()
        self.H_c = H.tocsc()
        self.num_nodes, self.num_edges = H.shape

        assert self.num_edges == len(W), "Length of W {} must equal the number of edges {}".format(len(W), self.num_edges)
        self.W = W

        self.edges = [Nodeset(self.H_c.indices[self.H_c.indptr[c]:self.H_c.indptr[c+1]]) for c in range(self.H_c.shape[1])]

    def generate_next_hyperedge(self, T, k):
        """
        Given the set of minimal transversals T, add another edge E

        Args:
            T:  List of generalized nodes as a list of nodesets
            k:  index of the next edge
            w:  current weight of the edge
        """
        # print("logging, k", k)
        if k >= self.num_edges:
            # This transversal is minimal and finished
            # print("Finished a transversal")
            yield T
        else:
            E = self.edges[k]

            type_α = NodesetCounter() # generalized nodes X of T that are of type α
            type_β = NodesetCounter() # generalized nodes X of T that are of type β
            type_γ = NodesetCounter() # {(X1, X2): T[X1 & X2], ...} data structure for pairs in type γ

            E_r = E.copy() # the remainder of E after removing all of the X2 intersections

            # X are the generalized nodes
            for X in T:
                XE = X & E # intersection of X and E

                if not bool(XE): # the set is empty
                    # α case
                    type_α[X] = T[X]
                elif X.issubset(E):
                    # β case
                    type_β[X] = T[X]
                    E_r = E_r - X
                else: # X and E overlap but X is not a subset of E (or E_r special case)
                    # γ case
                    X1 = X - XE
                    X2 = XE
                    E_r = E_r - X2
                    type_γ[(X1, X2)] = T[X] # T[X] = T[X1, X2]

            κ_α = len(type_α.keys())
            κ_β = len(type_β.keys())
            κ_γ = len(type_γ.keys())

            # print('logging, T', T)
            # print('logging, E', E)
            # print('logging, type_α', type_α)
            # print('logging, type_β', type_β)
            # print('logging, type_γ', type_γ)
            # print('logging, E_r', E_r)

            offT_seed = type_α + type_β # by construction, keys do not overlap

            # Offspring: Regarding X in type_γ, there will be a new T for every choice of
            # of c = T[X1 & X2] elements from (X1, X2) for (X1, X2) : c in type_γ
            # All of the next transversals will require T[X] multiples of X for all X in type_α, type_β
            # E is composed of all of the X2's and E_r (if it exists)

            for l, offT in self.generate_offspring(offT_seed, type_γ):
                # Lemma 2 :: if κ_β > 0 all offspring are minimal transversals and are all of them
                if κ_β > 0:
                    # print('logging, κ_β > 0, κ_γ > 0:', offT)
                    yield from self.generate_next_hyperedge(offT, k + 1)

                # Lemma 3 :: if κ_β = 0 offspring except the first (T_0) are minimal transversals
                if κ_β == 0:
                    # First offspring (T_0) hasn't hit new edge
                    # appropriate nodes need to be considered
                    if l == 0:
                        # candidate appropriate nodes
                        cand_appr_nodes = NodesetCounter(X[1] for X in type_γ) # all of the X2's
                        if bool(E_r):
                            cand_appr_nodes[E_r] = 1 # E_r is a guaranteed appropriate node if κ_γ > 0
                        for offT_appr in self.generate_appropriate_nodes(offT, cand_appr_nodes, k):
                            # print('logging, κ_β = 0, l = 0, appr:', offT_appr)
                            yield from self.generate_next_hyperedge(offT_appr, k + 1)

                    else: # l > 0: # Not the first offspring, all offspring are minimal transversals
                        # print('logging, κ_β = 0, l > 0:', offT)
                        yield from self.generate_next_hyperedge(offT, k + 1)

    def generate_offspring(self, offT_seed, type_γ):
        if not bool(type_γ): # κ_γ = 0 and there is no fancy branching to be made
            offT = offT_seed.copy()
            # print('logging, κ_γ = 0:', offT)
            yield 0, offT
        else:
            # Making sure that the zeroth offspring contains all X1's
            for l, i_set in enumerate(product(*(range(c + 1) for c in type_γ.values()))):
                # build the lth offspring
                for X, i in zip(type_γ.keys(), i_set): # TODO: relying on the same ordering between keys and values
                    c = type_γ[X]
                    X1, X2 = X
                    offT = offT_seed.copy()
                    if c - i > 0:
                        offT[X1] = c - i   # c, c-1, c-2, ...
                    if i > 0:
                        offT[X2] = i       # 0, 1, 2, ...
                # offT is now ready
                # print('logging, l = {}:'.format(l), offT)
                yield l, offT

    def generate_appropriate_nodes(self, offT, candidates, k):
        # calculate current contribution of offT
        offT_as_vector = sparse.lil_matrix((1, self.num_nodes), dtype='int8')
        offT_reps = []
        for X in offT:
            rep = next(iter(X)) # pick a representative for the generalized node
            offT_reps.append(rep) # for use later
            offT_as_vector[:, rep] = offT[X]

        # contribution of offT
        Hk = self.H_r[:,:k] # TODO Bottleneck
        node_contributions      = [Hk.indices[Hk.indptr[Tr]:Hk.indptr[Tr+1]] for Tr in offT_reps]
        total_contributions     = sparse.csr_matrix.dot(offT_as_vector, Hk) # TODO is this really efficient
        adjusted_contributions  = total_contributions - self.W[:k] # subtract from the weight

        # see which candidates pass
        for candidate in candidates:
            is_appropriate = True
            # print('logging, candidate', candidate)
            rc = next(iter(candidate)) # picking a representative
            candidate_contribution = Hk.indices[Hk.indptr[rc]:Hk.indptr[rc+1]]

            adjusted_contributions[:, candidate_contribution] += 1 # What would the contribution be if we added the candidate

            for node_contribution in node_contributions:
                if np.all(adjusted_contributions[:, node_contribution] > 0): # checking the data
                    # not an appropriate node
                    # print('logging, not appropriate candidate', candidate)
                    is_appropriate = False
                    break
            adjusted_contributions[:, candidate_contribution] -= 1 # Saving memory

            if is_appropriate:
                # print('logging, appropriate candidate', candidate)
                # is an appropriate node
                offT_copy = offT.copy()
                offT_copy[candidate] = 1
                yield offT_copy

    def find_all_transversals(self):
        """
        Begins the algorithm.
        """
        if len(self.edges) == 0:
            return []
        else:
            # Empty set of generalized nodes is the only minimal transversal of H_0
            sT = NodesetCounter() # list of generalized nodes

            transversal_generator = self.generate_next_hyperedge(sT, 0)

            transversals = list(transversal_generator) # TODO more fancy options for finding transversals
            return transversals
