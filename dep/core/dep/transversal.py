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

    def add_next_hyperedge(self, T, k):
        """
        Given the set of minimal transversals T, add another edge E

        Args:
            T:  List of generalized nodes as a list of nodesets
            k:  index of the next edge
        """
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
            else: # X and E overlap but X is not a subset of E (or E_r special case)
                # γ case
                X1 = X - XE
                X2 = XE
                E_r = E_r - X2
                type_γ[(X1, X2)] = T[X] # T[X] = T[X1, X2]

        κ_α = len(type_α.keys())
        κ_β = len(type_β.keys())
        κ_γ = len(type_γ.keys())

        print('T', T)
        print('E', E)
        print('type_α', type_α)
        print('type_β', type_β)
        print('type_γ', type_γ)

        offT_seed = type_α + type_β # by construction, keys do not overlap

        # Offspring: Regarding X in type_γ, there will be a new T for every choice of
        # of c = T[X1 & X2] elements from (X1, X2) for (X1, X2) : c in type_γ
        # All of the next transversals will require T[X] multiples of X for all X in type_α, type_β
        # E is composed of all of the X2's and E_r (if it exists)

        # TODO: figure out if this is necessary
        if κ_γ == 0: # the offspring seed is the only minimal transversal
            offT = offT_seed.copy()
            # E_r needs to be there is the case that κ_β = 0
            if κ_β == 0 and bool(E_r):
                offT[E_r] = 1 # Possibly include weight here
            print('yield, κ_γ = 0:', offT)
            yield offT

        # Making sure that the zeroth offspring contains on X1's
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

            # Lemma 2 :: if κ_β > 0 all offspring are minimal transversals and are all of them
            if κ_β > 0:
                print('yield, κ_β > 0, κ_γ > 0:', offT)
                yield offT

            # Lemma 3 :: if κ_β = 0 offspring except the first (T_0) are minimal transversals
            if κ_β == 0:
                # First offspring (T_0) hasn't hit new edge
                # appropriate nodes need to be considered
                if l == 0:

                    # candidate appropriate nodes
                    cand_appr_nodes = NodesetCounter(X[1] for X in type_γ) # all of the X2's
                    for offT_appr in self.generate_appropriate_nodes(offT, cand_appr_nodes, k):
                        print('yield, κ_β = 0, l = 0, appr:', offT_appr)
                        yield offT_appr

                    # including a potential E_r (E_r is a guaranteed appropriate node)
                    if bool(E_r):
                        offT[E_r] = 1
                        print('yield, κ_β = 0, l = 0, E_r:', offT)
                        yield offT

                else: # l > 0: # Not the first offspring, all offspring are minimal transversals
                    print('yield, κ_β = 0, l > 0:', offT)
                    yield offT

    def generate_appropriate_nodes(self, offT, candidates, k):
        # calculate current contribution of offT
        offT_as_vector = sparse.lil_matrix((1, self.num_nodes), dtype='int8')
        for X in offT:
            rep = next(iter(X)) # pick a representative for the generalized node
            offT_as_vector[:, rep] = offT[X]

        # contribution of offT
        Hk = self.H_r[:,:k] # TODO Bottleneck
        contribution = sparse.csr_matrix.dot(offT_as_vector, Hk)

        # subtract from the weight
        appropriate_weight = contribution - self.W[:k]

        # see which candidates pass
        for candidate in candidates:
            rc = next(iter(candidate)) # picking a representative
            candidate_contribution = Hk.indices[Hk.indptr[rc]:Hk.indptr[rc+1]]

            if 0 in appropriate_weight[candidate_contribution]:
                # not an appropriate node
                continue
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
            T = NodesetCounter() # list of generalized nodes
            T[self.edges[0]] += 1

            print(tuple(self.add_next_hyperedge(T, 1)))
