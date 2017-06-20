from scipy import sparse
from .utilities import DEPUtilities
import numpy as np
from ..datastructures.sets import *
from itertools import product

# ===== Some Data Structures =====
class Nodeset(frozenset):

    def __new__(cls, *args, **kwargs):
        return super(Nodeset,cls).__new__(cls,*args, **kwargs)

    def __repr__(self):
        return "n{}".format(set(self))

capture_methods(Nodeset, frozenset, Foofrozenset.capture_methods)

class NodesetCounter(Counter):

    def __init__(self, *args, **kwargs):
        super(NodesetCounter, self).__init__(*args, **kwargs)

    def __repr__(self):
        return "ns{}".format(dict(self))

capture_methods(NodesetCounter, Counter, Foocounter.capture_methods)

# ===== Enum Node Types =====
α, β, γ, δ = 0, 1, 2, 3

# ===== Custom Logging Procedures =====
def clog(label, data):
    print('clog :: {:>32} : {:<}'.format(label, str(data)))

def clog_types(types):
    clog("α type", types[α])
    clog("β type", types[β])
    clog("γ type", types[γ])
    clog("δ type", types[δ])

# ===== Main Methods =====
def compute_node_types(last_nodes, next_edge):

    # Generalized nodes of each type (α,β,γ,δ) where n is a generalized node from last_nodes
    types = (
        [], # α type: e does not overlap with n
        [], # β type: # n contained in e
        [], # γ type: # (n1, n2) = (n - e, n & e) data structure for pairs in type γ (n and e strictly overlap)
        [], # δ type: # Any new nodes introduced by the next_edge that were previously unseen
    )

    δ_remainder = next_edge.copy()

    # n are the generalized nodes
    # @Efficiency: Could replace next_edge with δ_remainder if need be
    for n in last_nodes:
        ne = n & next_edge # intersection of n and e

        if not bool(ne): # the set is empty
            # α case
            types[α].append(n)
        elif n.issubset(next_edge):
            # β case
            types[β].append(n)
            δ_remainder = δ_remainder - n
        else:
            # γ case
            n1 = n - ne
            n2 = ne
            δ_remainder = δ_remainder - ne
            types[γ].append((n1, n2))

    if bool(δ_remainder):
        types[δ].append(δ_remainder)

    return types

def compute_nodes_from_types(types):
    nodes = []
    nodes.extend(types[α])
    nodes.extend(types[β])
    nodes.extend(types[δ])
    unziped = list(zip(*types[γ]))
    if bool(unziped):
        nodes.extend(unziped[0])
        nodes.extend(unziped[1])
    return nodes

def test_compute_generalized_nodes(H):
    H = H.tocsc()
    all_nodes = Nodeset(list(range(H.shape[0])))
    edge_list = sparse_hypergraph_to_edge_list(H)
    clog("edge_list", edge_list)
    # generalized_nodes = [edge_list[0], all_nodes - edge_list[0]]
    generalized_nodes = [edge_list[0]]
    clog("generalized_nodes", generalized_nodes)
    i = 1
    while i < len(edge_list):
        generalized_nodes_as_types = compute_node_types(generalized_nodes, edge_list[i])
        clog_types(generalized_nodes_as_types)
        generalized_nodes = compute_nodes_from_types(generalized_nodes_as_types)
        clog("generalized_nodes", generalized_nodes)
        i += 1
    clog("Test Complete", None)

def sparse_hypergraph_to_edge_list(H):
    return [Nodeset(H.indices[H.indptr[c]:H.indptr[c+1]]) for c in range(H.shape[1])]

# ===== Main Class =====
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

        self.edges = sparse_hypergraph_to_edge_list(self.H_c)

    def add_next_hyperedge(self, last_nodes, transversal, k):
        """
        Given a minimal transversal for Hk, the generalized nodes from the k-th hypergraph, add the (k+1)-th edge
        """
        if k >= self.num_edges:
            # This transversal is minimal and finished
            clog('', '============')
            clog('completed', transversal)
            yield transversal
        else:
            next_edge = self.edges[k] # Upcoming edge
            next_edge_weight = self.W[k] # Upcoming weight for edge E

            clog('transversal', transversal)
            clog('k', k)
            clog('e', e)
            clog('w', w)

            types = compute_node_types(last_nodes, next_edge)

            # Manditory Members  : all of the α and β nodes need to be kept with the same weight
            # γ Choices          : Must select at least transversal[n1 & n2] total members out of n1 or n2 from γ
            # Contributing nodes : If total < w, need to consider members from β, δ and γ2, nodes
            manditory_members = NodesetCounter()

            for n in types[α]:
                manditory_members[n] = transversal[n]

            for n in types[β]:
                manditory_members[n] = transversal[n]

            clog_types(types)
            clog('manditory_members', manditory_members)

            w_β = sum(manditory_members[n] for n in types[β])

            for next_transversal in self.generate_next_transversal(type_αβγδ, k):
                yield from self.add_next_hyperedge(generalized_nodes, next_transversal, k + 1)

    def generate_γ_choices(self, γ_types, transversal):
        # Regarding nodes in γ_types, there will be a γ_choice for every choice
        # of c = transversal[n1 & n2] elements from (n1, n2) γ_types

        if len(γ_types) == 0:
            yield {}
        else:
            for selection_set in product(*(range(c + 1) for c in type_γ.values())):
                # clog('selection_set', selection_set)
                for n, s in zip(type_γ.keys(), selection_set): # TODO: relying on the same ordering between keys and values
                    c = type_γ[n]
                    n1, n2 = n
                    offspring = offspring_seed.copy()
                    if c - s > 0:
                        offspring[n1] = c - s   # c, c-1, c-2, ...
                    if s > 0:
                        offspring[n2] = s       # 0, 1, 2, ...
                    clog('offspring (selection)', offspring)
                    n2_contribution = sum(offspring[n2] for n2 in n2s)
                    offspring_edge_coverage = n2_contribution + w_β

                    yield offspring, offspring_edge_coverage

    def generate_next_transversal(self, type_αβγδ, k):
        w = self.W[k]
        (type_α, type_β, type_γ, type_δ) = type_αβγδ
        (w_α, w_β, w_γ, w_δ) = (sum(type_counter.values()) for type_counter in type_αβγδ)
        n2s = [n[1] for n in type_γ.keys()]

        if not bool(type_γ) and w_β >= w: # no fancy branching needs to be made
            type_αβ = type_α + type_β
            transversal = type_αβ.copy()
            clog('tr, w_γ = 0, w_β >= w:', transversal)
            yield transversal
        else:
            for offspring, coverage in self.generate_offspring(type_αβγδ, k):
                clog('offspring', offspring)
                clog('coverage', coverage)
                if coverage >= w: # Efficient
                    clog('tr, coverage > w', offspring)
                    yield offspring
                else:
                    # coverage < w... need to select (w - coverage) generalized nodes from n2s + type_δ + type_β
                    candidate_nodes = list(type_δ.keys()) + list(type_β.keys()) + n2s
                    clog('candidate_nodes', candidate_nodes)
                    integer_partition = list(list(perm) for part in DEPUtilities.partitionn(w-coverage, len(candidate_nodes), 0) for perm in DEPUtilities.unique_permutations(part))
                    clog('integer_partition', integer_partition)
                    candidates = list(NodesetCounter(dict(zip(candidate_nodes, pp))) for pp in integer_partition)
                    clog('candidates', candidates)
                    # check_if_appropriate = True
                    check_if_appropriate = bool(type_γ) or bool(type_β) # TODO: Figure out the best filter here
                    if check_if_appropriate:
                        for is_appropriate, candidate in zip(self.certify_appropriate_nodes(offspring, candidates, k), candidates):
                            if is_appropriate:
                                offspring = offspring + candidate
                                yield offspring
                    else:
                        for candidate in candidates:
                            offspring = offspring + candidate
                            yield offspring


    def compute_vector_contribution(self, nodeset, k, Hk = None):
        if Hk is None:
            Hk = self.H_r[:,:k+1] # TODO Bottleneck

        # converting nodeset to a vector
        nodeset_vector = sparse.lil_matrix((1, self.num_nodes), dtype='int8')
        nodeset_reps = []
        for n in nodeset:
            rep = next(iter(n)) # pick a representative for the generalized node
            nodeset_reps.append(rep) # for use later
            nodeset_vector[:, rep] = nodeset[n]

        # calculate current contribution of nodeset
        total_contributions = sparse.csr_matrix.dot(nodeset_vector, Hk) # TODO is this really efficient?

        return nodeset_vector, total_contributions

    def compute_node_contribution_dict(self, nodeset, H):
        node_contribution_dict = {}
        for n in nodeset:
            rep = next(iter(n))
            node_contribution_dict[n] = H.indices[H.indptr[rep]:H.indptr[rep+1]]
        return node_contribution_dict

    def certify_appropriate_nodes(self, offspring, candidates, k):

        Hk = self.H_r[:,:k+1] # TODO Bottleneck

        offspring_vector, offspring_contribution = self.compute_vector_contribution(offspring, k, Hk = Hk)
        adjusted_contribution = offspring_contribution - self.W[:k+1]

        node_contribution_dict = self.compute_node_contribution_dict(nodeset, Hk) # TODO determine this nodeset (needs to include offspring nodes and candidate nodes)

        # Huge optimization by caching invalidating nodes
        inappropriate_nodes = []
        for n, node_contribution in node_contribution_dict.items():
            if np.all(adjusted_contribution[:, node_contribution] > 0): # checking the data
                inappropriate_nodes.append(n)
        clog('inappropriate_nodes', inappropriate_nodes)

        # see which candidates care appropriate
        for candidate_set in candidates:
            is_appropriate = True # To be invalidated later
            clog('offspring', offspring)
            clog('candidate_set', candidate_set)

            for n in inappropriate_nodes:
                if n in candidate_set:
                    is_appropriate = False
                    break

            if is_appropriate:
                candidate_vector, candidate_contribution = self.compute_vector_contribution(candidate_set, k, Hk = Hk)
                clog('k', k)
                clog('offspring_contribution', offspring_contribution.toarray())
                clog('adjusted_contribution', adjusted_contribution)
                clog('candidate_contribution', candidate_contribution.toarray())

                # What would the contribution be if we added the candidate (and subtracted the weight)
                checking_contribution = adjusted_contribution + candidate_contribution
                clog('checking_contribution', checking_contribution)

                for n, node_contribution in node_contribution_dict.items():
                    print(n, node_contribution)
                    if np.all(checking_contribution[:, node_contribution] > 0): # checking the data
                        # not an appropriate node
                        # clog('not appropriate candidate', candidate)
                        is_appropriate = False
                        break

            clog('is_appropriate', is_appropriate)
            yield is_appropriate

    def find_all_transversals(self):
        """
        Begins the algorithm.
        """
        if len(self.edges) == 0:
            return []
        else:
            # Empty set of generalized nodes is the only minimal transversal of H_0
            starting_transversal = NodesetCounter() # list of generalized nodes

            transversal_generator = self.add_next_hyperedge([], starting_transversal, 0)

            transversals = list(transversal_generator) # TODO more fancy options for finding transversals
            return transversals
