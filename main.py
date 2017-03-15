from dep.core.marginal.marginalscenario import *
from dep.core.marginal.incidence import *
from dep.core.dep.dep import *
from dep.core.dep.transversal import *
from scipy import sparse
import numpy as np

def MiscTest():
    # ms = MarginalScenario(outcomes=[2,2,2], contexts=[[0,1],[1,2],[2,0]])
    # [
    # [1, 1, 1, 0, 0],
    # [0, 0, 1, 1, 1],
    # [1, 0, 0, 0, 1],
    # [0, 1, 0, 0, 1],
    # ]

    ms = MarginalScenario(outcomes=[2,2,2], contexts=[[0,1],[1,2],[2,0]])
    print(repr(ms))
    print(ms.display(['A', 'B', 'C']))
    M = incidence(ms).astype('int8').tocsr()
    # print(M)
    y = sparse.csr_matrix([[1,-1,1,1,-3,-2,0,0,0,0,0,0]], dtype='int8')
    # print(y.todense())
    # print(M.todense())
    e = Extender(M, y)
    e.completely_extend()
    # print(ec._suby.todense())
    # print(ec._subY.todense())
    # print(ec._subM.todense())
    # print(ec._Y_support)
    # print(ec._rM.todense())

def NodesetTest():
    A = Nodeset([0,1,3,4])
    B = Nodeset([0,4,5,2])
    C = Nodeset([0,4])
    print(A)
    print(B)
    print(A - B)
    print(A & B)
    print(C)
    print(C == (A&B))

def TransversalTest():

    # H = sparse.csr_matrix([
    #         [1,1,1,1,1,0,0,0,0],
    #         [1,0,1,1,1,0,0,0,0],
    #         [0,1,0,1,1,0,0,0,0],
    #         [0,1,1,0,1,1,1,0,0],
    #         [0,1,1,1,0,0,0,1,0],
    #         [0,1,1,0,0,0,0,0,1],
    #     ], dtype='int8')
    # W = np.array([1,1,1,1,1,1,1,1,1])
    # H = sparse.csr_matrix([
    #         [1,1,1,1],
    #         [0,1,1,1],
    #         [0,0,1,1],
    #         [0,0,0,1],
    #     ], dtype='int8')
    # W = np.array([1,1,1,2])
    H = sparse.csr_matrix([
            [1,0],
            [0,1],
            [1,1],
        ], dtype='int8')
    W = np.array([8,8])

    T = Transverer(H, W)
    transversals = T.find_all_transversals()
    print(transversals)

# ===== Sand Box =====
if __name__ == '__main__':
    # NodesetTest()
    TransversalTest()
    # MiscTest()

    # y = np.array([1,1,1,1,1])
    # y = sparse.csr_matrix(y)
    # M = sparse.csr_matrix([
    #     [1,1,0,1,0,1],
    #     [1,1,0,1,0,0],
    #     [0,0,0,1,0,1],
    #     [0,0,0,0,0,1],
    #     [0,0,0,0,0,1],
    #     ])
    # # print(y.dot(M))
    # # print(M.T * y)
    # print(sparse.csr_matrix.dot(y, M))