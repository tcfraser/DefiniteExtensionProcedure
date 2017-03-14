from scipy import sparse
import numpy as np
from dep.core.dep.utilities import DEPUtilities as du

class DefiniteExtensionPool():
    """
    Object storing the set of completed definite extensions
    """
    def __init__(self,):
        self._raw = []
        pass

    def is_reducible(self, y, Y):
        pass

    def append(self, y, Y):
        self._raw.append(y)
        pass

class ExtenderWorker():
    """
    Given a matrix M (in both formats), a multi-set y, and Y extension,
    search for modifications to y such that y*M = Y

    Args:
        M_r: csr format
        M_c: csc format
        y:   lil format
        Y:
    """
    def __init__(self, M_r, M_c, y, Y, **options):
        self.M_r = M_r.tocsr()
        self.M_c = M_c.tocsc()
        self.y = y
        self.Y = Y
        self.options = options

    def branch(self, y_pool):
        inhibitors = du.get_negative_indices(self.Y)
        if y_pool.is_reducible(self.y, self.Y):
            # This y has its extension reducible to one already found
            pass
        elif len(inhibitors) == 0:
            # The extension is positive semi-definite
            y_pool.append(self.y, self.Y)
        else:
            # Have to take care of this inhibitor at some point
            next_inhibitor = inhibitors[0]
            # Get the rows that contribute to this inhibitor
            contributors = self.M_c.indices[self.M_c.indptr[next_inhibitor]:self.M_c.indptr[next_inhibitor+1]]

            # Determining the appropriate nodes


            for c in contributors:
                y_new, Y_new = du.efficiently_progress(self.M_r, self.y, c, self.Y)
                next_worker = ExtenderWorker(self.M_r, self.M_c, y_new, Y_new, **self.options)
                next_worker.branch(y_pool)

class ExtensionContext():
    """
    Given a matrix M and a seed y to extend, compute sub-M such
    and indices that correspond to the sub-M context
    """
    def __init__(self, M, y, **options):
        self.M = M
        self.y = y
        self.y_support = self.y.indices
        self.negy = du.get_negative_indices(self.y)

        # Inhibiting set
        self.Y = self.y * self.M
        self.Y_support = self.Y.indices
        # self.inhib = self.Y_support[np.where((self.Y[:, self.Y_support] < 0).todense())[1]]
        self.negY = du.get_negative_indices(self.Y)

        # Compute potential contributors
        self.col_reduced_M = self.M.tocsc()[:, self.negY].tocsr()
        contributor_flux = du.get_flux(self.col_reduced_M.indptr)
        self.contributors = du.get_nonzero(contributor_flux)
        contributor_flux[self.negy] = 0
        self.useful_contributors = du.get_nonzero(contributor_flux)

        # Compute sub-context
        self.subM = self.col_reduced_M[self.useful_contributors, :]
        self.subM_row_sum = np.array(self.subM.sum(axis=1)).flatten()
        self.subM_col_sum = np.array(self.subM.sum(axis=0)).flatten()
        self.suby = self.y[:, self.useful_contributors]
        self.subY = self.Y[:, self.negY]
        self._log()

    def _log(self):
        print("y", self.y.todense())
        print("y_support", self.y_support)
        print("negy",self.negy)
        print("Y", self.Y.todense())
        print("Y_support", self.Y_support)
        print("negY",self.negY)
        print("contributors",self.contributors)
        print("useful_contributors",self.useful_contributors)
        print("subM")
        print(self.subM.todense())
        print("suby", self.suby.todense())
        print("subY", self.subY.todense())
        print("subM_row_sum", self.subM_row_sum)
        print("subM_col_sum", self.subM_col_sum)

class Extender():
    """
    Given a matrix M a multi-set y,
    search for modifications to y such that y*M = Y
    """
    def __init__(self, M, y, **options):
        self.M = M
        self.y = y
        self.ec = ExtensionContext(self.M, self.y)

    def completely_extend(self):
        ew = ExtenderWorker(self.ec.subM.tocsr(), self.ec.subM.tocsc(), self.ec.suby, self.ec.subY)
        y_pool = DefiniteExtensionPool()
        ew.branch(y_pool)
        y_pool = sparse.vstack(y_pool._raw)
        print(y_pool.shape)
        print(y_pool.todense()[-30:,:])
