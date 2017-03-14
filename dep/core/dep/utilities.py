from scipy import sparse
import numpy as np

class DEPUtilities():

    # @staticmethod
    # def get_contributors(M, y):


    #     return t.T * H

    @staticmethod
    def get_nonzero(nparray):
        return np.where(nparray != 0)[0]

    @staticmethod
    def get_flux(nparray):
        return nparray[1:] - nparray[:-1]

    @staticmethod
    def get_negative_indices(sparray):
        return sparse.find(sparray < 0)[1]

    @staticmethod
    def efficiently_progress(M, y, c, Y):
        """
        Given row c of M, compute how this changes y, Y and return the new y,Y

        Args:
            M: csr format
            y:
            c:
            Y:
        """
        assert M.format == 'csr', "M must be a sparse matrix in csr format"
        contribution = M.getrow(c)
        Y_new = Y + contribution
        y_new = y.copy()
        y_new[0,c] = y_new[0,c] + 1

        return y_new, Y_new

