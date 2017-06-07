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


    @staticmethod
    def partitionn(n,k,l=0):
        '''n is the integer to partition, k is the length of partitions, l is the min partition element size'''
        if k < 1:
            return
        if k == 1:
            if n >= l:
                yield (n,)
            return
        for i in range(l,n+1):
            for result in DEPUtilities.partitionn(n-i,k-1,i):
                yield (i,)+result

    @staticmethod
    def unique_permutations(seq):
        """
        Yield only unique permutations of seq in an efficient way.

        A python implementation of Knuth's "Algorithm L", also known from the
        std::next_permutation function of C++, and as the permutation algorithm
        of Narayana Pandita.
        """

        # Precalculate the indices we'll be iterating over for speed
        i_indices = range(len(seq) - 1, -1, -1)
        k_indices = i_indices[1:]

        # The algorithm specifies to start with a sorted version
        seq = sorted(seq)

        while True:
            yield seq

            # Working backwards from the last-but-one index,           k
            # we find the index of the first decrease in value.  0 0 1 0 1 1 1 0
            for k in k_indices:
                if seq[k] < seq[k + 1]:
                    break
            else:
                # Introducing the slightly unknown python for-else syntax:
                # else is executed only if the break statement was never reached.
                # If this is the case, seq is weakly decreasing, and we're done.
                return

            # Get item from sequence only once, for speed
            k_val = seq[k]

            # Working backwards starting with the last item,           k     i
            # find the first one greater than the one at k       0 0 1 0 1 1 1 0
            for i in i_indices:
                if k_val < seq[i]:
                    break

            # Swap them in the most efficient way
            (seq[k], seq[i]) = (seq[i], seq[k])                #       k     i
                                                               # 0 0 1 1 1 1 0 0

            # Reverse the part after but not                           k
            # including k, also efficiently.                     0 0 1 1 0 0 1 1
            seq[k + 1:] = seq[-1:k:-1]