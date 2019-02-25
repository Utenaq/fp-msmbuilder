"""
Utilities related to indexing upper triangular matrices with a diagonal
offset of 1. The semantics match ``numpy.triu_indices(n, k=1)``
"""
from numpy cimport npy_intp


cdef inline npy_intp ij_to_k(npy_intp i, npy_intp j, npy_intp n) nogil:
    """2D (i, j) square matrix index to linearized upper diagonal index

    [ 0  a0   0   0   0 ]    (i=0,j=1) -> 0
    [ 0   0  a1   0   0 ]    (i=1,j=2) -> 1
    [ 0   0   0  a2   0 ]    (i=2,j=3) -> 2
    [ 0   0   0   0  a3 ]       etc
    [ 0   0   0   0   0 ]    (i=3,j=4) -> 3

    For further explanation, see http://stackoverflow.com/a/27088560/1079728

    Parameters
    ----------
    i : int
        Row index
    j : int
        Column index
    n : int
        Matrix size. The matrix is assumed to be square

    Returns
    -------
    k : int
        Linearized upper triangular index

    See Also
    --------
    k_to_ij : the inverse operation
    """
    if j == i+1:
        return i
    return j

cdef inline void k_to_ij(npy_intp k, npy_intp n, npy_intp *i, npy_intp *j) nogil:
    """Linearized upper diagonal index to 2D (i, j) index

    [ 0  a0   0   0   0 ]    0 -> (i=0,j=1)
    [ 0   0  a1   0   0 ]    1 -> (i=1,j=2)
    [ 0   0   0  a2   0 ]    
    [ 0   0   0   0  a3 ]       etc
    [ 0   0   0   0   0 ]    

      http://stackoverflow.com/a/27088560/1079728

    Parameters
    ----------
    k : int
        Linearized upper triangular index

    Returns
    -------
    i : int
        Row index, written into *i on exit
    j : int
        Column index, written into *j on exit
    """

    i[0] = k
    j[0] = k + 1
