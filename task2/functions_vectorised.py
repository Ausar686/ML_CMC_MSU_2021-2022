import numpy as np


def prod_non_zero_diag(X: np.ndarray) -> int:
    """
    Compute product of nonzero elements from matrix diagonal, 
    return -1 if there is no such elements.
    
    Return type: int / np.integer / np.int32 / np.int64
    """
    b = X.ravel()[::X.shape[1] + 1]
    if b[b != 0].shape[0]: 
        return b[b != 0].prod()
    return -1


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Return True if both 1-d arrays create equal multisets, False if not.
    
    Return type: bool / np.bool_
    """
    if x.shape[0] != y.shape[0]:
        return False
    return np.min(np.sort(x) == np.sort(y))


def max_after_zero(x: np.ndarray) -> int:
    """
    Find max element after zero in 1-d array, 
    return -1 if there is no such elements.

    Return type: int / np.integer / np.int32 / np.int64
    """
    if not np.max(x[:-1] == 0):
        return -1
    return x[1:][x[:-1] == 0].max()


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Sum up image channels with weights.

    Return type: np.ndarray
    """
    return (image * weights).sum(axis = 2)


def run_length_encoding(x: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Make run-length encoding.

    Return type: (np.ndarray, np.ndarray)
    """
    x_ = np.hstack((x[1:], x[-1]+1))
    y = x != x_
    y_ = np.hstack((-1, y[:-1]))
    return (x[y], np.unique(y * np.arange(1, x.shape[0] + 1))[1:] - np.unique(y * np.arange(1, x.shape[0] + 1))[:-1])

def pairwise_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Return pairwise object distance.

    Return type: np.ndarray
    """
    a = np.add.outer(np.sum(X**2, axis=1), np.sum(Y**2, axis=1))
    b = np.dot(X, Y.T)
    return np.sqrt(a - 2*b)

