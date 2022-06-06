from typing import List


def prod_non_zero_diag(X: List[List[int]]) -> int:
    """
    Compute product of nonzero elements from matrix diagonal, 
    return -1 if there is no such elements.

    Return type: int
    """
    flag = False
    res = 1
    size = min(len(X), len(X[0]))
    for i in range(size):
        if (X[i][i] != 0):
            res *= X[i][i]
            flag = True
    if flag:
        return res
    else:
        return -1


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Return True if both 1-d arrays create equal multisets, False if not.

    Return type: bool
    """
    if (len(x) != len(y)):
        return False
    x = sorted(x)
    y = sorted(y)
    for i in range(len(x)):
        if (x[i] != y[i]):
            return False
    return True


def max_after_zero(x: List[int]) -> int:
    """
    Find max element after zero in 1-d array, 
    return -1 if there is no such elements.

    Return type: int
    """
    flag = False
    prev = False
    res = 0
    for i in range(len(x)):
        if prev:
            if not flag:
                res = x[i]
                flag = True
            else:
                if res < x[i]:
                    res = x[i]
            if x[i]:
                prev = False
        else:
            if not x[i]:
                prev = True
    if flag:
        return res
    else:
        return -1


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Sum up image channels with weights.

    Return type: List[List[float]]
    """
    dim1 = len(image)
    dim2 = len(image[0])
    dim3 = len(image[0][0])
    res = [[0 for j in range(dim2)] for i in range(dim1)]
    for i in range(dim1):
        for j in range(dim2):
            for k in range(dim3):
                res[i][j] += image[i][j][k] * weights[k]
    return res


def run_length_encoding(x: List[int]) -> (List[int], List[int]):
    """
    Make run-length encoding.

    Return type: (List[int], List[int])
    """
    arr1 = []
    arr2 = []
    cur = x[0]
    curlen = 0
    for i in range(len(x)):
        if x[i] == cur:
            curlen += 1
        else:
            arr1.append(cur)
            arr2.append(curlen)
            cur = x[i]
            curlen = 1
    arr1.append(cur)
    arr2.append(curlen)
    return (arr1, arr2)


def pairwise_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Return pairwise object distance.

    Return type: List[List[float]]
    """
    dim = len(X[0])
    res = [[0 for j in range(len(Y))] for i in range(len(X))]
    for i in range(len(X)):
        for j in range(len(Y)):
            for k in range(dim):
                res[i][j] += (X[i][k] - Y[j][k]) ** 2
            res[i][j] = res[i][j] ** 0.5
    return res
