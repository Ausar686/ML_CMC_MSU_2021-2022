import numpy as np
from collections import defaultdict

def kfold_split(num_objects, num_folds):
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds (last fold can be longer) and returns num_folds train-val 
       pairs of indexes.

    Parameters:
    num_objects (int): number of objects in train set
    num_folds (int): number of folds for cross-validation split

    Returns:
    list((tuple(np.array, np.array))): list of length num_folds, where i-th element of list contains tuple of 2 numpy arrays,
                                       the 1st numpy array contains all indexes without i-th fold while the 2nd one contains
                                       i-th fold
    """
    if num_folds <= 1:
        return [(np.arange(num_objects), np.array([]))]
    num_elems = num_objects // num_folds
    del_obj = np.arange(0, num_elems)
    del_obj.resize(1, num_elems)
    del_obj = (np.repeat(del_obj, num_folds - 1, axis = 0) + np.arange(0, num_folds - 1).reshape(-1, 1) * (num_objects + num_elems)).ravel()
    data = np.arange(0, num_objects)
    data.resize(1, num_objects)
    data = np.repeat(data, num_folds - 1, axis = 0)
    rest = np.delete(data, np.s_[del_obj]).reshape(-1, num_objects - num_elems)
    vals = np.arange(0, num_elems)
    vals.resize(1, num_elems)
    vals = np.repeat(vals, num_folds - 1, axis = 0) + np.arange(0, num_folds - 1).reshape(-1, 1) * num_elems
    res = []
    for i in range(num_folds - 1):
        res.append((rest[i], vals[i]))
    res.append((np.arange(0, (num_folds - 1) * num_elems), np.arange((num_folds - 1) * num_elems, num_objects)))
    return res

def knn_cv_score(X, y, parameters, score_function, folds, knn_class):
    """Takes train data, counts cross-validation score over grid of parameters (all possible parameters combinations) 

    Parameters:
    X (2d np.array): train set
    y (1d np.array): train labels
    parameters (dict): dict with keys from {n_neighbors, metrics, weights, normalizers}, values of type list,
                       parameters['normalizers'] contains tuples (normalizer, normalizer_name), see parameters
                       example in your jupyter notebook
    score_function (callable): function with input (y_true, y_predict) which outputs score metric
    folds (list): output of kfold_split
    knn_class (obj): class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight), value - mean score over all folds
    """
    res = {}
    for n in parameters["n_neighbors"]:
        for metric_ in parameters["metrics"]:
            for weight in parameters["weights"]:
                for normalizer in parameters["normalizers"]:
                    if normalizer[1] == "None":
                        X1 = X
                    else:
                        scl = normalizer[0]
                        scl.fit(X)
                        X1 = scl.transform(X)
                    score = 0
                    for fold in range(len(folds)):
                        X_train = X1[folds[fold][0], :]
                        X_test = X1[folds[fold][1], :]
                        y_train = y[folds[fold][0]]
                        y_test = y[folds[fold][1]]
                        clf = knn_class(n_neighbors = n, weights = weight, metric = metric_)
                        clf.fit(X_train, y_train)
                        y_predict = clf.predict(X_test)
                        score += score_function(y_test, y_predict)
                    res[(normalizer[1], n, metric_, weight)] = score / len(folds)
    return res
 
