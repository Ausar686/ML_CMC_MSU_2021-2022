import numpy as np


class Preprocesser:
    
    def __init__(self):
        pass
    
    def fit(self, X, Y=None):
        pass
    
    def transform(self, X):
        pass
    
    def fit_transform(self, X, Y=None):
        pass
    
    
class MyOneHotEncoder(Preprocesser):
    
    def __init__(self, dtype=np.float64):
        super(Preprocesser).__init__()
        self.dtype = dtype
        
    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        #your code here
        self.class_category = []
        self.nclass_category = np.zeros(X.shape[1])
        self.offset = np.zeros(X.shape[1])
        self.dim1 = 0
        idx = 0
        for column in X.columns:
            classes = np.unique(X[column])
            n_classes = classes.shape[0]
            self.class_category.append(classes)
            self.nclass_category[idx] = n_classes
            self.offset[idx] = self.dim1
            self.dim1 += n_classes
            idx += 1
        self.class_category = np.array(self.class_category)

    
    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        #your code here
        res = np.zeros([X.shape[0], self.dim1])
        for idx in range(X.shape[0]):
            for column_idx in range(X.columns.shape[0]):
                cur_type = np.where(self.class_category[column_idx] == X.iloc[idx][X.columns[column_idx]])[0]
                offset = int(self.offset[column_idx])
                res[idx, offset + cur_type] = 1
        return res
        
    
    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)
    
    def get_params(self, deep=True):
        return {"dtype": self.dtype}
    
    
class SimpleCounterEncoder:
    
    def __init__(self, dtype=np.float64):
        self.dtype=dtype
        
    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        #your code here
        self.n = X.shape[0]
        self.list = []
        X_ = np.array(X)
        Y_ = np.array(Y)
        for cat_idx in range(X.shape[1]):
            cur_dict = dict()
            for idx in range(X.shape[0]):
                if not(X_[idx, cat_idx] in cur_dict):
                    cur_dict[X_[idx, cat_idx]] = [Y_[idx], 1]
                else:
                    cur_dict[X_[idx, cat_idx]][0] += Y_[idx]
                    cur_dict[X_[idx, cat_idx]][1] += 1
            self.list.append(cur_dict)
        self.list = np.array(self.list)
            
    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        #your code here
        res = np.zeros([X.shape[0], 3 * X.shape[1]])
        X_ = np.array(X)
        for idx in range(X.shape[0]):
            for col_idx in range(X.shape[1]):
                col_name = X_[idx][col_idx]
                success = self.list[col_idx][col_name][0] / self.list[col_idx][col_name][1]
                counter = self.list[col_idx][col_name][1] / self.n
                relation = (success + a) / (counter + b)
                res[idx, 3 * col_idx] = success
                res[idx, 3 * col_idx + 1] = counter
                res[idx, 3 * col_idx + 2] = relation
        return res
    
    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)
    
    def get_params(self, deep=True):
        return {"dtype": self.dtype}

    
def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_ : (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_ :], idx[:(n_splits - 1) * n_]

    
class FoldCounters:
    
    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds
        
    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        #your code here
        self.folds = [fold[0] for fold in group_k_fold(X.shape[0], n_splits = self.n_folds, seed = seed)]
        self.data = []
        self.n = [X.shape[0] - len(self.folds[num_fold]) for num_fold in range(self.n_folds)]
        X_ = np.array(X)
        Y_ = np.array(Y)
        for num_fold in range(self.n_folds):
            fold_list = []
            for col_idx in range(X.shape[1]):
                cur_dict = dict()
                for idx in range(X.shape[0]):
                    if idx in self.folds[num_fold]:
                        continue
                    if not(X_[idx, col_idx] in cur_dict):
                        cur_dict[X_[idx, col_idx]] = [Y_[idx], 1]
                    else:
                        cur_dict[X_[idx, col_idx]][0] += Y_[idx]
                        cur_dict[X_[idx, col_idx]][1] += 1
                fold_list.append(cur_dict)
            self.data.append(fold_list)
            
    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """
        #your code here
        res = np.zeros([X.shape[0], 3 * X.shape[1]])
        res = np.zeros([X.shape[0], 3 * X.shape[1]])
        for num_fold in range(self.n_folds):
            for idx in self.folds[num_fold]:
                for col_idx in range(X.shape[1]):
                    success = self.data[num_fold][col_idx][X.iloc[idx, col_idx]][0] / self.data[num_fold][col_idx][X.iloc[idx, col_idx]][1]
                    counter = self.data[num_fold][col_idx][X.iloc[idx, col_idx]][1] / self.n[num_fold]
                    relation = (success + a) / (counter + b)
                    res[idx, 3 * col_idx] = success
                    res[idx, 3 * col_idx + 1] = counter
                    res[idx, 3 * col_idx + 2] = relation
        return res
        
    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)
 
       
def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    #your code here
    elems = np.unique(x)
    enc = SimpleCounterEncoder()
    tmp = enc.fit_transform(x.reshape(-1, 1), y)[:, 0]
    res = [] 
    for elem in elems:
        for idx in range(x.shape[0]):
            if x[idx] == elem:
                res.append(tmp[idx])
                break
    res = np.array(res)
    return res
