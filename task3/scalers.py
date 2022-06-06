import numpy as np

class MinMaxScaler:
    def fit(self, data):
        """Store calculated statistics

        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.mins = data.min(axis = 0)
        self.maxs = data.max(axis = 0)
        
    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        return (data - self.mins) / (self.maxs - self.mins) 


class StandardScaler:
    def fit(self, data):
        """Store calculated statistics
        
        Parameters:
        data (np.array): train set, size (num_obj, num_features)
        """
        self.avg = data.mean(axis = 0)
        self.disp = data.var(axis = 0)
        
    def transform(self, data):
        """
        Parameters:
        data (np.array): train set, size (num_obj, num_features)

        Return:
        np.array: scaled data, size (num_obj, num_features)
        """
        return (data - self.avg) / np.sqrt(self.disp)
