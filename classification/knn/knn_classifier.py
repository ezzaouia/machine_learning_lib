import numpy as np

class KNNClassifier(object):
    
    def __init__(sefl):
        pass
    
    # Method the train the model
    # KNN is just remembring the data 
    def train(self, Xtr, Ytr):
        self.X = Xtr
        self.Y = Ytr
    
    # 
    def predict(self, Xte):
        pass
    
    def compute_distances_two_loops(self, Xte):
        Xtr = self.X
        num_train = Xtr.shape[0]
        num_test = Xte.shape[0]
        for i in range(num_test):
            for j in range(num_train):
                dists[i, j] = np.sqrt(np.sum(Xtr[j] - Xte[i]) ** 2)
        return dists