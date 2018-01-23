import numpy as np
from numpy import *
from sklearn.tree import DecisionTreeClassifier
class BatchClassifier:

    def __init__(self, window_size=100, max_models=10):
        self.H = []
        self.h = None
        # TODO
        self.window_size = window_size
        self.max_models = max_models
        self.window = {'X':[],'y':[]}

    def partial_fit(self, X, y=None, classes=None):
        # TODO
        if self.h==None:
            self.h = DecisionTreeClassifier()
        N,D = X.shape
        for i in range(N):
            self.window['X'].append(X[i])
            self.window['y'].append(y[i])
            if len(self.window['X']) == self.window_size:
                self.h.fit(np.array(np.array(self.window['X'])),np.array(self.window['y']))
                self.window = {'X':[],'y':[]}
                if(len(self.H)==self.max_models):
                    self.H.pop(0)
                self.H.append(self.h)
        return self

    def predict(self, X):
        # TODO
        N,D = X.shape
        preds = []
        for i in range(len(self.H)):
            preds.append(self.H[i].predict(X))
        preds = np.transpose(preds).tolist()
        for i in range(len(preds)):
            p = preds[i]
            preds[i]= max(p,key=p.count)
        # You also need to change this line to return your prediction instead of 0s:
        return preds
