# from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator


class MajorityClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        c = Counter(y)
        self.mode = c.most_common(1)[0][0]  ##getting the most common features in the data set
        self.fraction = np.array(list(c.values())) / X.shape[0]  ##getting the list of all the values

        return self

    def predict(self, X):
        return self.mode_ * np.ones(X.shape[0])

    def predict_proba(self, X):
        np.dot(np.ones(X.shape[0], 1)), self.fraction.reshape(-1, 3)
