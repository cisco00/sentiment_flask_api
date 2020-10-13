from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
import numpy as np

data = pd.read_csv('imbd.csv', delimiter ='\t', names=['review', 'rating'])
y = data['review']
x = data['rating']

class model(BaseEstimator, ClassifierMixin):
    def gaussianNB(self):
        pass

    def fit(self, X, y = None):
        return self

        


