import pickle
import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import mode

class GMMClassifier(ClusterMixin, BaseEstimator):
    def __init__(self, n_components=2, covariance_type='full', init_params='kmeans', reg_covar=1e-06):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.init_params = init_params
        self.reg_covar = reg_covar

        self.gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type)

    def fit(self, X, y=None):
        self.gmm.fit(X)
        return self

    def predict(self, X):
        y_pred = self.gmm.predict(X)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        mapped_preds = np.zeros_like(y)

        for cluster in range(self.n_components):
            mask = y_pred == cluster
            if np.any(mask):
                mapped_preds[mask] = mode(y[mask])[0]

        return accuracy_score(y, mapped_preds)


parameters = {'covariance_type': ['full', 'tied', 'diag', 'spherical'],
              'init_params': ['kmeans', 'k-means++', 'random', 'random_from_data'],
              'reg_covar': [1e-07, 1e-06, 1e-05, 1e-04]}

estimator = GMMClassifier()

# load the correct dataset here
X = [] 
y = []

clf = GridSearchCV(estimator, parameters, n_jobs=-1, verbose=1)
clf.fit(X, y)

print(f"Best params: {clf.best_params_}\n Best score: {clf.best_score_}")

with open('gauss-grid_search.pkl', 'wb') as f:
    pickle.dump(clf, f)
