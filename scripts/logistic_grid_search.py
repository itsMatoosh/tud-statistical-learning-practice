import pickle
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

parameters = {'penalty': [None, 'l2', 'l1', 'elasticnet'],
              'C': [0.0, 0.25, 0.50, 0.75, 1.0],
              'fit_intercept': [True, False],
              'l1_ratio': [0.0, 0.25, 0.50, 0.75, 1.0],
              'tol': [1e-03, 1e-04, 1e-05]}

estimator = LogisticRegression(random_state=42, max_iter=1000)

with open("processed_data/X_train_pca.pkl", "rb") as f:
    X = pickle.load(f)

y = np.load("processed_data/y_train.npy")

clf = GridSearchCV(estimator, parameters, n_jobs=-1, verbose=1)
clf.fit(X, y)

print(f"Best params: {clf.best_params_}\n Best score: {clf.best_score_}")

with open('logi-grid_search.pkl', 'wb') as f:
    pickle.dump(clf, f)
