import numpy as np
import pickle

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.stats import mode

RNG=42
features = ['pca', 'sift']

y_train = np.load("processed_data/y_train.npy")
y_test = np.load("processed_data/y_test.npy")

n_classes = len(np.unique(y_test))

class GMMClassifier(ClusterMixin, BaseEstimator):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=RNG)

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


def make_model(): return GMMClassifier(n_components=n_classes)

print("Starting auto-generated data experiment")

for feature in features:
    with open(f"processed_data/X_train_{feature}.pkl", "rb") as f:
        X = pickle.load(f)

    print(f"Training with {feature} features with shape: {X.shape}")
    model = make_model()
    scores = cross_val_score(model, X, y_train, cv=5, n_jobs=-1, scoring='accuracy')
    print(f"Finished Cross-validation with with {feature} features.\nCross-val Accuracy: {scores.mean():.2f} ({scores.std():.2f} std)")

    with open(f"processed_data/X_test_{feature}.pkl", "rb") as f:
        X_test = pickle.load(f)

    model.fit(X)
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy against manual labels: {accuracy:.2f}")

features = ['pca', 'sift']

print("Starting manual data experiment")
for feature in features:
    with open(f"processed_data/X_test_{feature}.pkl", "rb") as f:
        X = pickle.load(f)

    print(f"Training with {feature} features with shape: {X.shape}")
    model = make_model()
    scores = cross_val_score(model, X, y_test, cv=5, n_jobs=-1, scoring='accuracy')

    print(f"Finished Cross-validation with with {feature} features.\nCross-val Accuracy: {scores.mean():.2f} ({scores.std():.2f} std)")

print("Starting with just flat features")

with open(f"processed_data/X_test_flat.pkl", "rb") as f:
    X_test = pickle.load(f)

model = make_model()
scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
print(f"Finished Cross-validation with with manual flat features.\nCross-val Accuracy: {scores.mean():.2f} ({scores.std():.2f} std)")
