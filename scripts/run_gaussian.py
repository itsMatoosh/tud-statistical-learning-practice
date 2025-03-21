import numpy as np
import pickle

from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score

RNG=42
features = ['pca', 'sift']

y_train = np.load("processed_data/y_train.npy")
y_test = np.load("processed_data/y_test.npy")

n_classes = len(np.unique(y_test))

def make_model(): return GaussianMixture(n_components=n_classes, covariance_type="full", random_state=RNG)

print("starting experiment")
for feature in features:
    with open(f"processed_data/X_train_{feature}.pkl", "rb") as f:
        X = pickle.load(f)

    print(f"Training with {feature} features with shape: {X.shape}")
    model = make_model()
    scores = cross_val_score(model, X, y_train, cv=5)

    print(f"Finished Cross-validation with with {feature} features.\nCross-val Accuracy: {scores.mean():.2f} ({scores.std():.2f} std)")
    model.fit(X, y_train)

    with open(f"processed_data/X_test_{feature}.pkl", "rb") as f:
        X_test = pickle.load(f)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

# Flat
with open(f"processed_data/X_train_flat.pkl", "rb") as f:
    X = pickle.load(f)

print(f"Training with flat features with shape: {X.shape}")
model = make_model()
scores = cross_val_score(model, X, y_train, cv=5, verbose=1)

print(f"Finished Cross-validation with with {feature} features.\nCross-val Accuracy: {scores.mean():.2f} ({scores.std():.2f} std)")
model.fit(X, y_train)

with open(f"processed_data/X_test_{feature}.pkl", "rb") as f:
    X_test = pickle.load(f)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
