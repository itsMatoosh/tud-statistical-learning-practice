import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

RNG=42
features = ['pca', 'sift', 'flat']

y_train = np.load("processed_data/y_train.npy")
y_test = np.load("processed_data/y_test.npy")

def make_model(): return LogisticRegression(random_state=RNG, max_iter=1000)

for feature in features:
    with open(f"processed_data/X_train_{feature}.pkl", "rb") as f:
        X = pickle.load(f)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    print(f"Training with {feature} features with shape: {X.shape}")
    model = make_model()
    scores = cross_val_score(model, X, y_train, cv=5)

    print(f"Finished Cross-validation with with {feature} features.\nCross-val Accuracy: {scores.mean():.2f} ({scores.std():.2f} std)")
    model.fit(X, y_train)

    with open(f"processed_data/X_test_{feature}.pkl", "rb") as f:
        X_test = pickle.load(f)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test set accuracy: {accuracy:.2f}")


# Flat
# test_model = make_model()
