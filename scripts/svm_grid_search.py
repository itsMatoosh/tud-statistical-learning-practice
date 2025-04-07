import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle


x_train = np.load(f"processed_data/X_train_pca.npy")
x_test = np.load(f"processed_data/X_test_pca.npy")

y_train = np.load("processed_data/y_train.npy")
y_test = np.load("processed_data/y_test.npy")


model = SVC(cache_size=1000, max_iter=5000)

param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf", "poly"],
    "gamma": ["scale", "auto"],
}

print("starting grid search")

grid_search = GridSearchCV(
    model, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=True
)
grid_search.fit(x_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)

with open("svm-grid_search5000.pkl", "wb") as f:
    pickle.dump(grid_search, f)
