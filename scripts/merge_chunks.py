import pickle
import numpy as np

features = ['test_flat']

for feature in features:
    file_name = f"processed_data/X_{feature}"

    X = np.load(f"{file_name}.npy")

    with open(f"{file_name}.pkl", "wb") as f:
        pickle.dump(X, f)
