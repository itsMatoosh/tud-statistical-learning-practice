import numpy as np
import pickle

from tqdm import tqdm
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_predict

DATA_DIR = "processed_data"

features = [('PCA', '_pca'), ('SIFT', '_sift')]

for name, idenitifier in tqdm(features):
    X_train = np.load(f"X_train{idenitifier}.npy")
    y_train = np.load(f"y_train.npy")

    svm_model = svm.SVC(kernel='rbf', verbose=True, random_state=42)
    
    scores = cross_val_predict(svm_model, X_train, y_train, n_jobs=-1) # n_jobs=-1 to use all processors
    print(f"Cross-validation accuracy on {name} train features: {scores.mean():.2f} ({scores.std():.2f} std)")
    
    # Fit on full dataset
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    
    X_test = np.load(f"X_test{idenitifier}.npy")
    y_test = np.load(f"y_test.npy")
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test data {accuracy:.2f}")    
    
