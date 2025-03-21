import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_DIR = "data"
RNG_SEED = 42
BATCH_SIZE = 8

transform = transforms.Compose([
    transforms.Resize((160, 160)),
])

def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)

base_train = datasets.ImageFolder(f"{DATA_DIR}/train", transform=transform)
# base_test = datasets.ImageFolder(f"{DATA_DIR}/test", transform=transform)

print("dataset folds loaded")

def preprocess_data(loader):
    X = []
    y = []
    for images, labels in tqdm(loader, desc="Flattening data"):
        # Convert images to numpy arrays and flatten
        images_flat = [np.array(img).flatten() for img in images]
        X.extend(images_flat)
        y.extend(labels)
    return np.array(X), np.array(y)

base_loader_train = DataLoader(base_train, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
# base_loader_test = DataLoader(base_test, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)

print("dataset loaders created")

X_train, y_train = preprocess_data(base_loader_train)
# X_test, y_test = preprocess_data(base_loader_test)

print("preprocessing done")

# Initialize the scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.fit_transform(X_test)

print("scalar done")

# Perform PCA
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train_scaled)
np.save("x_train_pca.npy", X_train_pca)
np.save("y_train.npy", y_train)
print("x_train_pca.npy and labels written to disk")

# X_test_pca = pca.fit_transform(X_test_scaled)
# np.save("x_test_pca.npy", X_test_pca)
# np.save("y_test.npy", y_test)
# print("x_test_pca.npy and labels written to disk")

explained_variance_ratio = np.array(pca.explained_variance_ratio_)

cumulative_variance = np.cumsum(explained_variance_ratio)

print(f"Cumulative variance is: {cumulative_variance}")
