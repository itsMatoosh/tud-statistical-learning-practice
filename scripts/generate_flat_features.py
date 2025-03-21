import numpy as np
import pickle

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler

tensor_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

DATA_DIR = "data"
RNG_SEED = 42
BATCH_SIZE = 8

def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)

tensor_train = datasets.ImageFolder(f"{DATA_DIR}/train", transform=tensor_transform)

tensor_loader_train = DataLoader(tensor_train, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

cutoff_point = len(tensor_train) // 7

idx = 0
X = []
for images, _ in tqdm(tensor_loader_train):
    # Flatten images to shape
    print(images[0].shape)
    images_flat = [img.numpy().transpose(1, 2, 0).flatten() for img in images]
    images_flat = scaler.transform(images_flat)
    print(len(images_flat) len(images_flat[0]), np.array(images_flat).shape())

    break

    X.extend(images_flat)

    if len(X) >= cutoff_point:
        to_write, X = X[:cutoff_point], X[cutoff_point:]
        np.save(f"X_train_flat_{idx}.npy", to_write)
        idx += 1

if len(X) > 0:
    np.save(f"X_train_flat.npy", X)
