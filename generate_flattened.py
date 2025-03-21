import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler

tensor_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

DATA_DIR = "../data"
RNG_SEED = 42
BATCH_SIZE = 8
BOVW_CLUSTERS = 500

def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), list(labels)

tensor_train = datasets.ImageFolder(f"{DATA_DIR}/train", transform=tensor_transform)
tensor_test = datasets.ImageFolder(f"{DATA_DIR}/test", transform=tensor_transform)

tensor_loader_train = DataLoader(tensor_train, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
tensor_loader_test = DataLoader(tensor_test, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)

X = []
for images, labels in tqdm(tensor_loader_train):
    # Flatten images to shape
    images_flat = [img.numpy().transpose(1, 2, 0).flatten() for img in images]
    X.extend(images_flat)
    
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

np.save("X_train_flat.npy", X_scaled)

X_test = []
for images, labels in tqdm(tensor_loader_test):
    # Flatten images to shape
    images_flat = [img.numpy().transpose(1, 2, 0).flatten() for img in images]
    X_test.extend(images_flat)
    
scaler_test = StandardScaler()
X_scaled_test = scaler_test.fit_transform(X_test)

np.save("X_test_flat.npy", X_scaled_test)