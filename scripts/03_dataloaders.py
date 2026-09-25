# Step 3 — DataLoaders for train/validation/test, then inspect one sample.
# Depends on `train_data`, `valid_data`, `test_data`, `class_names`, `SEED` from
# Step 2.

from torch.utils.data import DataLoader

torch.manual_seed(SEED)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Each dataset entry is an (image, label) tuple; look at the very first one.
X_sample, y_sample = train_data[0]
print("sample image shape :", tuple(X_sample.shape))   # [channels, rows, cols]
print("sample image dtype :", X_sample.dtype)
print("sample label       :", y_sample, "->", class_names[y_sample])
