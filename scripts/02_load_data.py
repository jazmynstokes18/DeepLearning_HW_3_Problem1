# Step 2 — Load Fashion MNIST, scale to [0, 1], re-seed, split 55k/5k train/validation.
# Depends on `torch`, `SEED` from the setup cell above.

import torchvision
import torchvision.transforms.v2 as T

# ToImage() gives a tensor in [C, H, W] layout; ToDtype(..., scale=True) casts the
# uint8 pixels to float32 and divides by 255, putting every input in [0, 1].
to_float_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

train_and_valid_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=True, download=True, transform=to_float_tensor)
test_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=False, download=True, transform=to_float_tensor)

# Human-readable label names, in class-index order (0-9).
class_names = train_and_valid_data.classes

# Re-seed right before the split (as requested) so the 55k/5k partition is
# reproducible regardless of what ran earlier in the session.
torch.manual_seed(SEED)
train_data, valid_data = torch.utils.data.random_split(
    train_and_valid_data, [55000, 5000])

print(f"train      : {len(train_data):,} images")
print(f"validation : {len(valid_data):,} images")
print(f"test       : {len(test_data):,} images")
print(f"class names: {class_names}")
