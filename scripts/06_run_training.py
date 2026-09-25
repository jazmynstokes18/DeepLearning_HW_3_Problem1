# Step 6 — Train for 20 epochs with SGD(lr=0.1); keep the returned history.
# Depends on `model`, `criterion`, `device` (Step 4), `train_loader`, `valid_loader`
# (Step 3), and `train2`/`evaluate_tm` (Step 5). This is the long-running cell —
# expect it to take a few minutes.

n_epochs = 20

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# The metric must live on the same device as the tensors it consumes, otherwise
# torchmetrics raises a device-mismatch error.
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

history = train2(model, optimizer, criterion, accuracy,
                  train_loader, valid_loader, n_epochs)
