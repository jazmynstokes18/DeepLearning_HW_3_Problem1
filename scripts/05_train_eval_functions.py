# Step 5 — Training/evaluation helper functions only. Nothing is executed here;
# no optimizer or training run exists yet, so this cell just defines the two
# functions used by the next step.

def evaluate_tm(model, data_loader, metric):
    """Run `model` over every batch in `data_loader` under no_grad and return the
    aggregated value of `metric` (a torchmetrics metric object)."""
    model.eval()           # eval mode: disables dropout / uses running BN stats
    metric.reset()         # torchmetrics accumulates internally; clear stale state
    with torch.no_grad():  # no autograd graph needed for evaluation
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)
    return metric.compute()


def train2(model, optimizer, criterion, metric, train_loader, valid_loader, n_epochs):
    """Train `model` for `n_epochs`, printing loss/accuracy each epoch, and return
    a history dict with keys 'train_losses', 'train_metrics', 'valid_metrics'
    (one entry per epoch) so the caller can plot learning curves afterward."""
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}

    for epoch in range(n_epochs):
        model.train()
        metric.reset()
        running_loss, n_seen = 0.0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            n_seen += X_batch.size(0)
            metric.update(y_pred, y_batch)

        train_loss = running_loss / n_seen
        train_metric = metric.compute().item()
        valid_metric = evaluate_tm(model, valid_loader, metric).item()

        history["train_losses"].append(train_loss)
        history["train_metrics"].append(train_metric)
        history["valid_metrics"].append(valid_metric)

        print(f"epoch {epoch + 1:2d}/{n_epochs} | "
              f"train loss: {train_loss:.4f} | "
              f"train acc: {train_metric:.4f} | "
              f"valid acc: {valid_metric:.4f}")

    return history


print("evaluate_tm() and train2() defined. Nothing executed yet.")
