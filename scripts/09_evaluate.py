# Step 9 — Final, single evaluation on the held-out test set.
# Depends on `model`, `test_loader`, `accuracy` from earlier steps. This is the
# only place test_loader is used.

test_accuracy = evaluate_tm(model, test_loader, accuracy).item()
n_params_final = sum(p.numel() for p in model.parameters())

print(f"test accuracy      : {test_accuracy:.4f}")
print(f"model parameters   : {n_params_final:,}")
