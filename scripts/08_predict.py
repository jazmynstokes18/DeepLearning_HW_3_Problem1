# Step 8 — Predict on 3 validation images, show softmax probabilities + top-4,
# then display the images with their predicted/true labels.
# Depends on `model`, `device`, `valid_loader`, `class_names` from earlier steps.

model.eval()

# Take the first batch from the validation loader and predict on its first 3
# images.
X_new, y_new = next(iter(valid_loader))
X_new = X_new[:3].to(device)
y_new = y_new[:3]

with torch.no_grad():
    y_pred_logits = model(X_new)

y_pred = y_pred_logits.argmax(dim=1)  # index of the largest logit per image

print("predicted :", [class_names[i] for i in y_pred])
print("actual    :", [class_names[i] for i in y_new])
print("correct   :", (y_pred.cpu() == y_new).tolist())

# The model outputs logits; softmax turns them into class probabilities.
y_proba = F.softmax(y_pred_logits, dim=1)

# round(decimals=...) is not implemented on the "mps" backend, so move to cpu
# first whenever that's the active device (a no-op cost on cuda/cpu).
if device == "mps":
    y_proba = y_proba.cpu()

print("\nclass probabilities (rounded):")
print(y_proba.round(decimals=3))

# Top-4 classes per image, re-softmaxed over just those 4 logits so the reported
# values sum to 1 per image.
y_top4_values, y_top4_indices = torch.topk(y_pred_logits, k=4, dim=1)
y_top4_probas = F.softmax(y_top4_values, dim=1)
if device == "mps":
    y_top4_probas = y_top4_probas.cpu()

print("\ntop-4 probabilities:")
print(y_top4_probas.round(decimals=3))
print("\ntop-4 class indices:")
print(y_top4_indices)


# Display the 3 validation images with their predicted vs. true class names.
fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
for ax, image, pred, true in zip(axes, X_new.cpu(), y_pred.cpu(), y_new):
    ax.imshow(image.squeeze(), cmap="binary")  # squeeze drops the channel axis
    ax.set_title(f"pred: {class_names[pred]}\ntrue: {class_names[true]}",
                 fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.show()
