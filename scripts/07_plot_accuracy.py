# Step 7 — Plot training vs. validation accuracy (the assignment's required
# addition), print the final numbers, and plot training loss as a cross-check.
# Depends on `history`, `n_epochs` from Step 6.

epochs = np.arange(n_epochs)

# Training accuracy is a running average over the epoch, so it belongs at the
# epoch's midpoint (epoch + 0.5). Validation accuracy is measured once, at the
# end of the epoch, so it sits at epoch + 1.0.
plt.figure()
plt.plot(epochs + 0.5, history["train_metrics"], ".--", label="Training accuracy")
plt.plot(epochs + 1.0, history["valid_metrics"], ".-", label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Fashion MNIST classifier - learning curves")
plt.grid(True)
plt.legend()
plt.axis([0.5, n_epochs, 0.7, 1.0])
plt.show()

print(f"final training accuracy   : {history['train_metrics'][-1]:.4f}")
print(f"final validation accuracy : {history['valid_metrics'][-1]:.4f}")

# Second figure: training loss per epoch, as a cross-check. Accuracy rising while
# loss falls is the expected pattern for a healthy training run.
plt.figure()
plt.plot(epochs + 0.5, history["train_losses"], ".--", color="tab:red",
         label="Training loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("Fashion MNIST classifier - training loss")
plt.grid(True)
plt.legend()
plt.show()

print(f"final training loss       : {history['train_losses'][-1]:.4f}")
