# Step 4 — Define the ImageClassifier MLP, move it to device, create the loss.
# Depends on `nn`, `device` from Step 1.

class ImageClassifier(nn.Module):
    """Fully connected classifier for 28x28 grayscale Fashion MNIST images.

    Architecture: Flatten -> Linear(784,300) -> ReLU -> Linear(300,100) -> ReLU
    -> Linear(100,10). The final layer outputs raw logits (no activation), since
    nn.CrossEntropyLoss expects logits and applies log-softmax internally.
    """

    def __init__(self, n_inputs=784, n_hidden1=300, n_hidden2=100, n_classes=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),                          # [N, 1, 28, 28] -> [N, 784]
            nn.Linear(n_inputs, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_classes),        # raw logits, no activation
        )

    def forward(self, X):
        return self.mlp(X)


torch.manual_seed(SEED)  # reproducible weight initialization
model = ImageClassifier(784, 300, 100, 10).to(device)

# Standard multi-class classification loss: expects raw logits + integer class
# labels, and applies log-softmax internally.
criterion = nn.CrossEntropyLoss()

n_params = sum(p.numel() for p in model.parameters())
print(model)
print(f"\ntotal trainable parameters: {n_params:,}")
