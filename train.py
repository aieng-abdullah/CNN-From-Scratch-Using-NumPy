import numpy as np
from mnist_preprocess import load_data
from CNN import convop, ReLU, Pooling, flatten, FullyConnectedLayer, SoftMax, lossFunction, SGD


# Load MNIST data
x_train, y_train, x_test, y_test = load_data()

# Initialize layers
conv = convop(F=3, K=8, in_channels=1)
relu = ReLU()
pool = Pooling(Fp=2, Sp=2)
flat = flatten()
fc = FullyConnectedLayer(n_input=13*13*8, n_output=10)  # 28x28 -> conv(3) -> pool(2)
softmax = SoftMax()
loss_fn = lossFunction()

# Training parameters
epochs = 10
batch_size = 64
lr = 0.1

# Init optimizer 
optimizer = SGD(layer_params=[{'layer': conv}, {'layer': fc}], lr=lr)

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    # Shuffle data
    perm = np.random.permutation(x_train.shape[0])
    x_train_shuffled = x_train[perm]
    y_train_shuffled = y_train[perm]

    total_loss = 0
    num_batches = x_train.shape[0] // batch_size

    for b in range(num_batches):
        start = b * batch_size
        end = start + batch_size
        X_batch = x_train_shuffled[start:end]
        Y_batch = y_train_shuffled[start:end]

        # Forward pass
        out = conv.forward(X_batch)
        out = relu.forward(out)
        out = pool.forward(out)
        out = flat.forward(out)
        out = fc.forward(out)
        out = softmax.forward(out)

        # Compute loss
        loss = loss_fn.forward(Y_batch, out)
        total_loss += loss

        # Backward pass 
        dZ = softmax.backward(Y_batch)
        dZ = fc.backward(dZ)
        dZ = flat.backward(dZ)
        dZ = pool.backward(dZ)
        dZ = relu.backward(dZ)
        dZ = conv.backward(dZ)

        # Update weights
        optimizer.step()

        if (b + 1) % 100 == 0:
            print(f"Batch {b + 1}/{num_batches}, Loss: {loss:.4f}")

    print(f"Epoch {epoch + 1} average loss: {total_loss / num_batches:.4f}")

# Evaluate 
out = conv.forward(x_test)
out = relu.forward(out)
out = pool.forward(out)
out = flat.forward(out)
out = fc.forward(out)
out = softmax.forward(out)

preds = np.argmax(out, axis=1)
labels = np.argmax(y_test, axis=1)
accuracy = np.mean(preds == labels)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
