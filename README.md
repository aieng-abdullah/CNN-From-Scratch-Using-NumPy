# CNN From Scratch Using NumPy

A complete, from-first-principles implementation of a **Convolutional Neural Network (CNN)** written entirely in **NumPy**.  
No TensorFlow. No PyTorch. No Keras. Just raw math and matrix operations.

This project reconstructs CNN theory into a fully functional handwritten digit recognizer trained on the **MNIST dataset**, achieving **~85.40% accuracy** after 10 epochs.  
It is both a research exercise and a technical demonstration of deep learning fundamentals.

---

## 1. Overview

This project implements every essential component of a CNN — convolution, pooling, activation, flattening, fully connected layers, backpropagation, and optimization — **manually**.

The objective:  
Understand how each mathematical operation transforms data, how gradients propagate, and how deep learning *actually* learns without relying on black-box libraries.

---

## 2. Motivation

Deep learning frameworks are powerful, but they hide the math that drives them.  
This project strips away all abstraction to reveal the **core mechanics** beneath the surface.

Key goals:

- Rebuild convolutional networks using only NumPy arrays and matrix algebra.  
- Trace gradients through every layer manually.  
- Develop a rigorous, experiment-driven understanding of CNNs.  
- Build an appreciation for the complexity automated by modern frameworks.

This project is not about convenience — it’s about comprehension.

---

## 3. Architecture

Input (28×28×1)
↓
Convolution Layer (3×3 kernels, 8 filters)
↓
ReLU Activation
↓
MaxPooling Layer (2×2)
↓
Flatten Layer
↓
Fully Connected Layer (13×13×8 → 10)
↓
Softmax
↓
Output (10 classes)



A minimalist yet complete CNN design suitable for grayscale images such as MNIST.

---

## 4. Mathematical Foundations

Every component of this model was built from its mathematical definition.

### 4.1 Convolution
The convolution operation computes a weighted sum of local regions in the input:

\[
S_{i,j}^{(k)} = (X * W^{(k)})_{i,j} + b^{(k)} = \sum_m \sum_n \sum_c X_{i+m, j+n, c} \cdot W^{(k)}_{m,n,c} + b^{(k)}
\]

- \(X\) = input tensor  
- \(W^{(k)}\) = kernel weights for the \(k\)-th filter  
- \(b^{(k)}\) = bias  
- \(*\) denotes convolution  

Stride and zero padding are applied to control output size.

### 4.2 ReLU Activation
The Rectified Linear Unit introduces non-linearity:

\[
f(x) = \max(0, x)
\]

Derivative for backpropagation:

\[
f'(x) = 
\begin{cases} 
1 & x > 0 \\
0 & x \le 0 
\end{cases}
\]

### 4.3 Max Pooling
Downsamples the feature map:

\[
Y_{i,j,c} = \max_{(m,n) \in R} X_{s \cdot i + m, s \cdot j + n, c}
\]

- \(R\) = pooling region  
- \(s\) = stride  

Gradients propagate only through the maximum value indices.

### 4.4 Flatten
Reshapes the 3D feature maps into a vector:

\[
\text{flatten}(X) \in \mathbb{R}^{H \cdot W \cdot C}
\]

### 4.5 Fully Connected Layer
Performs affine transformation:

\[
Z = W \cdot X + b
\]

- \(X\) = input vector  
- \(W\) = weight matrix  
- \(b\) = bias vector  

### 4.6 Softmax & Cross-Entropy Loss
Softmax converts logits into probabilities:

\[
\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

Cross-entropy loss for multi-class classification:

\[
L = -\sum_i y_i \log(\hat{y}_i)
\]

Gradients propagate using the chain rule:

\[
\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i
\]

---

## 5. Implementation Details

The network is modular, composed of self-contained classes for each operation:

| Module | Description |
|---------|-------------|
| `convop` | Performs convolution and computes weight gradients. |
| `ReLU` | Applies activation and stores masks for backprop. |
| `Pooling` | Implements max pooling and its backward pass. |
| `Flatten` | Reshapes tensor outputs for dense layers. |
| `FullyConnectedLayer` | Handles linear transformations and updates. |
| `SoftMax` | Converts final logits to probability distributions. |
| `lossFunction` | Computes categorical cross-entropy loss. |
| `SGD` | Optimizer implementing Stochastic Gradient Descent. |

Each layer defines `forward()` and `backward()` methods for seamless pipeline integration.

---

## 6. Dataset Preparation

Dataset: **MNIST Handwritten Digits**

Preparation steps:

- Normalized pixel values to [0, 1].  
- Expanded each image to include a channel dimension (28×28×1).  
- One-hot encoded labels for classification.  
- Optionally reduced dataset size during experimentation for computational efficiency.

---

## 7. Training Process

**Training configuration:**

- Epochs: 10  
- Batch Size: 64  
- Learning Rate: 0.1  
- Optimizer: SGD  
- Loss: Cross-Entropy

**Training phases:**

1. **Forward Pass**: data flows through convolution → activation → pooling → flatten → dense → softmax.
2. **Backward Pass**: gradients propagate from softmax to convolution; weights updated manually via SGD.

---

## 8. Results

| Metric | Value |
|--------|--------|
| Epochs | 10 |
| Test Accuracy | 85.40% |
| Optimizer | Stochastic Gradient Descent |
| Activation Function | ReLU |
| Loss Function | Cross-Entropy |

---

## 9. Key Learnings

1. **Mathematical Clarity:** Every equation implemented reinforced the transformation of data layer by layer.  
2. **Gradient Intuition:** Verified gradients exposed the fragile balance between stability and learning.  
3. **Computational Awareness:** Manual convolution revealed why GPUs and vectorization are essential.  
4. **Appreciation for Abstraction:** Understanding the underlying mechanics gives full control.  
5. **True Understanding:** Knowing how to fix a broken gradient is more valuable than calling `.fit()`.

---

## 10. Project Structure

CNN-from-scratch/
│
├── CNN.py # Core layer implementations
├── mnist_preprocess.py # Data loading and normalization
├── train.py # Training pipeline
├── cnn_from_scratch.PNG # Output visualization
└── README.md # Documentation


---

## 11. Experimental Notes

Challenges encountered:

- Gradient dimension mismatches in convolution backprop.  
- Overflow in softmax, solved via numerical stability.  
- Vanishing updates due to improper learning rate.  
- Memory constraints from manual loops.  
- Floating-point instability corrected via normalization.

---

## 12. Future Improvements

- Add multiple convolutional layers for deeper representation learning.  
- Integrate advanced optimizers like Adam or RMSProp.  
- Implement dropout and batch normalization.  
- Visualize learned feature maps and convolutional filters.  
- Extend to CIFAR-10 or Fashion-MNIST datasets.  
- Accelerate via vectorization or GPU-compatible libraries.

---

## 13. References

- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition.*  
- Goodfellow, I., Bengio, Y., & Courville, Y. (2016). *Deep Learning.* MIT Press.  
- Stanford CS231n: *Convolutional Neural Networks for Visual Recognition.*  

---

## 14. Author

Developed by **[Abdullah Al Arif]**  
AI/ML Engineer focused on understanding neural computation through first-principles engineering.  
Passionate about building real intelligence through mathematics, not just automation.
