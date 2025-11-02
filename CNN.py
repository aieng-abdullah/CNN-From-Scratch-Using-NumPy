import numpy as np

# Input Layer 
class input_layer:
    def __init__(self,H ,W,c):
        self.H = H
        self.W = W
        self.c = c
        
    def forward(self, B=1):
        X = np.random.rand(B,self.H, self.W, self.c)
        return X


# Convolution Operation
class convop:
    def __init__(self, F, K, in_channels, stride=1, padding=0):
        self.F = F
        self.K = K
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.W = np.random.rand(F, F, in_channels, K)
        self.b = np.zeros((1,1,1,K))

    def forward(self, X):
        self.X = X
        B, H, W, C = X.shape
        if self.padding > 0:
            self.X_padded = np.pad(X, ((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)), 'constant')
        else:
            self.X_padded = X

        H_out = (H + 2*self.padding - self.F) // self.stride + 1
        W_out = (W + 2*self.padding - self.F) // self.stride + 1

        Y = np.zeros((B, H_out, W_out, self.K))
        for b in range(B):
            for i in range(H_out):
                for j in range(W_out):
                    for k in range(self.K):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        window = self.X_padded[b, h_start:h_start+self.F, w_start:w_start+self.F, :]
                        Y[b, i, j, k] = np.sum(window * self.W[:, :, :, k]) + self.b[0,0,0,k] 
        return Y

    def backward(self, dY, lr=0.001):
        B, H_out, W_out, K = dY.shape
        dX_padded = np.zeros_like(self.X_padded)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        for b in range(B):
            for i in range(H_out):
                for j in range(W_out):
                    for k in range(K):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        window = self.X_padded[b, h_start:h_start+self.F, w_start:w_start+self.F, :]
                        dW[:, :, :, k] += window * dY[b, i, j, k]
                        db[:, :, :, k] += dY[b, i, j, k]
                        dX_padded[b, h_start:h_start+self.F, w_start:w_start+self.F, :] += self.W[:, :, :, k] * dY[b, i, j, k]

        if self.padding > 0:
            dX = dX_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            dX = dX_padded

        self.W -= lr * dW / B
        self.b -= lr * db / B
        return dX


# Activation function ReLU
class ReLU:
    def forward(self,X):
        self.mask = X > 0
        A = np.maximum(0, X)
        return A
    def backward(self, dA):
        return dA * self.mask


# Pooling Layer
class Pooling:
    def __init__(self, Fp=2, Sp=2):
        self.Fp = Fp
        self.Sp = Sp

    def forward(self, A):
        self.A = A
        B,H, W, K = A.shape
        H_out = (H - self.Fp) // self.Sp + 1
        W_out = (W - self.Fp) // self.Sp + 1
        P = np.zeros((B, H_out, W_out, K))
        self.max_indices = {}

        for b in range(B):
            for i in range(H_out):
                for j in range(W_out):
                    for k in range(K):
                        window = A[b, i*self.Sp:i*self.Sp+self.Fp, j*self.Sp:j*self.Sp+self.Fp, k]
                        P[b, i, j, k] = np.max(window)
                        # store index of max for backward
                        self.max_indices[(b,i,j,k)] = np.unravel_index(np.argmax(window), window.shape)
        return P

    def backward(self, dP):
        B,H_out,W_out,K = dP.shape
        dA = np.zeros_like(self.A)

        for b in range(B):
            for i in range(H_out):
                for j in range(W_out):
                    for k in range(K):
                        idx = self.max_indices[(b,i,j,k)]
                        dA[b, i*self.Sp + idx[0], j*self.Sp + idx[1], k] = dP[b,i,j,k]
        return dA


# Flatten layer
class flatten:
    def forward(self, P):
        self.orig_shape = P.shape
        B = P.shape[0]
        Z = P.reshape(B, -1)
        return Z
    def backward(self, dZ):
        return dZ.reshape(self.orig_shape)


# Fully connected layer
class FullyConnectedLayer:
    def __init__(self, n_input , n_output):
        self.W = np.random.randn(n_input, n_output) * 0.01
        self.b = np.zeros((1, n_output))

    def forward(self, X):
         self.X = X
         Z = np.dot(X, self.W) + self.b
         return Z
    
    def backward(self, dz , lr=0.001):
        B = self.X.shape[0]
        self.dw = np.dot(self.X.T, dz) / B
        self.db = np.sum(dz, axis=0, keepdims=True) / B
        dx = np.dot(dz, self.W.T)
        self.W -= lr* self.dw
        self.b -= lr* self.db
        return dx


# SoftMax
class SoftMax:
    def forward(self, Z):
        e_x = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        self.out = e_x / np.sum(e_x, axis=1, keepdims=True)
        return self.out
    def backward(self, Y):
        # Y is one-hot labels
        return (self.out - Y) / Y.shape[0]


# Loss Function
class lossFunction:
    def forward(self,Y,Y_hat):
        L = -np.mean(np.sum(Y * np.log(Y_hat + 1e-9), axis=1))
        return L



#Optimization
class SGD:
    def __init__(self, layer_params, lr=0.01):
        self.layer_params = layer_params
        self.lr = lr

    def step(self):
        for p in self.layer_params:
            layer = p['layer']
            if hasattr(layer, 'W') and hasattr(layer, 'dw'):
                layer.W -= self.lr * layer.dw
            if hasattr(layer, 'b') and hasattr(layer, 'db'):
                layer.b -= self.lr * layer.db

       



#Test
B = 5
input_layer_obj = input_layer(H=28, W=28, c=1)
input_data = input_layer_obj.forward(B=B)

conv_layer = convop(F=3, K=8, in_channels=input_data.shape[-1])
conv_output = conv_layer.forward(input_data)

relu_layer = ReLU()
relu_output = relu_layer.forward(conv_output)

pool_layer = Pooling(Fp=2, Sp=2)
pooled_output = pool_layer.forward(relu_output)

flatten_layer = flatten()
flattened_output = flatten_layer.forward(pooled_output)

fc_layer = FullyConnectedLayer(n_input=flattened_output.shape[1], n_output=10)
fc_linear = fc_layer.forward(flattened_output)

softmax_layer = SoftMax()
softmax_output = softmax_layer.forward(fc_linear)

print(softmax_output)
