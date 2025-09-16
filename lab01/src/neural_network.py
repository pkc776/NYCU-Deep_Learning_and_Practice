import numpy as np

class SGDOptimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, layer):
        layer.W -= self.learning_rate * layer.grad_W
        layer.b -= self.learning_rate * layer.grad_b

class Layer:
    def __init__(self, input_size, output_size, activation=None):
        self.W = np.random.randn(input_size, output_size) * 0.1
        self.b = np.zeros((1, output_size))
        self.activation = activation
        
        # Store forward pass values for backprop
        self.input = None
        self.z = None
        self.a = None

        # Store gradients
        self.grad_W = None
        self.grad_b = None
    
    def forward(self, X):
        self.input = X
        self.z = np.dot(X, self.W) + self.b
        if self.activation:
            self.a = self.activation_forward(self.z)
        else:
            self.a = self.z
        return self.a
    
    def activation_forward(self, z):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        return z
    
    def activation_derivative(self):
        if self.activation == 'sigmoid':
            return self.a * (1 - self.a)
        elif self.activation == 'relu':
            return np.where(self.z > 0, 1, 0)
        elif self.activation == 'tanh':
            return 1 - np.power(self.a, 2)
        return 1
    
    def backward(self, grad_output):
        m = self.input.shape[0]
        grad_z = grad_output * self.activation_derivative()
        
        self.grad_W = np.dot(self.input.T, grad_z) / m
        self.grad_b = np.sum(grad_z, axis=0, keepdims=True) / m
        
        grad_input = np.dot(grad_z, self.W.T)
        return grad_input

    def update(self, learning_rate):
        # gradient descent
        self.W -= learning_rate * self.grad_W
        self.b -= learning_rate * self.grad_b

class NeuralNetwork:
    def __init__(self, layer_sizes, activations, optimizer=None):
        self.layers = []
        self.optimizer = optimizer
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], activation=activations[i]))
        
        print(f"Neural Network: {' -> '.join(map(str, layer_sizes))}")
        print(f"Activations: {activations}")
        if optimizer:
            print(f"Optimizer: {type(optimizer).__name__}")
    
    def forward(self, X):
        current_output = X
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output
    
    def backward(self, y_true, y_pred):
        grad_output = 2 * (y_pred - y_true)
        
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
    
    def update(self, learning_rate=None):
        if self.optimizer:
            for layer in self.layers:
                self.optimizer.update(layer)
        else:
            if learning_rate is None:
                raise ValueError("learning_rate must be provided when no optimizer is set")
            for layer in self.layers:
                layer.update(learning_rate)

    def train(self, X, y, epochs, learning_rate=None):
        history = []
        n_samples = X.shape[0]
        
        use_sgd = self.optimizer and isinstance(self.optimizer, SGDOptimizer)
        
        for epoch in range(epochs):
            if use_sgd:
                epoch_loss = 0.0
                
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                
                for i in range(n_samples):
                    x_sample = X_shuffled[i:i+1]
                    y_sample = y_shuffled[i:i+1]
                    
                    y_pred = self.forward(x_sample)
                    loss = self.compute_loss(y_pred, y_sample)
                    epoch_loss += loss
                    self.backward(y_sample, y_pred)
                    self.update()
                
                average_loss = epoch_loss / n_samples
                history.append(average_loss)
                
                if epoch % 1000 == 0:
                    print(f'Epoch {epoch}, Loss: {average_loss}')
            else:
                y_pred = self.forward(X)
                loss = self.compute_loss(y_pred, y)
                history.append(loss)
                
                self.backward(y, y_pred)
                self.update(learning_rate)
                
                if epoch % 1000 == 0:
                    print(f'Epoch {epoch}, Loss: {loss}')
                    
        return history
    
    def compute_loss(self, y_pred, y_true):
        # MSE loss
        return np.mean((y_pred - y_true) ** 2)

    def predict(self, X):
        return self.forward(X)
