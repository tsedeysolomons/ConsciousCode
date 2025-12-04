import numpy as np

print("🧠 CONSCIOUSCODE - NEURAL BUILDING BLOCKS")
print("=" * 50)

class NeuralComponents:
    """Core components for building neural networks"""
    
    @staticmethod
    def create_layer_weights(input_size, output_size, method='xavier'):
        """Initialize weights for a neural layer"""
        if method == 'xavier':
            # Xavier/Glorot initialization - good for sigmoid/tanh
            std = np.sqrt(2.0 / (input_size + output_size))
        elif method == 'he':
            # He initialization - good for ReLU
            std = np.sqrt(2.0 / input_size)
        else:
            std = 0.1  # Simple small random
            
        weights = np.random.randn(input_size, output_size) * std
        return weights
    
    @staticmethod
    def create_biases(output_size):
        """Initialize biases for a layer"""
        return np.zeros((1, output_size))
    
    @staticmethod
    def forward_pass(inputs, weights, biases, activation='sigmoid'):
        """Single layer forward pass"""
        # Linear transformation: inputs × weights + biases
        z = np.dot(inputs, weights) + biases
        
        # Apply activation function
        if activation == 'sigmoid':
            a = 1 / (1 + np.exp(-z))
        elif activation == 'relu':
            a = np.maximum(0, z)
        elif activation == 'tanh':
            a = np.tanh(z)
        else:
            a = z  # Linear (no activation)
            
        return a, z  # Return both activation and pre-activation
    
    @staticmethod
    def calculate_loss(y_true, y_pred, method='mse'):
        """Calculate loss between predictions and true values"""
        if method == 'mse':
            # Mean Squared Error
            return np.mean((y_true - y_pred) ** 2)
        elif method == 'binary_crossentropy':
            # For binary classification
            epsilon = 1e-15  # Avoid log(0)
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

print("1. TESTING NEURAL COMPONENTS")
print("-" * 30)

# Create a mini neural network
inputs = np.array([[0.1, 0.9], [0.8, 0.2]])
print(f"Input data:\n{inputs}")
print(f"Input shape: {inputs.shape}")

# Initialize weights and biases
weights = NeuralComponents.create_layer_weights(2, 3, 'xavier')
biases = NeuralComponents.create_biases(3)

print(f"\nWeights:\n{weights}")
print(f"Biases: {biases}")

# Forward pass through the layer
activations, pre_activation = NeuralComponents.forward_pass(
    inputs, weights, biases, 'sigmoid'
)

print(f"\nPre-activation (weighted sum):\n{pre_activation}")
print(f"Post-activation (neuron output):\n{activations}")

print("\n2. BUILDING A COMPLETE NETWORK")
print("-" * 30)

class SimpleNetwork:
    """A simple 2-layer neural network"""
    
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = NeuralComponents.create_layer_weights(input_size, hidden_size)
        self.biases1 = NeuralComponents.create_biases(hidden_size)
        self.weights2 = NeuralComponents.create_layer_weights(hidden_size, output_size)
        self.biases2 = NeuralComponents.create_biases(output_size)
        
    def predict(self, X):
        """Forward pass through the entire network"""
        # Input → Hidden layer
        hidden_act, _ = NeuralComponents.forward_pass(X, self.weights1, self.biases1, 'sigmoid')
        
        # Hidden → Output layer  
        output_act, _ = NeuralComponents.forward_pass(hidden_act, self.weights2, self.biases2, 'sigmoid')
        
        return output_act

# Create and test our network
print("Creating a 2-4-1 neural network (like for XOR problem):")
network = SimpleNetwork(2, 4, 1)

# Test with some data
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = network.predict(test_data)

print(f"\nTest data:\n{test_data}")
print(f"Network predictions:\n{predictions}")

print("\n3. VISUALIZING THE NETWORK STATE")
print("-" * 30)

print("Network Architecture:")
print(f"  Input → Hidden: {network.weights1.shape} weights")
print(f"  Hidden biases: {network.biases1.shape}")
print(f"  Hidden → Output: {network.weights2.shape} weights") 
print(f"  Output biases: {network.biases2.shape}")

print(f"\nTotal parameters: {network.weights1.size + network.biases1.size + network.weights2.size + network.biases2.size}")

print("\n🎯 DAY 2 AFTERNOON COMPLETE!")
print("   You've built the core components of a neural network!")
print("   Ready for Day 3: Activation functions and their derivatives!")