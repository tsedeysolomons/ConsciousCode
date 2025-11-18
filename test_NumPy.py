import numpy as np

print("CONSCIOUSCODE - NUMPY BRAIN SIMULATION TEST")
print("=" * 50)

# 1. Create artificial neurons (arrays)
print("1. CREATING ARTIFICIAL NEURONS:")
inputs = np.array([0.2, 0.8, 0.5])  # Like sensory inputs
weights = np.array([0.7, 0.1, 0.3]) # Like synaptic strengths
bias = 0.1

print(f"   Inputs: {inputs}")
print(f"   Weights: {weights}")
print(f"   Bias: {bias}")

# 2. Simulate a single neuron calculation
print("\n2. SINGLE NEURON CALCULATION:")
neuron_output = np.dot(inputs, weights) + bias  # .dot Matrix multiplication (how neurons connect)
print(f"   Neuron output: {neuron_output:.3f}") # np.dot() = How neurons talk to each other


# 3. Create multiple neurons (neural layer)
print("\n3. NEURAL LAYER SIMULATION:")
layer_inputs = np.array([
    [0.1, 0.9],  # Sample 1
    [0.8, 0.2],  # Sample 2  
    [0.4, 0.6]   # Sample 3
])
layer_weights = np.array([
    [0.5, 0.3, 0.1],  # Neuron 1 weights
    [0.2, 0.8, 0.4]   # Neuron 2 weights
])

print(f"   Input shape: {layer_inputs.shape} (3 samples, 2 features)")
print(f"   Weights shape: {layer_weights.shape} (2 inputs, 3 neurons)")
print(f"   Matrix multiplication possible: {layer_inputs.shape[1] == layer_weights.shape[0]}")

# 4. Neural layer activation
layer_output = np.dot(layer_inputs, layer_weights)
print(f"   Layer output shape: {layer_output.shape}")
print(f"   Layer output:\n{layer_output}")

# 5. Apply activation function (sigmoid)
print("\n4. ACTIVATION FUNCTION (SIGMOID):")
activated_output = 1 / (1 + np.exp(-layer_output))  #np.dot() = How neurons talk to each other
print(f"   Activated output:\n{activated_output}")

# 6. Basic operations
print("\n5. BASIC NEURAL OPERATIONS:")
print(f"   Max activation: {np.max(activated_output):.3f}")
print(f"   Min activation: {np.min(activated_output):.3f}") 
print(f"   Mean activation: {np.mean(activated_output):.3f}")

print("\n NUMPY TEST SUCCESSFUL!")
print("   Your brain simulation tools are working!")

# Real Use: Create blank neural layers or bias terms
biases = np.zeros((1, 5))    # 5 neurons, no initial bias
weights = np.ones((3, 2))    # All connections start at strength 1

print(f"\n   Blank biases shape: {biases}") # with out shape the out  Blank biases shape: [[0. 0. 0. 0. 0.]]
print(f"  \n Initial weights shape: {weights.shape}") # with ih .shape the out  Initial weights shape: (3, 2)
print(f"  \n Initial weights : {weights}") # Initial weights shape: [[1. 1.]
                                                                     # [1. 1.]
                                                                     # [1. 1.]]

# Real Use: Initialize random synaptic strengths
np.random.seed(42)  # For reproducible results
synaptic_weights = np.random.randn(4, 3) * 0.1  # Small random values 4(down) and 3(across/side (wedemon))
print(f"Random neural connections:\n{synaptic_weights}")                                                                     