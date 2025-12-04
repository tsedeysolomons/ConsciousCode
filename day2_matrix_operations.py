import numpy as np

print("🧠 CONSCIOUSCODE - DAY 2: NEURAL MATRIX OPERATIONS")
print("=" * 60)

print("1. UNDERSTANDING NEURAL NETWORK MATHEMATICS")
print("-" * 40)

# Neural networks are essentially matrix operations
# Let's break down the core concepts:

print("\n🔹 Neural Network = Series of Matrix Multiplications")

# Example: Simple neural network with 3 layers
# Input (2 features) → Hidden (3 neurons) → Output (1 neuron)

print("\n2. CREATING NEURAL LAYERS AS MATRICES")
print("-" * 40)

# Input data: 3 samples, each with 2 features
inputs = np.array([
    [0.1, 0.9],  # Sample 1
    [0.8, 0.2],  # Sample 2
    [0.4, 0.6]   # Sample 3
])
print(f"Input data shape: {inputs.shape}")  # (3, 2) - 3 samples, 2 features

# Weights from input to hidden layer: 2 inputs → 3 neurons
weights_input_hidden = np.array([
    [0.5, 0.3, 0.1],  # Input 1 connections to 3 hidden neurons
    [0.2, 0.8, 0.4]   # Input 2 connections to 3 hidden neurons
])
print(f"Weights shape: {weights_input_hidden.shape}")  # (2, 3) - 2 inputs, 3 neurons

print("\n3. MATRIX MULTIPLICATION - NEURAL INFORMATION FLOW")
print("-" * 40)

# The magic happens here: inputs (3,2) × weights (2,3) = hidden_activations (3,3)
hidden_activations = np.dot(inputs, weights_input_hidden)
print(f"Hidden layer activations shape: {hidden_activations.shape}")
print(f"Hidden activations:\n{hidden_activations}")

print("\n Understanding the dimensions:")
print(f"   Inputs: {inputs.shape} = (samples, features)")
print(f"   Weights: {weights_input_hidden.shape} = (features, neurons)")
print(f"   Output: {hidden_activations.shape} = (samples, neurons)")

print("\n4. ADDING BIAS - NEURAL ACTIVATION THRESHOLD")
print("-" * 40)

# Each neuron has a bias (activation threshold)
biases_hidden = np.array([0.1, 0.2, 0.3])  # One bias per hidden neuron
print(f"Biases: {biases_hidden}")

# Add biases to activations
hidden_with_bias = hidden_activations + biases_hidden
print(f"Hidden with bias:\n{hidden_with_bias}")

print("\n5. ACTIVATION FUNCTION - NEURAL FIRING DECISION")
print("-" * 40)

# Apply sigmoid activation (neurons decide whether to fire)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

hidden_output = sigmoid(hidden_with_bias)
print(f"Hidden layer output (after activation):\n{hidden_output}")

print("\n6. COMPLETE FORWARD PASS SIMULATION")
print("-" * 40)

# Now from hidden layer to output layer
weights_hidden_output = np.array([
    [0.7],  # Hidden neuron 1 → Output
    [0.3],  # Hidden neuron 2 → Output  
    [0.5]   # Hidden neuron 3 → Output
])
print(f"Output weights shape: {weights_hidden_output.shape}")  # (3, 1)

# Hidden output (3,3) × output weights (3,1) = final output (3,1)
output_activations = np.dot(hidden_output, weights_hidden_output)
bias_output = np.array([0.1])
final_output = sigmoid(output_activations + bias_output)

print(f"Final neural network output:\n{final_output}")
print(f"Output shape: {final_output.shape}")  # (3, 1) - 3 predictions

print("\n7. PRACTICAL EXERCISE - MANUAL CALCULATION")
print("-" * 40)

# Let's manually calculate one neuron to understand the process
print("Manual calculation for first sample, first hidden neuron:")
sample_1 = inputs[0]  # [0.1, 0.9]
weights_neuron_1 = weights_input_hidden[:, 0]  # [0.5, 0.2] - connections to neuron 1

manual_calc = (sample_1[0] * weights_neuron_1[0] + 
               sample_1[1] * weights_neuron_1[1] + 
               biases_hidden[0])

print(f"  Inputs: {sample_1}")
print(f"  Weights: {weights_neuron_1}")
print(f"  Bias: {biases_hidden[0]}")
print(f"  Manual: ({sample_1[0]}×{weights_neuron_1[0]}) + ({sample_1[1]}×{weights_neuron_1[1]}) + {biases_hidden[0]}")
print(f"  Result: {manual_calc:.3f}")
print(f"  NumPy dot: {hidden_with_bias[0, 0]:.3f}")
print(f"  Match: {np.isclose(manual_calc, hidden_with_bias[0, 0])}")

print("\n🎉 DAY 2 MORNING COMPLETE!")
print("   You now understand the matrix mathematics behind neural networks!")
print("   This is exactly how biological brains process information!")