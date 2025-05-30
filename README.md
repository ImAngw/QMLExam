# MNIST Classification with Quantum Phase Estimation
is a hybrid quantum-classical machine learning project that leverages Quantum Phase Estimation (QPE) to encode digit-specific phases into a quantum circuit. By integrating QPE into a neural architecture, the model learns to classify handwritten digits from the MNIST dataset based on phase-encoded quantum features

## Dependencies
- PyTorch
- NumPy
- qiskit
- matplotlib (for plotting)

## utility_functions.py
This module contains helper functions used for visualization and label processing; it contains:
- Function: plot_loss
- Function: return_quantum_labels
- Function: draw_circle_with_rays

## hybrid_neural_network.py
This module contains the definition of the quantum-classical neural network.
- Class: NewQuantumLayer
- Class: CustomAngleFunction
- Class: NewHybridNN

## QML_Relation
This PDF contains all the details about the model and the results achieved.


