import numpy as np
from neural_network import NeuralNetwork, SGDOptimizer
from utils import generate_linear, generate_XOR_easy, train_and_evaluate

def run_experiment(experiment_name, dataset_name, layer_sizes, activations, epochs, learning_rate, use_sgd=False):
    print(f"\n{'='*50}")
    print(f"Experiment: {dataset_name} ({experiment_name})")
    print(f"Architecture: {layer_sizes}, Activations: {activations}")
    
    if dataset_name == "Linear":
        X, y = generate_linear(n=100)
    elif dataset_name == "XOR":
        X, y = generate_XOR_easy()
    else:
        raise ValueError("Unknown dataset name")
        
    if use_sgd:
        optimizer = SGDOptimizer(learning_rate=learning_rate)
        nn = NeuralNetwork(layer_sizes, activations, optimizer=optimizer)
        train_and_evaluate(nn, X, y, epochs, None, experiment_name, dataset_name)
    else:
        nn = NeuralNetwork(layer_sizes, activations)
        train_and_evaluate(nn, X, y, epochs, learning_rate, experiment_name, dataset_name)
