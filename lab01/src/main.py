from train import run_experiment
import numpy as np

def main():
    """Define and run all neural network experiments"""
    np.random.seed(42)

    experiments = [
        # A. Learning Rate Comparison
        {
            "experiment_name": "A_Baseline_lr_0.1",
            "dataset_name": "XOR",
            "layer_sizes": [2, 8, 8, 1],
            "activations": ['relu', 'relu', 'sigmoid'],
            "epochs": 50000,
            "learning_rate": 0.1
        },
        {
            "experiment_name": "A_High_lr_0.5",
            "dataset_name": "XOR",
            "layer_sizes": [2, 8, 8, 1],
            "activations": ['relu', 'relu', 'sigmoid'],
            "epochs": 50000,
            "learning_rate": 0.5
        },
        {
            "experiment_name": "A_Low_lr_0.01",
            "dataset_name": "XOR",
            "layer_sizes": [2, 8, 8, 1],
            "activations": ['relu', 'relu', 'sigmoid'],
            "epochs": 50000,
            "learning_rate": 0.01
        },

        # B. Hidden Layer Size Comparison
        {
            "experiment_name": "B_Small_Net_4x4",
            "dataset_name": "XOR",
            "layer_sizes": [2, 4, 4, 1],
            "activations": ['relu', 'relu', 'sigmoid'],
            "epochs": 50000,
            "learning_rate": 0.1
        },
        {
            "experiment_name": "B_Large_Net_16x16",
            "dataset_name": "XOR",
            "layer_sizes": [2, 16, 16, 1],
            "activations": ['relu', 'relu', 'sigmoid'],
            "epochs": 50000,
            "learning_rate": 0.1
        },

        # C. Activation Function Comparison
        {
            "experiment_name": "C_No_Activation",
            "dataset_name": "XOR",
            "layer_sizes": [2, 8, 8, 1],
            "activations": [None, None, 'sigmoid'],
            "epochs": 50000,
            "learning_rate": 0.1
        },

        # D. Additional Experiments
        {
            "experiment_name": "D_Tanh_Activation_8x8",
            "dataset_name": "XOR",
            "layer_sizes": [2, 8, 8, 1],
            "activations": ['tanh', 'tanh', 'sigmoid'],
            "epochs": 50000,
            "learning_rate": 0.1
        },

        # E. Optimizer Comparison
        {
            "experiment_name": "E_SGD_Optimizer",
            "dataset_name": "XOR",
            "layer_sizes": [2, 8, 8, 1],
            "activations": ['relu', 'relu', 'sigmoid'],
            "epochs": 50000,
            "learning_rate": 0.1,
            "use_sgd": True
        },
    ]

    # Copy all experiments for Linear dataset
    linear_experiments = []
    for exp in experiments:
        new_exp = exp.copy()
        new_exp["dataset_name"] = "Linear"
        new_exp["experiment_name"] = f"Linear_{new_exp['experiment_name']}"
        if "Linear" in new_exp["experiment_name"]:
             new_exp["epochs"] = 50000
        linear_experiments.append(new_exp)

    experiments.extend(linear_experiments)

    # Run all experiments
    for i, params in enumerate(experiments):
        print(f"--- Running experiment {i+1}/{len(experiments)}: {params['experiment_name']} ---")
        run_experiment(**params)
        print(f"--- Experiment {i+1}/{len(experiments)}: {params['experiment_name']} completed ---")

if __name__ == "__main__":
    main()
