import numpy as np
import matplotlib.pyplot as plt
import os

def get_result_dir(dataset_name):
    """Get result directory based on dataset name"""
    if dataset_name.upper() == "XOR":
        result_dir = os.path.join('src', 'results', 'xor')
    elif dataset_name.upper() == "LINEAR":
        result_dir = os.path.join('src', 'results', 'linear')
    else:
        result_dir = os.path.join('src', 'results')
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir

def train_and_evaluate(nn, X, y, epochs, learning_rate, experiment_name="default", dataset_name="XOR"):
    """Train, evaluate and plot results"""
    print(f"\n--- Training ({experiment_name}) ---")
    print(f"LR: {learning_rate}, Epochs: {epochs}")
    
    history = nn.train(X, y, epochs, learning_rate)
    plot_loss_history(history, experiment_name, dataset_name)
    evaluate_model(nn, X, y)
    show_result(nn, X, y, experiment_name, dataset_name)

def plot_loss_history(history, experiment_name, dataset_name):
    """Plot loss history"""
    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.title(f'Model Loss History ({experiment_name})')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    result_dir = get_result_dir(dataset_name)
    filepath = os.path.join(result_dir, f'{experiment_name}_loss_history.png')
    plt.savefig(filepath)
    plt.close()
    print(f"\nLoss history saved to {filepath}")

def evaluate_model(nn, X, y):
    """Evaluate model and show detailed results"""
    y_pred = nn.predict(X)
    loss = nn.compute_loss(y_pred, y)
    
    y_pred_class = (y_pred > 0.5).astype(int)
    accuracy = np.mean(y_pred_class == y)
    
    print(f"\n=== Testing Result ===")
    for i in range(len(X)):
        print(f"Ground truth: {y[i][0]}, prediction: {y_pred[i][0]}")
        
    print(f"\nloss: {loss:.6f}")
    print(f"accuracy: {accuracy * 100:.2f}%")

def show_result(nn, X, y, experiment_name, dataset_name):
    y_pred = nn.predict(X)
    y_pred_class = (y_pred > 0.5).astype(int)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title('Ground Truth', fontsize=18)
    plt.plot(X[y.ravel() == 0, 0], X[y.ravel() == 0, 1], 'ro', label='Class 0')
    plt.plot(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1], 'bo', label='Class 1')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.title('Predict Result', fontsize=18)
    plt.plot(X[y_pred_class.ravel() == 0, 0], X[y_pred_class.ravel() == 0, 1], 'ro', label='Class 0')
    plt.plot(X[y_pred_class.ravel() == 1, 0], X[y_pred_class.ravel() == 1, 1], 'bo', label='Class 1')
    plt.legend()
    plt.grid(True)

    filename = f'{experiment_name}_prediction_result.png'
    result_dir = get_result_dir(dataset_name)
    filepath = os.path.join(result_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Prediction comparison saved to {filepath}")

def generate_linear(n=100):
    import numpy as np
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    import numpy as np
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)

