import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#check simplenn code for understanding the NN

data = pd.read_csv('../simplenn/mnist_data_set/mnist.csv')
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n] / 255

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255

def ReLU(Z):
    return np.maximum(0, Z)   

def softmax(Z):
    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))   
    return e_Z / np.sum(e_Z, axis=0, keepdims=True)  

def deriv_ReLU(Z):
    return Z > 0  

def init_params():
    W1 = np.random.randn(16, 784) * 0.01   
    b1 = np.zeros((16, 1))  
    W2 = np.random.randn(10, 16) * 0.01   
    b2 = np.zeros((10, 1))   
    return W1, b1, W2, b2

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1  
    A1 = ReLU(Z1)  
    Z2 = W2.dot(A1) + b2  
    A2 = softmax(Z2)   
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))   
    one_hot_Y[np.arange(Y.size), Y] = 1  
    one_hot_Y = one_hot_Y.T  
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size   
    one_hot_Y = one_hot(Y)  
    dZ2 = A2 - one_hot_Y   
    dW2 = 1 / m * dZ2.dot(A1.T)  
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)   
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)   
    dW1 = 1 / m * dZ1.dot(X.T)   
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)  
    return dW1, db1, dW2, db2

def get_predictions(A2):
    return np.argmax(A2, axis=0)  

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size   

def compute_loss(A2, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    log_probs = -np.log(A2 + 1e-8)   # how confident were we in the right answer?
    loss = np.sum(one_hot_Y * log_probs) / m   # average loss across all examples
    return loss

# Base Optimizer class - blueprint for all optimizers
class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate    
        self.initialized = False

    def initialize(self, params):
        #Set up optimizer's internal tracking variables
        pass

    def update(self, params, grads):
        #update weights based on gradients
        pass

# SGD with Momentum - remembers past gradients to move faster in consistent directions
class SGDMomentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)    #learning rate - how big/small each step is
        self.momentum = momentum   # how much past gradients affect the next step (0.9 = 90%)
        self.velocities = {}   # stores the "momentum" for each weight

    def initialize(self, params):
        #Create velocity tracker for each weight, starting at zero
        for key, param in params.items():
            self.velocities[key] = np.zeros_like(param)    #For each weight, create a velocity array with the same shape, all zeros.
        self.initialized = True     #Mark that initialization is finished.

    def update(self, params, grads):    #This function performs ONE update step: uses gradients, applies momentum, returns new weights

        if not self.initialized:    #If it's the first update, initialize velocities first.
            self.initialize(params)

        updated_params = {}    #dict to store updated weights
        for key in params.keys():
            # Take 90% of the previous velocity (if momentum=0.9). Add the new gradient. This builds speed in directions where gradient stays consistent
            self.velocities[key] = self.momentum * self.velocities[key] + grads[key]
            #Move the weight in the opposite direction of the velocity. Velocity already contains the smoothed gradient.
            updated_params[key] = params[key] - self.learning_rate * self.velocities[key]

        return updated_params

# RMSprop gives bigger steps to weights that rarely change, smaller steps to jumpy ones
class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta = beta   # how much to remember past squared gradients
        self.epsilon = epsilon   # tiny number to avoid dividing by zero
        self.cache = {}   # tracks how jumpy/big each weight's gradient has been

    def initialize(self, params):
       
        for key, param in params.items():   #for each weight (W1, b1, W2...)
            self.cache[key] = np.zeros_like(param)    #create cache with same size as weight. Init with 0. It will accumulate squared gradients over time.
        self.initialized = True

    def update(self, params, grads):   #This performs one training update using RMSprop logic.
        if not self.initialized:
            self.initialize(params)

        updated_params = {}    #dict to store weights
        for key in params.keys():
            #  Track average of squared gradients
            #  If gradients are consistently large, cache becomes large => the weight is noisy / unstable 
            #  If gradients are small, cache stays small => the weight is stable 
            self.cache[key] = self.beta * self.cache[key] + (1 - self.beta) * (grads[key] ** 2)
            #  Divide by sqrt of cache to adapt learning rate per weight
            #  If cache is large (weight is unstable) =>  divide by big number => take a smaller step
            #  If cache is small (weight is stable) => divide by small number => take a bigger step
            updated_params[key] = params[key] - self.learning_rate * grads[key] / (np.sqrt(self.cache[key]) + self.epsilon)

        return updated_params

# Adam - combines the best of Momentum and RMSprop
class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1   # how much to remember past gradients (momentum)
        self.beta2 = beta2   # how much to remember past squared gradients (RMSprop)
        self.epsilon = epsilon   # tiny number for stability (avoids zero div err)
        self.m = {}   # momentum tracker (like velocity in SGD)
        self.v = {}   # volatility tracker (like cache in RMSprop)
        self.t = 0   # counts how many updates we've done

    def initialize(self, params):
        #Init momentum and volatility trackers with 0
        for key, param in params.items():
            self.m[key] = np.zeros_like(param)
            self.v[key] = np.zeros_like(param)
        self.initialized = True

    def update(self, params, grads):
        #Combines momentum + adaptive LRs with bias correction
        if not self.initialized:
            self.initialize(params)

        self.t += 1   # count the update step (for bias correction)
        updated_params = {}

        for key in params.keys():
            # Update momentum (like SGD)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            # Update volatility tracker (like RMSprop)
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # Early in training, both m and v are too close to zero because they start at zero. So we correct them:
            m_corrected = self.m[key] / (1 - self.beta1 ** self.t)
            v_corrected = self.v[key] / (1 - self.beta2 ** self.t)

            # Apply both momentum and adaptive learning rate
            updated_params[key] = params[key] - self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)

        return updated_params

# Split data into small batches for training
def create_minibatches(X, Y, batch_size):
    #Instead of using all data at once, use small random batches
    m = X.shape[1]
    minibatches = []

    # Shuffle the data
    permutation = np.random.permutation(m)
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[permutation]

    # Split into batches
    num_complete_batches = m // batch_size
    for k in range(num_complete_batches):
        minibatch_X = shuffled_X[:, k * batch_size:(k + 1) * batch_size]
        minibatch_Y = shuffled_Y[k * batch_size:(k + 1) * batch_size]
        minibatches.append((minibatch_X, minibatch_Y))

    # Handle leftover data
    if m % batch_size != 0:
        minibatch_X = shuffled_X[:, num_complete_batches * batch_size:]
        minibatch_Y = shuffled_Y[num_complete_batches * batch_size:]
        minibatches.append((minibatch_X, minibatch_Y))

    return minibatches

# Main training loop
def train_with_optimizer(X_train, Y_train, optimizer, epochs=500, batch_size=128, verbose=True):
    #Train the network and track how well it's learning
    W1, b1, W2, b2 = init_params()

    history = {
        'loss': [],
        'accuracy': [],
        'grad_norms': []   # track gradient sizes to see learning dynamics
    }

    start_time = time.time()

    for epoch in range(epochs):
        minibatches = create_minibatches(X_train, Y_train, batch_size)

        for minibatch_X, minibatch_Y in minibatches:
            # Forward pass: make predictions
            Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, minibatch_X)

            # Backward pass: calculate how to improve
            dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, minibatch_X, minibatch_Y)

            # Package everything for the optimizer
            params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
            grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

            # Let optimizer update the weights
            updated_params = optimizer.update(params, grads)
            W1, b1, W2, b2 = updated_params['W1'], updated_params['b1'], updated_params['W2'], updated_params['b2']

        # Check performance on full training set
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_train)
        train_loss = compute_loss(A2, Y_train)
        predictions = get_predictions(A2)
        accuracy = get_accuracy(predictions, Y_train)

        # Track how big the gradients are
        grad_norm = np.sqrt(np.sum(dW1**2) + np.sum(dW2**2))

        history['loss'].append(train_loss)
        history['accuracy'].append(accuracy)
        history['grad_norms'].append(grad_norm)

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {train_loss:.4f}, Accuracy = {accuracy:.4f}")

    training_time = time.time() - start_time

    if verbose:
        print(f"Training done in {training_time:.2f} seconds")
        print(f"Final: Loss = {history['loss'][-1]:.4f}, Accuracy = {history['accuracy'][-1]:.4f}\n")

    return W1, b1, W2, b2, history, training_time

# Compare all three optimizers
def compare_optimizers(epochs=500):
    #Train with each optimizer and create comparison charts
    print("Optimizer Comparison: SGD Momentum vs RMSprop vs Adam")
    print(f"Training for {epochs} epochs with batch size 128\n")

    batch_size = 128

    # Set up the three optimizers
    optimizers = {
        'SGD Momentum': SGDMomentum(learning_rate=0.01, momentum=0.9),
        'RMSprop': RMSprop(learning_rate=0.001, beta=0.9),
        'Adam': Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
    }

    results = {}

    # Train with each optimizer
    for name, optimizer in optimizers.items():
        print(f"Training with {name}")
        W1, b1, W2, b2, history, train_time = train_with_optimizer(
            X_train, Y_train, optimizer, epochs=epochs, batch_size=batch_size, verbose=True
        )
        results[name] = {
            'history': history,
            'time': train_time,
            'params': (W1, b1, W2, b2)
        }

    # Comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Training Loss
    ax = axes[0, 0]
    for name, result in results.items():
        ax.plot(result['history']['loss'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Training Accuracy
    ax = axes[0, 1]
    for name, result in results.items():
        ax.plot(result['history']['accuracy'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Gradient Norms (shows how "active" the learning is)
    ax = axes[1, 0]
    for name, result in results.items():
        ax.plot(result['history']['grad_norms'], label=name, linewidth=2, alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Size')
    ax.set_title('Learning Activity (Gradient Magnitudes)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')   # log scale makes patterns easier to see so yeah

    # Plot 4: Summary Table
    ax = axes[1, 1]
    ax.axis('off')

    table_data = []
    for name, result in results.items():
        final_acc = result['history']['accuracy'][-1]
        final_loss = result['history']['loss'][-1]
        time_taken = result['time']

        # When did it hit 90% accuracy?
        acc_history = np.array(result['history']['accuracy'])
        epochs_to_90 = np.where(acc_history >= 0.90)[0]
        epochs_to_90 = epochs_to_90[0] if len(epochs_to_90) > 0 else 'N/A'

        table_data.append([name, f"{final_acc:.4f}", f"{final_loss:.4f}", str(epochs_to_90), f"{time_taken:.2f}s"])

    table = ax.table(cellText=table_data,
                     colLabels=['Optimizer', 'Final Acc', 'Final Loss', 'Epochs to 90%', 'Time'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Styling I ain't familiar with
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Performance Summary', fontsize=12, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved as 'optimizer_comparison.png'")
    plt.show()

    return results

if __name__ == "__main__":
    results = compare_optimizers(epochs=50)
