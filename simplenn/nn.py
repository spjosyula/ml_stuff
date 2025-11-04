import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('./mnist_data_set/mnist.csv')
data = np.array(data)
m, n = data.shape   # m = no. of examples, n = features + labels (784 pixels + 1 label)
np.random.shuffle(data)  #shuffle data to prevent bias

data_dev = data[0:1000].T   #takes first 1000 examples for validation (transpose -> features along rows, samples along col)
Y_dev = data_dev[0]   #first row that has all labels (0-9) 
X_dev = data_dev[1:n] / 255   #remaining rows have values ranging 0-255 (pixels) and /255 will normalize them btw [0,1]

data_train = data[1000:m].T  #remaining data is taken as training set
Y_train = data_train[0]   #same steps as before
X_train = data_train[1:n] / 255   

def init_params():
    W1 = np.random.randn(16, 784) * 0.01  #It creates a (16×784) weight matrix where each hidden neuron gets a random small weight for every input pixel ensuring neurons start differently and the network can learn unique features.
    b1 = np.zeros((16, 1))  #Initializes biases for the 16 hidden neurons to zero to shift activations during learning.
    W2 = np.random.randn(10, 16) * 0.01   #Creates small random weights connecting 16 hidden neurons to 10 output neurons (digits 0–9). 
    b2 = np.zeros((10, 1))  #Initializes biases for the 10 output neurons to zero to help adjust final predictions.
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)   #negative number becomes 0, positive stays the same (this adds non-linearity)

def softmax(Z):   #turns raw numbers (scores) into probabilities that add up to 1.
    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return e_Z / np.sum(e_Z, axis=0, keepdims=True)    #axis=0 means "do this down each column". keepdims=True means "keep it as a 2D array, don’t squeeze to 1D" so shapes match when subtracting.
def forward_prop(W1, b1, W2, b2, X):    #main function that sends data through the whole network.
    Z1 = W1.dot(X) + b1    #Weights -> importance to give each pixel, X -> i/p data, b -> bias, shifts activation up/down regardless of i/p
    A1 = ReLU(Z1)    #activation o/p of hidden layer Z1 
    Z2 = W2.dot(A1) + b2    #raw scores for each digit (0-9) (Z2 is o/p layer)
    A2 = softmax(Z2)    #produces the softmax (probabilities) of all raw scores (z2)
    return Z1, A1, Z2, A2

def one_hot(Y):    #turns numbers (like class 0, 1, 2) into a format where only the correct class is marked as 1, and all others are 0 like turning “this belongs to class 2” into [0, 0, 1]. Y is the array of classes
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))   #total examples x total no. of unique classes 
    one_hot_Y[np.arange(Y.size), Y] = 1    #For each example i, go to row i and set column Y[i] to 1.
    one_hot_Y = one_hot_Y.T    #now each col = 1 example
    return one_hot_Y

def deriv_ReLU(Z):  #derivative of ReLU
    return Z > 0
def backward_prop(Z1, A1, Z2, A2, W2, X, Y):    #forward pass create probabilities, backward prop calculates gradients (losses) to update weights
    m = Y.size
    one_hot_Y = one_hot(Y)    #convert labels into encoded vectors
    dZ2 = A2 - one_hot_Y    #error vector = predicted vector - true label vector (shows how much the o/p neuron should change its activation to reduce loss)
    dW2 = 1 / m * dZ2.dot(A1.T)    #How much each weight in W2 should change
    db2 = 1 / m * np.sum(dZ2, 1, keepdims=True)    #gradient for output bias -> sum all o/p errors per neuron
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)    #backpropagate to hidden layer. If hidden neuron isnt active (ReLU = 0), its not updated
    dW1 = 1 / m * dZ1.dot(X.T)     #Gradient for hidden layer weights -> Tells us how to update i/p -> hidden layer weights
    db1 = 1 / m * np.sum(dZ1, 1, keepdims=True)    #Same as before (db2)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):   #updates wights and biases to become more intelligent, along with a hyperparamater alpha (learning rate)
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):    #converts softmax into predicted labels (opposite of one_hot)
    return np.argmax(A2, 0)    #argmax basically takes index of the max values for each example. (0 indicates each column)

def get_accuracy(predictions, Y):   #straightforward accuracy calc -> checks how many predictions are true
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):     #core training loop
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):    #basically epochs
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):    #make predictions on new data
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2): #index is basically the ith training example
    current_img = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_img = current_img.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_img, interpolation='nearest')
    plt.show()

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 100, 0.1)

test_prediction(4, W1, b1, W2, b2)
test_prediction(6, W1, b1, W2, b2)

