from read_cifar import *
import matplotlib.pyplot as plt


def sigmoid(mat):
    """ 
    Returns the sigmoid of matrix mat
    """
    return 1 / (1 + np.exp(-mat))

def learn_once_mse(w1, b1, w2, b2, data, targets, learning_rate):
    """ 
    performs one learning step and one gradient descent
    returns the updated weights and biases and the loss for monitoring purpose
    """
    batch_size, d_out = np.shape(targets) # data has a shape of (N, d_out)
    # Forward pass
    a0 = data # the data are the input of the input layer
    z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
    a1 = sigmoid(z1) # output of the hidden layer (sigmoid activation function)
    z2 = np.matmul(a1, w2) + b2  # input of the output layer
    a2 = sigmoid(z2)  # output of the output layer (sigmoid activation function)
    predictions = a2  # the predicted values are the outputs of the output layer

    # Compute loss (MSE)
    loss = np.mean(np.square(predictions - targets))    

    # Error backpropagation
    dc_da2 = 2/(batch_size*d_out) * (a2 - targets)          # dim : N x d_out
    dc_dz2 = dc_da2 * a2 * (1 - a2)                # dim : N x d_out
    a1t = np.transpose(a1)                         # dim : d_h x N
    dc_dw2 = np.matmul(a1t, dc_dz2)                # dim : d_h x d_out
    dc_db2 = np.sum(dc_dz2, axis=0)                # dim : 1 x d_h ; line vector containing the sum of all the values over a line
    w2t = np.transpose(w2)                         # dim : d_out x d_h
    dc_da1 = np.matmul(dc_dz2, w2t)                # dim : N x d_h
    dc_dz1 = dc_da1 * a1 * (1 - a1)                # dim : N x d_h
    a0t = np.transpose(a0)                         # dim : d_in x N
    dc_dw1 = np.matmul(a0t, dc_dz1)                # dim : d_in x d_h 
    dc_db1 = np.sum(dc_dz1, axis=0)                # dim :  1 x d_in

    # Parameter update
    w1 = w1 - learning_rate * dc_dw1
    b1 = b1 - learning_rate * dc_db1
    w2 = w2 - learning_rate * dc_dw2
    b2 = b2 - learning_rate * dc_db2    

    return (w1, b1, w2, b2, loss)

def one_hot(m):
    res = [[1 if m[i]==j else 0 for j in range(10)] for i in range(len(m))]
    res = np.array(res)
    return res

def softmax(z):
    exp_z = np.exp(z)
    sum_lines = np.sum(exp_z, axis=1, keepdims=True) # sum of values line by line
    return exp_z / sum_lines

def learn_once_cross_entropy(w1, b1, w2, b2, data, labels_train, learning_rate) :
    """
    performs one learning step and one gradient descent step
    returns the updated weights and biases and the loss for monitoring purpose
    """
    # one_hot vector encoding the label
    y = one_hot(labels_train) 

    # Forward pass
    a0 = data # the data are the input of the first layer
    z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
    a1 = sigmoid(z1)  # output of the hidden layer (sigmoid activation function)
    z2 = np.matmul(a1, w2) + b2  # input of the output layer
    a2 = softmax(z2) # output of the output layer (softmax activation function)
    
    # Compute loss : cross-entropy loss 
    loss = -np.mean(y*np.log(a2))  

    # Error backpropagation
    dc_dz2 = a2 - y                          # dim : N x d_out
    # all the other gradients do not change
    a1t = np.transpose(a1)                         # dim : d_h x N
    dc_dw2 = np.matmul(a1t, dc_dz2)                # dim : d_h x d_out
    dc_db2 = np.sum(dc_dz2, axis=0)                # dim : 1 x d_h ; line vector containing the sum of all the values over a line
    w2t = np.transpose(w2)                         # dim : d_out x d_h
    dc_da1 = np.matmul(dc_dz2, w2t)                # dim : N x d_h
    dc_dz1 = dc_da1 * a1 * (1 - a1)                # dim : N x d_h
    a0t = np.transpose(a0)                         # dim : d_in x N
    dc_dw1 = np.matmul(a0t, dc_dz1)                # dim : d_in x d_h 
    dc_db1 = np.sum(dc_dz1, axis=0)                # dim :  1 x d_in

    # Parameter update
    w1 = w1 - learning_rate * dc_dw1
    b1 = b1 - learning_rate * dc_db1
    w2 = w2 - learning_rate * dc_dw2
    b2 = b2 - learning_rate * dc_db2

    # accuracy calculation
    predictions_vect = np.argmax(a2, axis=1) # the predicted label (i.e. each line of a2) of each input in the batch is the index of the maximum of a2 on each line
    true_predictions = np.sum(labels_train == predictions_vect) # true prediction means the prediction is equal to the target
    total_predictions = np.shape(labels_test)[0] # number of input data in the batch i.e. number of lines of the labels_train matrix i.e. first dimension of the matrix
    accuracy = true_predictions/total_predictions
    
    # return the accuracy for the training function
    return w1, b1, w2, b2, loss, accuracy

def train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch):
    """
    performs num_epoch of training steps
    returns final weights and biases
    and returns train_accuracies : list of train accuracies accross epochs, list of floats
    """
    train_accuracies = []
    for epoch in range(num_epoch):
        print(f"epoch : {epoch}")
        # one training step
        w1, b1, w2, b2, loss, accuracy = learn_once_cross_entropy(w1, b1, w2, b2, data_train, labels_train, learning_rate)
        print(f"loss : {loss}\naccuracy : {accuracy}\n")
        train_accuracies.append(accuracy)
         
    return w1, b1, w2, b2, train_accuracies

def test_mlp(w1, b1, w2, b2, data_test, labels_test):
    """
    Only forward pass
    Returns the test accuracy
    """
    # code similar to the learn_once_cross_entropy function, but no gradient descent
    # one_hot vector encoding the label
    y = one_hot(labels_test) 

    # Forward pass
    a0 = data_test # the data are the input of the first layer
    z1 = np.matmul(a0, w1) + b1  # input of the hidden layer
    a1 = sigmoid(z1)  # output of the hidden layer (sigmoid activation function)
    z2 = np.matmul(a1, w2) + b2  # input of the output layer
    a2 = softmax(z2) # output of the output layer (softmax activation function)    
    
    # accuracy 
    predictions = np.argmax(a2, axis=1) # returns index of the max of each line, returns a line vector
    true_predictions = np.sum(labels_test == predictions)
    total_predictions = labels_test.shape[0]
    test_accuracy = true_predictions/total_predictions
    
    return test_accuracy

def run_mlp_training(data_train, labels_train, data_test, labels_test, d_h, learning_rate, num_epoch):

    N, d_in = np.shape(data_train)  # input dimension
    d_out = 10  # 10 classes in the CIFAR 10 dataset
    print(f"N : {N}, d_in : {d_in}, d_out : {d_out}")

    # Random initialization of the network weights and biaises
    w1 = 2 * np.random.rand(d_in, d_h) - 1  # first layer weights
    b1 = np.zeros((1, d_h))  # first layer biaises
    w2 = 2 * np.random.rand(d_h, d_out) - 1  # second layer weights
    b2 = np.zeros((1, d_out))  # second layer biaises

    # Training
    w1, b1, w2, b2, train_accuracies = train_mlp(w1, b1, w2, b2, data_train, labels_train, learning_rate, num_epoch)

    # Testing
    test_accuracy = test_mlp(w1, b1, w2, b2, data_test, labels_test)

    return train_accuracies, test_accuracy


def plot_mlp(train_accuracies):

    plt.title("Evolution of learning accuracy across epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Learning accuracy")
    plt.grid()
    plt.plot(range(len(train_accuracies)), train_accuracies)
    plt.show()


if __name__ == "__main__":
    path = "D:/ECL/3A/MOD/IA/TD1/mod_4_6-td1-main/data/cifar-10-batches-py/"
    # Test read_cifar
    data, labels = read_cifar(path) # read the whole dataset

    data_train, labels_train, data_test, labels_test = split_dataset(data, labels, 0.9)
    train_accuracies, test_accuracy = run_mlp_training(data_train, labels_train, data_test, labels_test, d_h=64, learning_rate=0.1, num_epoch=100)
    
    plot_mlp(train_accuracies)
    
