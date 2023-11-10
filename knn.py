from read_cifar import *
import matplotlib.pyplot as plt


def distance_matrix(a, b):
    """
    returns the L2 Euclidian distance matrix
    """
    a2 = np.sum(np.square(a), axis=1, keepdims=True) # sum over each line 
    b2 = np.sum(np.square(b), axis=1, keepdims=True)

    dists = np.sqrt(a2 + b2.T - 2 * np.dot(a, b.T))
    return dists

# Select the most frequent one
def democracy(arr):
    """
    majority vote in the labels array 
    returns the most frequent label in the array
    """
    values, count = np.unique(arr, return_counts=True)
    return values[np.argmax(count)]

def knn_predict(dists, labels_train, k):

    nearest_indices = np.argsort(dists.T)[:, :k]
    nearest_labels = [labels_train[i] for i in nearest_indices]
    predictions = np.array([democracy(arr) for arr in nearest_labels])

    return predictions

def evaluate_knn(data_train, labels_train, data_test, labels_test, k):

    dists = distance_matrix(data_train, data_test)      
    prediction = knn_predict(dists, labels_train, k)

    correct = 0
    for pred, test in zip(prediction, labels_test):
        if pred == test:
            correct += 1

    return correct / len(labels_test)


def plot_evaluate_knn(data_train, labels_train, data_test, labels_test):

    dists = distance_matrix(data_train, data_test)
    sorted_dist = np.argsort(dists.T)

    k_list = [i for i in range(1, 21)]
    accuracies = []
    for k in range(1, 21):
        print("\n Evaluating k=%d" % k)
        accuracies.append(evaluate_knn(
            data_train=data_train,
            labels_train=labels_train,
            data_test=data_test,
            labels_test=labels_test,
            k=k,
            dists=dists,
            sorted_dist=sorted_dist
        ))

    plt.title("Variation of the accuracy as a function of k")
    plt.xlabel('k')
    plt.ylabel("Accuracy")
    plt.grid()
    plt.plot(k_list, accuracies, 'o-')
    plt.show()


if __name__ == "__main__":
    data, labels = read_cifar("./data/cifar-10-batches-py")

    split = 0.9
    data_train, data_test, labels_train, labels_test = split_dataset(data, labels, split)

    plot_evaluate_knn(data_train, labels_train, data_test, labels_test)
