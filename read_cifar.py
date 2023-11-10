import pickle
import os 
import numpy as np
import random

def read_cifar_batch(batch):
    """
    batch : is a string, path of a single batch
    returns : matrix data, vector labels
    """
    with open(batch, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    # print(dict.keys())
    labels = dict[b'labels']
    data = dict[b'data']
    return(data, labels)

def read_cifar(path):
    """
    parameter : path of directory containing 5 data batches + test batch 
    returns : data, labels
    """
    batches = ["data_batch_1/", "data_batch_2/", "data_batch_3/", "data_batch_4/", "data_batch_5/", "test_batch/"]
    list_data = []
    list_labels = []
    for name in batches:
        file_path = os.path.join(path, name)
        data_i, labels_i = read_cifar_batch(file_path)
        list_data.append(data_i)
        list_labels.append(labels_i)
    data = np.concatenate(list_data)
    labels = np.concatenate(list_labels)
    return data, labels

def split_dataset(data, labels, split):
    nb_im = len(data)
    shuffled = [i for i in range(0, nb_im)]
    np.random.shuffle(shuffled) # liste d'entiers mélangés sans répétition entre 0 et 59 999 : indices des images

    split_index = round(split*nb_im)
    # print(split_index)
    train_index = shuffled[:split_index]
    test_index = shuffled[split_index:]
    data_train = []
    labels_train = []
    for i in train_index:
        data_train.append(data[i])
        labels_train.append(labels[i])
    
    data_test = []
    labels_test = []
    for i in test_index:
        data_test.append(data[i])
        labels_test.append(labels[i])

    data_train = np.array(data_train, dtype=np.float32)
    data_test = np.array(data_test, dtype=np.int32)
    labels_train = np.array(labels_train, dtype=np.float32)
    labels_test = np.array(labels_test, dtype=np.int32)
    
    return(data_train, labels_train, data_test, labels_test)


if __name__ == "__main__":
    
    path = "D:/ECL/3A/MOD/IA/TD1/mod_4_6-td1-main/data/cifar-10-batches-py/"

    # Test read_cifar_batch
    batch = "D:/ECL/3A/MOD/IA/TD1/mod_4_6-td1-main/data/cifar-10-batches-py/data_batch_1/"
    data, labels = read_cifar_batch(batch)
    print(f"data shape : {np.shape(data)} - labels shape : {np.shape(labels)}")

    # Test read_cifar
    data, labels = read_cifar(path)
    print(f"data shape : {np.shape(data)} - labels shape : {np.shape(labels)}")
    
    # Test split_dataset
    data_train, labels_train, data_test, labels_test = split_dataset(data, labels, 0.75)
    
    print(f"size labels_train : {len(data_train)} - {len(labels_train)}")
    print(f"size labels_test : {len(data_test)} - {len(labels_test)}")
    print(data_train.shape, data_test.shape)