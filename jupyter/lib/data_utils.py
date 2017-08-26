import csv
import numpy as np
from random import shuffle

def load_MNIST(filename):
    """
    load all the data from MNIST

    Input:
        - filename: path to the csv file
    Output:
        - data: list of data
    """
    with open(filename, 'rt') as csvfile:
        fileToRead = csv.reader(csvfile)

        # skip the header
        headers = next(fileToRead)

        # split labels and data and save in dictionary
        data= []
        for row in fileToRead:
            data.append(row)

        return data

def get_MNIST_data(num_training=41000, num_validation=1000, num_test=1000, subtract_mean=True):
    """
    Load the MNIST dataset(42,000 in total) from disk and perform preprocessing to prepare
    it for classifiers.

    Inputs:
        - num_training: number of data used in training
        - num_validation: number of data used in validation
        - num_test: number of data used in test
        - subtract_mean: indicate whether to normalize the data

    Outputs:
        - datadict: prepared data dictionary with 'X_train', 'y_train', 'X_val', 'y_val', 
            'X_test' and 'y_test'
    """
    # load MNIST data
    mnist_path = '../data/train.csv'
    mnist_data = np.array(load_MNIST(mnist_path), dtype=np.float32)

    # shuffle and split data into training, validation and test sets
    shuffle(mnist_data)
    X_train = mnist_data[:num_training,1:].reshape((-1,28,28,1))
    y_train = mnist_data[:num_training,0]
    if num_validation:
        X_val = X_train[:num_validation,:].reshape((-1,28,28,1))
        y_val = y_train[:num_validation]
    if num_test:
        X_test = mnist_data[num_training:num_training+num_test,1:].reshape((-1,28,28,1))
        y_test = mnist_data[num_training:num_training+num_test,0]

    # normalize the data: subtract the mean from images
    if subtract_mean:
        mean_img = np.mean(X_train, axis=0)
        X_train -= mean_img
        if num_validation:
            X_val -= mean_img
        if num_test:
            X_test -= mean_img

    # merge into a dictionary
    datadict = {'X_train': X_train, 'y_train': y_train}
    if num_test:
        datadict['X_test'] = X_test
        datadict['y_test'] = y_test
    if num_validation:
        datadict['X_val'] = X_val
        datadict['y_val'] = y_val

    return datadict

def load_model(filename):
    """
    load the TensorFlow model
    """
    pass

# test functions
'''
def tester():
    datadict = get_MNIST_data()
    print(datadict['X_train'].shape)
'''
#tester()