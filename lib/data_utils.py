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
    X_train = mnist_data[:num_training,1:]
    y_train = mnist_data[:num_training,0]
    if num_validation:
        X_val = X_train[:num_validation,:]
        y_val = y_train[:num_validation]
    if num_test:
        X_test = mnist_data[num_training:num_training+num_test,1:]
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

def create_submission(model, test_path, save_path):
    """
    use Keras trained model to create submission for Kaggle competition

    Inputs:
        - model: trained Keras model
        - test_path: the path to the test data
        - save_path: the path to save the submission with file name
    """
    X_test = []
    with open(test_path, 'rt') as csvfile:
        fileToRead = csv.reader(csvfile)

        # skip the header
        headers = next(fileToRead)

        for x in fileToRead:
            X_test.append(x)

    predictions = model.predict(np.array(X_test, dtype=np.float32))
    with open(save_path, 'rt') as csvfile:
        fileToWrite = csv.writer(out, delimiter=',', lineterminator='\n')

        # write the header
        fileToWrite.writerow(['ImageID', 'Label'])
        # write the predictions
        i = 0
        for row in fileToWrite:
            fileToWrite.writerow([i+1, predictions[i]])
            i += 1

# test functions
'''
def tester():
    with open('../data/test.csv', 'rt') as csvfile:
        fileToRead = csv.reader(csvfile)

        # skip the header
        headers = next(fileToRead)

        img = np.array(next(fileToRead), dtype=np.int).reshape((28,28))
        from matplotlib.pyplot import imshow, show
        imshow(img)
        show()
tester()
'''