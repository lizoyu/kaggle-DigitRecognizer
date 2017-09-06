from lib.data_utils import get_MNIST_data
from keras.models import load_model
from keras.backend import tf as ktf
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Read the MNIST data. Notice that we assume that it's 'kaggle-DigitRecognizer/data/train.csv', and we use helper function to read into a dictionary.
# by default, there would be 41000 training data, 1000 test data and 1000 validation data(within traning set)
data = get_MNIST_data(fit=True)

# load the model(checkpoint)
tunemodel = load_model('../models/tuneResNet_early_04-0.0146.h5', custom_objects={'ktf': ktf})

# set the loss and optimizer
rmsprop = RMSprop(lr=0.0001)
tunemodel.compile(optimizer=rmsprop, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit the model
checkpoint = ModelCheckpoint('../models/tuneResNet_early_{epoch:02d}-{loss:.4f}.h5',
                             monitor='loss',
                             save_best_only=False)
earlystop = EarlyStopping(min_delta=0.001, patience=1)
tunemodel.fit(data['X_train'], data['y_train'].reshape(-1, 1),
              batch_size=16, epochs=10, validation_data=(data['X_test'], data['y_test'].reshape(-1, 1)),
              callbacks=[checkpoint, earlystop], initial_epoch=5)