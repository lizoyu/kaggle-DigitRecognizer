# coding: utf-8
# # CNN transfer learning - Keras+TensorFlow
# This is for CNN models transferred from pretrained model, using Keras based on TensorFlow. First, some preparation work.
from keras.layers import Input, Dense, Lambda
from keras.optimizers import RMSprop
from keras.backend import tf as ktf
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from lib.data_utils import get_MNIST_data

# Read the MNIST data. Notice that we assume that it's 'kaggle-DigitRecognizer/data/train.csv', and we use helper function to read into a dictionary.
# by default, there would be 41000 training data, 1000 test data and 1000 validation data(within traning set)
data = get_MNIST_data(fit=True)

# ## fine-tune transfer
# build the model
# preprocess to (28,28,3), then build a resize layer using tf.resize_images() to (224,224,3) as input
inputs = Input(shape=(28,28,3))
inputs_resize = Lambda(lambda img: ktf.image.resize_images(img, (224,224)))(inputs) # resize layer
resnet50 = ResNet50(include_top=False, input_tensor=inputs_resize, input_shape=(224,224,3), pooling='avg')
x = resnet50.output
#x = Dense(units=1024, activation='relu')(x)
predictions = Dense(units=10, activation='softmax')(x)

# connect the model
tunemodel = Model(inputs=inputs, outputs=predictions)
#tunemodel.summary()

# set the loss and optimizer
rmsprop = RMSprop(lr=0.0001)
tunemodel.compile(optimizer=rmsprop, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit the model
checkpoint = ModelCheckpoint('../models/tuneResNet_{epoch:02d}-{loss:.2f}.h5',
                             monitor='loss',
                             save_best_only=True)
tunemodel.fit(data['X_train'], data['y_train'].reshape(-1, 1),
                batch_size=16, epochs=10, callbacks=[checkpoint], initial_epoch=0)

# test the model and see accuracy
score = tunemodel.evaluate(data['X_test'], data['y_test'].reshape(-1, 1), batch_size=32)
print(score)