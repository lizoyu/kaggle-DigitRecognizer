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

# ## Freeze-weights transfer
# We would use ResNet50 provided in Keras. In this section, the pretrained model would all be freezed, and new output layer would be attatched to the model, and only this output layer would be trained.
# build the model
# preprocess to (28,28,3), then build a resize layer using tf.resize_images() to (224,224,3) as input
inputs = Input(shape=(28,28,3))
inputs_resize = Lambda(lambda img: ktf.image.resize_images(img, (224,224)))(inputs)
resnet50 = ResNet50(include_top=False, input_tensor=inputs_resize, input_shape=(224,224,3), pooling='avg')
x = resnet50.output
x = Dense(units=1024, activation='relu')(x)
predictions = Dense(units=10, activation='softmax')(x)

freezemodel = Model(inputs=inputs, outputs=predictions)

# freeze all ResNet50 layers
for layer in resnet50.layers:
    layer.trainable = False

# set the loss and optimizer
freezemodel.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit the model
checkpoint = ModelCheckpoint('../models/freezeResNet_{epoch:02d}-{accuracy:.2f}.h5',
                             monitor='accuracy',
                             save_best_only=True)
freezemodel.fit(data['X_train'], data['y_train'].reshape(-1,1), batch_size=16, epochs=10, callbacks=[checkpoint])

# test the model and see accuracy
score = freezemodel.evaluate(data['X_test'], data['y_test'].reshape(-1, 1), batch_size=32)
print(score)