import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D, Conv2D as Convolution2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, Adadelta
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def change_dimensions(df):
    ''' Compile the 3 dimensions of the images standardize the values'''

    X = df[b'data']
    new_X = []
    for idx in range(X.shape[0]):
        img=df[b'data'][idx].reshape(32,32,3)
        img_new= np.stack([img.reshape(3,32,32)[0],
                      img.reshape(3,32,32)[1],
                      img.reshape(3,32,32)[2]],
                      axis=2)
        new_X.append(img_new)
    X_array = np.array(new_X)
    X_stand = np.array(X_array, dtype="float") / 255.0
    return X_stand

def get_labels(meta_data):
    meta_data = unpickle(meta_data)
    label_called = meta_data[b'label_names']
    label_names = []
    for label in label_called:
        new = label.decode("utf-8")
        label_names.append(new)
    return label_names, len(label_names)


def build_model():
    model = Sequential()

    model.add(Convolution2D(48, 3, 3, border_mode='same', input_shape=(32,32,3)))
    model.add(Activation('relu'))

    model.add(Convolution2D(48, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(96, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(96, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(192, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(192, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))

    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256))

    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_model_history(model_history, name):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.savefig(name)
    plt.close()


if __name__ == '__main__':

    df = unpickle('cifar-10-batches-py/data_batch_1')
    label_names, num_classes = get_labels('cifar-10-batches-py/batches.meta')
    X_stand = change_dimensions(df)
    y = np.array(df[b'labels'])
    labels = np_utils.to_categorical(y, num_classes)
    X_train, X_test, y_train, y_test = train_test_split(X_stand, labels, test_size=.2)

    model = build_model()
    model_info = model.fit(X_train, y_train,
                           batch_size=250, epochs=35,
                           validation_data = (X_test, y_test),
                           verbose=1)
    score = model.evaluate(X_test, y_test, verbose=0)
    plot_model_history(model_info, 'Model2.png')
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    preds = model.predict(X_test)
    for idx in range(20):
        real = label_names[np.argmax(y_test[idx,:])]
        predicted = label_names[np.argmax(preds[idx,:])]
        print('Predicted category: {}'.format(predicted))
        print('Real category: {}'.format(real))
        plt.imshow(X_test[idx,:,:,:])
        plt.savefig('images/image{}.png'.format(idx,predicted,real))
        plt.close()
