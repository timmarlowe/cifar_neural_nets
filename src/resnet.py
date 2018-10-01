# import the necessary packages
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
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
import pandas as pd
import argparse
import random
import pickle
import cv2
import os
import matplotlib
matplotlib.use("Agg")

class ResNet:
    @staticmethod
    def residual_module(data, K, stride, chanDim, red=False,
        reg=0.0001, bnEps=2e-5, bnMom=0.9):
        # the shortcut branch of the ResNet module should be
        # initialize as the input (identity) data
        shortcut = data
        # the first block of the ResNet module are the 1x1 CONVs
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps,
            momentum=bnMom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False,
            kernel_regularizer=l2(reg))(act1)
        # the second block of the ResNet module are the 3x3 CONVs
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps,
            momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride,
            padding="same", use_bias=False,
            kernel_regularizer=l2(reg))(act2)
        # the third block of the ResNet module is another set of 1x1
        # CONVs
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps,
            momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False,
            kernel_regularizer=l2(reg))(act3)
        # if we are to reduce the spatial size, apply a CONV layer to
        # the shortcut
        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride,
                use_bias=False, kernel_regularizer=l2(reg))(act1)
        # add together the shortcut and the final CONV
        x = add([conv3, shortcut])
        # return the addition as the output of the ResNet module
        return x

    @staticmethod
    def build(width, height, depth, classes, stages, filters,
        reg=0.0001, bnEps=2e-5, bnMom=0.9, dataset="cifar"):
        # initialize the input shape to be "channels last" and the
        # channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        # set the input and apply BN
        inputs = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps,
            momentum=bnMom)(inputs)
        # check if we are utilizing the CIFAR dataset
        if dataset == "cifar":
            # apply a single CONV layer
            x = Conv2D(filters[0], (3, 3), use_bias=False,
                padding="same", kernel_regularizer=l2(reg))(x)
        # check to see if we are using the Tiny ImageNet dataset
        elif dataset == "tiny_imagenet":
            # apply CONV => BN => ACT => POOL to reduce spatial size
            x = Conv2D(filters[0], (5, 5), use_bias=False,
                padding="same", kernel_regularizer=l2(reg))(x)
            x = BatchNormalization(axis=chanDim, epsilon=bnEps,
                momentum=bnMom)(x)
            x = Activation("relu")(x)
            x = ZeroPadding2D((1, 1))(x)
            x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        # loop over the number of stages
        for i in range(0, len(stages)):
            # initialize the stride, then apply a residual module
            # used to reduce the spatial size of the input volume
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], stride,
                chanDim, red=True, bnEps=bnEps, bnMom=bnMom)
            # loop over the number of layers in the stage
            for j in range(0, stages[i]-1):
                # apply a ResNet module
                x = ResNet.residual_module(x, filters[i + 1],
                    (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)
        # apply BN => ACT => POOL
        x = BatchNormalization(axis=chanDim, epsilon=bnEps,
            momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)
        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)
        # create the model
        model = Model(inputs, x, name="resnet")
        # return the constructed network architecture
        return model

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

    filters = [48,48,96,96]
    stages = (9,9,9)

    model_res = ResNet()
    model = model_res.build(32, 32, 3, num_classes,stages,filters)
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
    model_info = model.fit(X_train, y_train,
                           batch_size=250, epochs=30,
                           validation_data = (X_test, y_test),
                           verbose=1)
    score = model.evaluate(X_test, y_test, verbose=1)
    plot_model_history(model_info, 'resnet_model.png')
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
