import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras

from keras import optimizers
from keras import backend

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout,Flatten,\
                         Conv2D, MaxPooling2D, ZeroPadding2D,\
                         Input, GlobalAveragePooling2D,\
                         TimeDistributed,LSTM,\
                         AveragePooling2D, Concatenate, BatchNormalization



# %%
def add_denseLayer(inp, growth_rate):#bottleneck layer
    outp = Conv2D(filters=4*growth_rate, kernel_size=1, \
                  strides=1, use_bias=False)(inp)
    outp = BatchNormalization(epsilon=1.001e-5)(outp)
    outp = Activation('relu')(outp)
    #
    outp = Conv2D(filters=growth_rate, kernel_size=3, \
                  strides=1, padding='same', use_bias=False)(outp)
    outp = BatchNormalization(epsilon=1.001e-5)(outp)
    outp = Activation('relu')(outp)
    #
    outp = Concatenate()([inp,outp])
    
    return outp;


def add_denseBlock(inp, numOfLayer, growth_rate):
    
    if numOfLayer < 1:
        return inp;
    
    outp = add_denseLayer(inp, growth_rate)
    for i in range(numOfLayer-1):
        outp = add_denseLayer(outp, growth_rate)
    
    return outp;


def add_transitionBlock(inp, theta):
    #theta*m,size. determines reduction
    prev = backend.int_shape(inp)[-1]
    outp = Conv2D(filters=int(prev*theta), kernel_size=1, padding='same', \
                  strides=1, use_bias=False)(inp)
    outp = BatchNormalization(epsilon=1.001e-5)(outp)
    outp = Activation('relu')(outp)
    
    outp = AveragePooling2D(pool_size=2, strides=2)(outp)
    
    return outp;


def build_denseNet(input_shape, numOfCatg, opt=Adam(), growth_rate=32, theta=0.5):
    inp = Input(shape=input_shape)
    #default channels_last
    pre = ZeroPadding2D(padding=3)(inp)
    ####
    pre = Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False)(pre)
    pre = BatchNormalization(epsilon=1.001e-5)(pre)
    pre = Activation('relu')(pre)
    ####
    pre = ZeroPadding2D(padding=1)(pre)
    pre = MaxPooling2D(pool_size=3, strides=2)(pre)
    #Let's do this
    ##########################
    #1
    numOfLayer = [1,3,4,2]
    first = add_denseBlock(pre, numOfLayer[0], growth_rate)
    first = add_transitionBlock(first, theta)
    #2
    second = add_denseBlock(first, numOfLayer[1], growth_rate)
    second = add_transitionBlock(second, theta)
    #3
    third = add_denseBlock(second, numOfLayer[2], growth_rate)
    third = add_transitionBlock(third, theta)
    #4
    fourth = add_denseBlock(third, numOfLayer[3], growth_rate)
    ##########################
    last = GlobalAveragePooling2D(name='last_layer')(fourth)
    outp = Dense(numOfCatg, activation='softmax')(last)
    #done
    
    model = Model(inputs=inp, outputs=outp)
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model;

# %%




