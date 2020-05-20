#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras


# In[2]:


import numpy as np

def get_dataset(training=True):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    if training:
        return (np.expand_dims(train_images, axis = 3), train_labels)
    return (np.expand_dims(test_images, axis = 3), test_labels)


# In[3]:


def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, 3, input_shape=(28, 28, 1), activation= 'relu',data_format="channels_last"))
    model.add(keras.layers.Conv2D(32, 3, activation= 'relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation = 'softmax'))
    model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model   


# In[4]:


def train_model(model, train_img, train_lab, test_img, test_lab, T):
    train_lab = keras.utils.to_categorical(train_lab)
    test_lab = keras.utils.to_categorical(test_lab)
    model.fit(train_img, train_lab, validation_data = (test_img, test_lab), epochs=T)


# In[5]:


def convert(n):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    return class_names[n]


# In[6]:


def predict_label(model, images, index):
    prediction = model.predict(images)
    idx = list(range(10))
    item = list(zip(prediction[index], idx))
    item.sort(reverse = True)
    print(convert(item[0][1])+ ': {:.2f}%'.format(item[0][0]*100))
    print(convert(item[1][1])+ ': {:.2f}%'.format(item[1][0]*100))
    print(convert(item[2][1])+ ': {:.2f}%'.format(item[2][0]*100))

