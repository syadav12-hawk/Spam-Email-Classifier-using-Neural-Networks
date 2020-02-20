# -*- coding: utf-8 -*-

import keras
from keras import models
import csv
from keras import layers
import numpy as np
import io
import requests
from sklearn import preprocessing
import pandas as pd
from keras.models import load_model

import pandas as pd


def load_spam_data(data):
    y=data[:,-1]
    x=data[:,:57]
        
    #Train Data
    x_train=x[:3680,:]
    y_train=y[:3680]
        
    #Test data
    x_test=x[3680:,:]
    y_test=y[3680:]
        
    return x_train,y_train,x_test,y_test       



url='https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
df = pd.read_csv(url, error_bad_lines=False,header=None)
data=df.to_numpy()
np.random.shuffle(data)


x_train,y_train,x_test,y_test=load_spam_data(data)

#Normalization
norm_X_train = preprocessing.normalize(x_train)
norm_X_test = preprocessing.normalize(x_test)

#Spliting into Validation Sets
partial_x_train=norm_X_train[:2945,:]
partial_y_train=y_train[:2945]
    
x_val=norm_X_train[2945:,:]
y_val=y_train[2945:]

#Model Design
model=models.Sequential()
model.add(layers.Dense(32,activation='relu',input_shape=(57,)))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
rmsprop=keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
#Compile model

model.compile(optimizer=rmsprop,loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(partial_x_train,partial_y_train,epochs=100
                  ,batch_size=256,
                  validation_data=(x_val,y_val))
history_dict=history.history

model.save("model_spam.hdf5")
print("Model Saved")

#print("Loading Wieghts")
#model.load_weights('my_model_wt.h5')

#Plot
import matplotlib.pyplot as plt
loss_values=history_dict['loss']
val_loss_values=history_dict['val_loss']
epochs=range(1,len(loss_values)+1)

#Loss Plot
plt.plot(epochs,loss_values,'bo',label='Training Loss')
plt.plot(epochs,val_loss_values,'b',label='Validation Loss')
plt.title('Traning and Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#Accuracy Plot
plt.clf()
acc_values=history_dict['accuracy']
val_acc_values=history_dict['val_accuracy']
plt.plot(epochs,acc_values,'bo',label='Training Accuracy')
plt.plot(epochs,val_acc_values,'b',label='Validation Accuracy')
plt.title('Traning and Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

result=model.evaluate(x_val,y_val)
print(result)



model_test=load_model("model_spam.hdf5")

result=model_test.evaluate(norm_X_test,y_test)
print("Test Results :")
print(result)
