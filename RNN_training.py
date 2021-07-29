
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from keras.layers import LeakyReLU
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Reshape,BatchNormalization
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Model
import keras
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input,Dropout
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import BatchNormalization,MaxPooling3D, Reshape,RNN,LSTM,SimpleRNN
from keras.layers import Concatenate,TimeDistributed
from keras.models import Model
import models
import utils

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1";

# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
def lr_schedule(epoch):
 
    lr = 1e-3
    if epoch > 50:
        lr *= 0.1
    if epoch > 80:
        lr *= 0.1
    print('Learning rate: ', lr)
    return lr


dataset_x = []
dataset_y = []

data_x,data_y = utils.Preprocessor()
dataset_x.extend(data_x)
dataset_y.extend(data_y)

dataset_x = np.array(dataset_x)
dataset_y = np.array(dataset_y)
#print(dataset_x )
print ("Dataset X shape",dataset_x.shape)
print ("Dataset Y shape",dataset_y.shape)

len_ = dataset_x.shape[0]
len_test = len_//10
len_valid = len_test//5

import random
from numpy import loadtxt,savetxt

numbers = np.array(random.sample(range(1196), 1196))
savetxt('numbers_44.csv', numbers, delimiter=',')

test = numbers[0:len_test]
train = numbers[len_test:]

test_x = dataset_x[test]
test_y = dataset_y[test]

valid_x = test_x[:len_valid]
valid_y = test_y[:len_valid]

final_test_x = test_x[len_valid:]
final_test_y = test_y[len_valid:]

train_x = dataset_x[train]
train_y = dataset_y[train]

print ("Dataset X shape",train_x.shape)
print ("Dataset Y shape",train_y.shape)

print ("Testing X shape",final_test_x.shape)
print ("Testing Y shape",final_test_y.shape)

print ("Validation X shape",valid_x.shape)
print ("Validation Y shape",valid_y.shape)

#####################################################################################################################
#---------------------------------ENCODER----------------------------------------------------------------------------
#####################################################################################################################

encoder =  models.Encoder_Gene()
rms = keras.optimizers.Adam(lr=0.001)
encoder.compile(optimizer=rms, loss='mse')


#Load the pretrained encoder weights here
encoder.load_weights("/content/AE_weights/encoder.h5")
encoder.trainable=False
rms = keras.optimizers.Adam(lr=0.001)
encoder.compile(optimizer=rms, loss='mse')
encoder.summary()

modelpre =  models.Time_dis_ENC(encoder)
modelpre.trainable=False
rms = keras.optimizers.Adam(lr=0.001)
modelpre.compile(optimizer=rms, loss='mse')

modelpre.summary()
#####################################################################################################################
#---------------------------------DECODER----------------------------------------------------------------------------
#####################################################################################################################

#load the pretrained decoder weights here
decoder =models.Decoder_Gene()
decoder.load_weights("/content/AE_weights/decoder.h5")
decoder.trainable=False
rms = keras.optimizers.Adam(lr=0.001)
decoder.compile(optimizer=rms, loss='mse')
decoder.summary()

#####################################################################################################################
#---------------------------------MIDDLE LAYER----------------------------------------------------------------------------
#####################################################################################################################


#Dense layer and output a RELU 1024-1024 dense layer. Add 
modelmid =models.Middle()
rms = keras.optimizers.Adam(lr=0.001)
modelmid.compile(optimizer=rms, loss='mse',
              metrics=['accuracy'])
modelmid.summary()


#SUPERMODEL
inp1 = layers.Input((3,128, 128, 1),name='Initial_three_images')
inp2 = layers.Input((1,1024),name='Zero_Tensor')

x = modelpre(inp1)
x = modelmid([x,inp2])
x = Reshape((4, 4, 64),name='Reshaping')(x)
output = decoder(x)


#modelpre.name = "Convolutional_Encoder"
#modelmid.name = "Seq2Seq_LSTM"
#decoder.name ="Convolutional_Decoder"

generator = Model(inputs=[inp1,inp2],outputs=output)
rms = keras.optimizers.Adam(lr=0.001)
generator.compile(optimizer=rms, loss='mse')

generator.summary()


lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)



generator.fit([train_x,np.zeros((train_x.shape[0],1,1024))],train_y,
            epochs=50,
            batch_size=1,
            shuffle=True,
            validation_data=([valid_x,np.zeros((valid_x.shape[0],1,1024))],valid_y),
            callbacks=[lr_scheduler])

modelmid.save_weights("AE_weights/tanh_seq2seq.h5")
print ('Weights Saved')
