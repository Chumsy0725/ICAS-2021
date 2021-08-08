import tensorflow as tf
import models
import utils
import PIL
from tensorflow.keras import backend as K
import numpy as np
import sys
from PIL import Image
import os
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    array_to_img,
    img_to_array,
    load_img,
)

import os

sys.modules["Image"] = Image
rms = tf.keras.optimizers.Adam(lr=0.001)


# Modify these for the location of the sequences.
train_x, train_y = utils.Preprocessor(
    "normal_test", cap="/")
print('train_x:', train_x.shape)
print('train_y:', train_y.shape)
rms = tf.keras.optimizers.Adam(lr=0.001)

encoder = models.Encoder_Gene()
encoder.load_weights('GAN_weights\enc_gen.h5')
encoder.compile(optimizer=rms, loss='mse')

modelpre = models.Time_dis_ENC(encoder)
modelpre.compile(optimizer=rms, loss='mse')

decoder = models.Decoder_Gene()
decoder.load_weights('GAN_weights\dec_gen.h5')
decoder.compile(optimizer=rms, loss='mse')


modelmid = models.Middle()
modelmid.load_weights('GAN_weights\RNN_middle.h5')
modelmid.compile(optimizer=rms, loss='mse',
                 metrics=['accuracy'])


# SUPERMODEL
inp1 = tf.keras.layers.Input((3, 128, 128, 1), name='Initial_three_images')
inp2 = tf.keras.layers.Input((1, 1024), name='Zero_Tensor')

x = modelpre(inp1)
x = modelmid([x, inp2])
x = tf.keras.layers.Reshape((4, 4, 64), name='Reshaping')(x)
output = decoder(x)

generator = tf.keras.Model(inputs=[inp1, inp2], outputs=output)
rms = tf.keras.optimizers.Adam(lr=0.001)
generator.compile(optimizer=rms, loss='mse')
generator.summary()


def load_image_to_array(address):
    image = load_img(
        address, grayscale=True, target_size=(128, 128), interpolation="hamming"
    )
    image = img_to_array(image)
    return image


def plot_image(image_array):
    plt.axis("off")
    plt.imshow(image_array[:, :, 0], cmap="gray")


def Predict_next_image(seq, model):
    img = model.predict([seq, np.zeros((seq.shape[0], 1, 1024))])
    img = (1 + img[0]) / 2
    return img


for i in range(train_x.shape[0]):
    predicted_image = Predict_next_image(np.array([train_x[i]]), generator)

    # plt.imshow(actual[:,:,0],cmap="gray")
    # plt.show()

    plt.imshow(predicted_image[:, :, 0], cmap="gray")
    plt.savefig("predictions on last year data with our model/"+str(i)+".jpeg")
    plt.show()
    plt.close()    # close the figure to show the next one.
#danger = (predicted_image - actual)
# plt.imshow(danger[:,:,0])
# plt.show()
