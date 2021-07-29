import numpy as np
import matplotlib.pyplot as plt
import models
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

decoder = models.Decoder_Gene()
decoder.load_weights("AE_weights/encoder.h5")

encoder = models.Encoder_Gene()
encoder.load_weights("AE_weights/encoder.h5")

decoder.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')
encoder.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')

ultra = layers.Input((128, 128, 1), name='Original_Image')
x = encoder(ultra)
out = decoder(x)
model = Model(inputs=ultra, outputs=out)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')
model.summary()

def load_image_to_array(address):
    image = tf.keras.preprocessing.image.load_img(
        address, grayscale=True, target_size=(128, 128), interpolation="hamming"
    )
    image = tf.keras.preprocessing.image.img_to_array(image)
    return image

def plot_image(image_array):
    plt.axis("off")
    plt.imshow(image_array[:, :, 0], cmap="gray")

def Predict_next_image(seq, model):
    seq = (seq / 127.5) - 1
    img = model.predict([seq])
    img = (1 + img[0]) / 2
    return img

actual = load_image_to_array("data/test/t6102.jpeg")
predicted_image = Predict_next_image(np.array([actual]),model)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(actual[:,:,0],cmap="gray")
ax2.imshow(predicted_image[:,:,0],cmap="gray")
danger = (predicted_image - actual)
ax3.imshow(danger[:,:,0])
plt.show()