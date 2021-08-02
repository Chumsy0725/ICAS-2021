import numpy as np
import matplotlib.pyplot as plt
import models
import utils
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def main():
    decoder = models.Decoder_Gene()
    decoder.load_weights("AE_weights/decoder.h5")

    encoder = models.Encoder_Gene()
    encoder.load_weights("AE_weights/encoder.h5")

    ultra = layers.Input((128, 128, 1), name='Original_Image')
    x = encoder(ultra)
    out = decoder(x)
    model = Model(inputs=ultra, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mse')
    model.summary()

    actual = utils.load_image_to_array("data/test/t6102.jpeg")
    predicted_image = utils.Predict_next_image(np.array([actual]), model)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(actual[:, :, 0], cmap="gray")
    ax2.imshow(predicted_image[:, :, 0], cmap="gray")
    danger = (predicted_image - actual)
    ax3.imshow(danger[:, :, 0], cmap="gray")
    plt.show()


if __name__ == '__main__':
    main()
