import models
import tensorflow as tf
from tensorflow.keras.models import Model

encoder = models.Encoder_Gene()
rms = tf.keras.optimizers.Adam(lr=0.001)
encoder.compile(optimizer=rms, loss=tf.keras.losses.Huber(delta=1.0))

decoder = models.Decoder_Gene()
rms = tf.keras.optimizers.Adam(lr=0.001)
decoder.compile(optimizer=rms, loss=tf.keras.losses.Huber(delta=1.0))

ultra = tf.keras.layers.Input((128, 128, 1), name='Original_Image')
x = encoder(ultra)
out = decoder(x)
model = Model(inputs=ultra, outputs=out, name='ConvAE')

rms = tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=rms, loss=tf.keras.losses.Huber(delta=1.0))


if __name__ == '__main__' :
    model.summary()
