import os
import datetime
import models
from utils import normalize
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def main():
    # feel free to remove this when training
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
    model.compile(optimizer=rms, loss=tf.keras.losses.Huber(delta=1.0), metrics=['mse'])

    train_datagen = ImageDataGenerator(rotation_range=0,
                                       width_shift_range=0,
                                       height_shift_range=0,
                                       shear_range=0,
                                       zoom_range=0,
                                       horizontal_flip=True,
                                       vertical_flip=False,
                                       fill_mode='nearest',
                                       preprocessing_function=normalize)

    train_generator = train_datagen.flow_from_directory("data/test",
                                                        target_size=(128, 128),
                                                        batch_size=1,
                                                        color_mode='grayscale',
                                                        class_mode="input")

    validation_datagen = ImageDataGenerator(preprocessing_function=normalize)

    validation_generator = validation_datagen.flow_from_directory("data/val",
                                                                  target_size=(128, 128),
                                                                  batch_size=32,
                                                                  color_mode='grayscale',
                                                                  class_mode="input")

    log_dir = "AE_logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(train_generator,
              epochs=10,
              validation_data=validation_generator,
              callbacks=[tensorboard_callback])

    path = os.path.join("AE_weights")
    if not os.path.exists(path):
        os.mkdir(path)

    encoder.save_weights("AE_weights/encoder.h5")
    decoder.save_weights("AE_weights/decoder.h5")
    print("Saved model to disk")


if __name__ == '__main__':
    main()
