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
    model.compile(optimizer=rms, loss=tf.keras.losses.Huber(delta=1.0))

    EPOCHS = 10
    BATCH_SIZE = 1

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
                                                        batch_size=BATCH_SIZE,
                                                        color_mode='grayscale',
                                                        class_mode="input")

    validation_datagen = ImageDataGenerator(preprocessing_function=normalize)

    validation_generator = validation_datagen.flow_from_directory("data/val",
                                                                  target_size=(128, 128),
                                                                  batch_size=32,
                                                                  color_mode='grayscale',
                                                                  class_mode="input")

    model.fit(train_generator,
              epochs=EPOCHS,
              validation_data=validation_generator)


if __name__ == '__main__':
    main()
