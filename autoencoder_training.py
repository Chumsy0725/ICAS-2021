import tf as tensorflow
import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import Model
from tensorflow.keras.losses import huber
def get_huber_loss_fn(**huber_loss_kwargs):

    def custom_huber_loss(y_true, y_pred):
                h = huber(y_true, y_pred,**huber_loss_kwargs)
                return(h)

    return custom_huber_loss
import models_lib
encoder = models.Encoder_Gene()
rms = tf.keras.optimizers.Adam(lr=0.001)
encoder.compile(optimizer=rms, loss=get_huber_loss_fn(delta=1.0))

decoder = models.Decoder_Gene()
rms = tf.keras.optimizers.Adam(lr=0.001)
decoder.compile(optimizer=rms, loss=tf.get_huber_loss_fn(delta=1.0))

ultra = tf.keras.layers.Input((128, 128, 1), name='Original_Image')
x = encoder(ultra)

out = decoder(x)
model = Model(inputs=ultra, outputs=out, name='ConvAE')
model.summary()

rms = tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=rms, loss=get_huber_loss_fn(delta=1.0))

EPOCHS = 100
BATCH_SIZE = 1
VALID_STEPS = 680
STEPS_PER_EPOCH = 575  ### train_len / batch size

def my_func(image):

    image = (image/127.5) -1
    return image

train_datagen = ImageDataGenerator(
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest',preprocessing_function=my_func)


##Modify the directories for the root files.
train_generator = train_datagen.flow_from_directory(
        "D:/ICAS2021/test", 
        target_size=(128, 128), 
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode="input")  

validation_datagen = ImageDataGenerator(preprocessing_function=my_func)

validation_generator = validation_datagen.flow_from_directory(
        "D:/ICAS2021/validation", 
        target_size=(128, 128), 
        batch_size=32,
        color_mode='grayscale',
        class_mode="input")  
def lr_schedule(epoch):
    lr = 1e-4
    return lr
lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)

model.fit_generator(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=VALID_STEPS,
        callbacks = [lr_scheduler])
