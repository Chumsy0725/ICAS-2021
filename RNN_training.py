import models
import utils
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model




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

paths=['data/train/icab1/front_camera','data/train/icab1/back_camera','data/train/icab1/side_camera','data/train/icab2']
for path0 in paths:
    data_x,data_y = utils.Preprocessor(path0,slen=3,cap = "/",flip=False)   
    dataset_x.extend(data_x)
    dataset_y.extend(data_y)

dataset_x = np.array(dataset_x)
dataset_y = np.array(dataset_y)
#print(dataset_x )
print("Dataset X shape", dataset_x.shape)
print("Dataset Y shape", dataset_y.shape)

len_ = dataset_x.shape[0]
print('len', len_)
len_test = len_//5
print('len test', len_test)
len_valid = len_test//5
print('len valid', len_valid)

numbers = np.array(random.sample(range(len_), len_))
np.savetxt('numbers_44.csv', numbers, delimiter=',')

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

print("Dataset X shape", train_x.shape)
print("Dataset Y shape", train_y.shape)

print("Testing X shape", final_test_x.shape)
print("Testing Y shape", final_test_y.shape)

print("Validation X shape", valid_x.shape)
print("Validation Y shape", valid_y.shape)


#####################################################################################################################
# ---------------------------------ENCODER----------------------------------------------------------------------------
#####################################################################################################################

encoder = models.Encoder_Gene()
rms = tf.keras.optimizers.Adam(lr=0.001)
encoder.compile(optimizer=rms, loss='mse')


# Load the pretrained encoder weights here
encoder.load_weights("AE_weights/encoder.h5")
encoder.trainable = False
rms = tf.keras.optimizers.Adam(lr=0.001)
encoder.compile(optimizer=rms, loss='mse')
# encoder.summary()

modelpre = models.Time_dis_ENC(encoder)
modelpre.trainable = False
rms = tf.keras.optimizers.Adam(lr=0.001)
modelpre.compile(optimizer=rms, loss='mse')

# modelpre.summary()


#####################################################################################################################
# ---------------------------------DECODER----------------------------------------------------------------------------
#####################################################################################################################

# load the pretrained decoder weights here
decoder = models.Decoder_Gene()
decoder.load_weights("AE_weights/decoder.h5")
decoder.trainable = False
rms = tf.keras.optimizers.Adam(lr=0.001)
decoder.compile(optimizer=rms, loss='mse')
# decoder.summary()

#####################################################################################################################
# ---------------------------------MIDDLE LAYER----------------------------------------------------------------------------
#####################################################################################################################


# Dense layer and output a RELU 1024-1024 dense layer. Add
modelmid = models.Middle()
rms = tf.keras.optimizers.Adam(lr=0.001)
modelmid.compile(optimizer=rms, loss='mse',
                 metrics=['accuracy'])
# modelmid.summary()


# SUPERMODEL
inp1 = layers.Input((3, 128, 128, 1), name='Initial_three_images')
inp2 = layers.Input((1, 1024), name='Zero_Tensor')

x = modelpre(inp1)
x = modelmid([x, inp2])
x = layers.Reshape((4, 4, 64), name='Reshaping')(x)
output = decoder(x)


#modelpre.name = "Convolutional_Encoder"
#modelmid.name = "Seq2Seq_LSTM"
#decoder.name ="Convolutional_Decoder"

generator = Model(inputs=[inp1, inp2], outputs=output)
rms = tf.keras.optimizers.Adam(lr=0.001)
generator.compile(optimizer=rms, loss='mse')

generator.summary()


lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)


generator.fit([train_x, np.zeros((train_x.shape[0], 1, 1024))], train_y,
              epochs=100,
              batch_size=5,
              shuffle=True,
              validation_data=(
                  [valid_x, np.zeros((valid_x.shape[0], 1, 1024))], valid_y),
              callbacks=[lr_scheduler])

modelmid.save_weights("RNN_weights/tanh_seq2seq.h5")
print('Weights Saved')
