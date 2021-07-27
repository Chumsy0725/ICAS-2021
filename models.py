from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential


def Encoder_Gene():
    inp1 = layers.Input((128, 128, 1), name="Image_Input")

    x = layers.Conv2D(16, (3, 3), padding="same", name="EC01_1")(inp1)
    x = layers.LeakyReLU(alpha=0.2, name="EA01_1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="EP01")(x)

    x = layers.Conv2D(32, (3, 3), padding="same", name="EC02_1")(x)
    x = layers.LeakyReLU(alpha=0.2, name="EA02_1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="EP02")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", name="EC03_1")(x)
    x = layers.LeakyReLU(alpha=0.2, name="EA03_1")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", name="EC03_2")(x)
    x = layers.LeakyReLU(alpha=0.2, name="EA03_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="EP03")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", name="EC04_1")(x)
    x = layers.LeakyReLU(alpha=0.2, name="EA04_1")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", name="EC04_2")(x)
    x = layers.LeakyReLU(alpha=0.2, name="EA04_2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="EP04")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", name="EC05_1")(x)
    x = layers.LeakyReLU(alpha=0.2, name="EA05_1")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", name="EC05_2")(x)
    x = layers.LeakyReLU(alpha=0.2, name="EA05_2")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", name="EC05_3", activation="tanh")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="EP05")(x)

    encoder = Model(inputs=inp1, outputs=x)
    return encoder


def Decoder_Gene():
    inp = layers.Input((4, 4, 64), name="Latent_Representation")
    x = layers.UpSampling2D((2, 2), name="DU0")(inp)

    x = layers.Conv2D(64, (3, 3), padding="same", name="DC01_01")(x)
    x = layers.LeakyReLU(alpha=0.2, name="DA01_01")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", name="DC01_02")(x)
    x = layers.LeakyReLU(alpha=0.2, name="DA01_02")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", name="DC01_03")(x)
    x = layers.LeakyReLU(alpha=0.2, name="DA01_03")(x)
    x = layers.UpSampling2D((2, 2), name="DU1")(x)

    x = layers.Conv2D(32, (3, 3), padding="same", name="DC02_01")(x)
    x = layers.LeakyReLU(alpha=0.2, name="DA02_01")(x)

    x = layers.Conv2D(32, (3, 3), padding="same", name="DC02_02")(x)
    x = layers.LeakyReLU(alpha=0.2, name="DA02_02")(x)
    x = layers.UpSampling2D((2, 2), name="DU2")(x)

    x = layers.Conv2D(16, (3, 3), padding="same", name="DC03_01")(x)
    x = layers.LeakyReLU(alpha=0.2, name="DA03_01")(x)

    x = layers.Conv2D(16, (3, 3), padding="same", name="DC03_02")(x)
    x = layers.LeakyReLU(alpha=0.2, name="DA03_02")(x)
    x = layers.UpSampling2D((2, 2), name="DU3")(x)

    x = layers.Conv2D(16, (3, 3), padding="same", name="DC04_01")(x)
    x = layers.LeakyReLU(alpha=0.2, name="DCA4_01")(x)
    x = layers.UpSampling2D((2, 2), name="DU4")(x)

    x = layers.Conv2D(1, (3, 3), padding="same", name="DC05_01")(x)
    output = layers.Activation("tanh", name="DA05_01")(x)

    decoder = Model(inputs=inp, outputs=output)
    return decoder


def Time_dis_ENC(encoder):
    model = Sequential()
    model.add(layers.TimeDistributed(encoder, input_shape=(3, 128, 128, 1)))
    model.add(layers.TimeDistributed(layers.Flatten(), input_shape=(3, 1024)))
    return model


def Middle():
    encoder_inputs = layers.Input(shape=(3, 1024), name="Encodings_of_images_sequence")
    encoder = layers.LSTM(1024, input_shape=(3, 1024), return_state=True, name="Seq2Seq_encoder")
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = layers.Input((1, 1024), name="Fixed_zero_tensor")
    decoder_lstm = layers.LSTM(1024, return_sequences=True, return_state=False, name="Seq2Seq_decoder")
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model


def Generator(modelpre, modelmid, decoder):
    inp1 = layers.Input((3, 128, 128, 1), name="Normal_Image_Sequence")
    inp2 = layers.Input((1, 1024), name="Zero_Fixed_Tensor")
    x = modelpre(inp1)
    x = modelmid([x, inp2])
    x = layers.Reshape((4, 4, 64), name="Reshaping_layer")(x)
    output = decoder(x)

    model = Model([inp1, inp2], output)
    return model


def Discriminator():
    def DisEncoder():
        inp1 = layers.Input((128, 128, 1))
        x = layers.Conv2D(16, (3, 3), padding="same")(inp1)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
        x = layers.Conv2D(32, (3, 3), padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
        x = layers.Conv2D(64, (3, 3), padding="same")(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)

        encoder_dis = Model(inputs=inp1, outputs=x, name="Discriminator_Encoder")
        return encoder_dis

    encoder = DisEncoder()

    model = Sequential()
    model.add(layers.TimeDistributed(encoder, input_shape=(4, 128, 128, 1), name="Time_distributed_01"))
    model.add(layers.TimeDistributed(layers.Flatten(name="Flatten"), input_shape=(4, 1024), name="Time_distributed_02"))
    model.add(layers.SimpleRNN(128,
                               input_shape=(4, 1024),
                               return_sequences=False,
                               activation="relu",
                               name="RNN_layer"))
    model.add(layers.Dense(2, activation="softmax", name="Categorical_output"))

    return model


if __name__ == "__main__":
    Model_en = Encoder_Gene()
    Model_en.summary()
    Model_de = Decoder_Gene()
    Model_de.summary()
    Model_mid = Middle()
    Model_mid.summary()
    Model_dis = Discriminator()
    Model_dis.summary()
