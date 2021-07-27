from tensorflow.keras import layers
from tensorflow.keras.models import Model


def Encoder_Gene():
    inp1 = layers.Input((128, 128, 1), name='Image_Input')

    x = layers.Conv2D(16, (3, 3), padding='same', name='EC01_1')(inp1)
    x = layers.LeakyReLU(alpha=0.2, name='EA01_1')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='EP01')(x)

    x = layers.Conv2D(32, (3, 3), padding='same', name='EC02_1')(x)
    x = layers.LeakyReLU(alpha=0.2, name='EA02_1')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='EP02')(x)

    x = layers.Conv2D(64, (3, 3), padding='same', name='EC03_1')(x)
    x = layers.LeakyReLU(alpha=0.2, name='EA03_1')(x)

    x = layers.Conv2D(64, (3, 3), padding='same', name='EC03_2')(x)
    x = layers.LeakyReLU(alpha=0.2, name='EA03_2')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='EP03')(x)

    x = layers.Conv2D(64, (3, 3), padding='same', name='EC04_1')(x)
    x = layers.LeakyReLU(alpha=0.2, name='EA04_1')(x)

    x = layers.Conv2D(64, (3, 3), padding='same', name='EC04_2')(x)
    x = layers.LeakyReLU(alpha=0.2, name='EA04_2')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='EP04')(x)

    x = layers.Conv2D(64, (3, 3), padding='same', name='EC05_1')(x)
    x = layers.LeakyReLU(alpha=0.2, name='EA05_1')(x)

    x = layers.Conv2D(64, (3, 3), padding='same', name='EC05_2')(x)
    x = layers.LeakyReLU(alpha=0.2, name='EA05_2')(x)

    x = layers.Conv2D(64, (3, 3), padding='same', name='EC05_3', activation='tanh')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='EP05')(x)

    encoder = Model(inputs=inp1, outputs=x)
    return encoder


def Decoder_Gene():
    inp = layers.Input((4, 4, 64), name='Latent_Representation')
    x = layers.UpSampling2D((2, 2), name='DU0')(inp)

    x = layers.Conv2D(64, (3, 3), padding='same', name='DC01_01')(x)
    x = layers.LeakyReLU(alpha=0.2, name='DA01_01')(x)

    x = layers.Conv2D(64, (3, 3), padding='same', name='DC01_02')(x)
    x = layers.LeakyReLU(alpha=0.2, name='DA01_02')(x)

    x = layers.Conv2D(64, (3, 3), padding='same', name='DC01_03')(x)
    x = layers.LeakyReLU(alpha=0.2, name='DA01_03')(x)
    x = layers.UpSampling2D((2, 2), name='DU1')(x)

    x = layers.Conv2D(32, (3, 3), padding='same', name='DC02_01')(x)
    x = layers.LeakyReLU(alpha=0.2, name='DA02_01')(x)

    x = layers.Conv2D(32, (3, 3), padding='same', name='DC02_02')(x)
    x = layers.LeakyReLU(alpha=0.2, name='DA02_02')(x)
    x = layers.UpSampling2D((2, 2), name='DU2')(x)

    x = layers.Conv2D(16, (3, 3), padding='same', name='DC03_01')(x)
    x = layers.LeakyReLU(alpha=0.2, name='DA03_01')(x)

    x = layers.Conv2D(16, (3, 3), padding='same', name='DC03_02')(x)
    x = layers.LeakyReLU(alpha=0.2, name='DA03_02')(x)
    x = layers.UpSampling2D((2, 2), name='DU3')(x)

    x = layers.Conv2D(16, (3, 3), padding='same', name='DC04_01')(x)
    x = layers.LeakyReLU(alpha=0.2, name='DCA4_01')(x)
    x = layers.UpSampling2D((2, 2), name='DU4')(x)

    x = layers.Conv2D(1, (3, 3), padding='same', name='DC05_01')(x)
    output = layers.Activation('tanh', name='DA05_01')(x)

    decoder = Model(inputs=inp, outputs=output)
    return decoder


if __name__ == '__main__':
    Model_en = Encoder_Gene()
    Model_en.summary()
    Model_de = Decoder_Gene()
    Model_de.summary()
