import models
import utils
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import argparse
from tensorflow.keras import backend as K

image_data_format = "channels_last"
K.set_image_data_format(image_data_format)


def l2_loss_mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

# AutoEncoder for Actual Occurence reconstruction


encoder = models.Encoder_Gene()
#encoder.name = 'Convolutional_encoder'
decoder = models.Decoder_Gene()
#decoder.name = 'Convolutional_decoder'
encoder.load_weights("AE_weights\encoder.h5")
decoder.load_weights("AE_weights\decoder.h5")
rms = tf.keras.optimizers.Adam(lr=0.0001)
encoder.compile(optimizer=rms, loss='mse')
decoder.compile(optimizer=rms, loss='mse')

ultra = tf.keras.layers.Input((128, 128, 1), name='Normal_Image_Input')
x = encoder(ultra)
out = decoder(x)
AE = Model(inputs=ultra, outputs=out)
#AE.name = 'Convolutional Autoencoder'
rms = tf.keras.optimizers.Adam(lr=0.0001)
AE.compile(optimizer=rms, loss='mse')
AE.summary()

# custom function to train the generative adversarial network


def train_GAN(data, AE, args):

    x_train, y_train = data

    n_batch_per_epoch = x_train.shape[0] // args.batch_size
    epoch_size = n_batch_per_epoch * args.batch_size
    img_dim = x_train.shape[-3:]

    if args.opt == "adam":
        opt_dcgan = tf.keras.optimizers.Adam(lr=args.lr, beta_1=args.beta1,
                                             beta_2=0.999, epsilon=1e-08)
        opt_discriminator = tf.keras.optimizers.Adam(
            lr=args.lr, beta_1=args.beta1, beta_2=0.999, epsilon=1e-08
        )
    elif args.opt == "sgd":
        opt_dcgan = tf.keras.optimizers.SGD(
            lr=args.lr, momentum=0.0, decay=0.0, nesterov=False)
        opt_discriminator = tf.keras.optimizers.SGD(lr=args.lr, momentum=0.0,
                                                    decay=0.0, nesterov=False)
    else:
        print("optimizer sgd or adam")

    enc_gen = models.Encoder_Gene()
    #enc_gen.name = "Convolutional_Encoder"
    enc_gen.load_weights(
        "AE_weights\encoder.h5")

    enc_time = models.Time_dis_ENC(enc_gen)
    #enc_time.name = "Time_Distributed_Convolutional_Encoder"

    dec_gen = models.Decoder_Gene()
    #dec_gen.name = "Convolutional_Decoder"
    dec_gen.load_weights(
        "AE_weights\decoder.h5")

    RNN_middle = models.Middle()
    #RNN_middle.name = "Seq2Seq"
    RNN_middle.load_weights(
        "RNN_weights\tanh_seq2seq.h5")

    generator_model = models.Generator(enc_time, RNN_middle, dec_gen)
    #generator_model.name = "Generator"
    generator_model.compile(loss=l2_loss_mse, optimizer=opt_discriminator)

    discriminator_model = models.Discriminator()
    #discriminator_model.name = "Discriminator"
    discriminator_model.trainable = False

    DCGAN_model = models.DCGAN(generator_model, discriminator_model)
    #DCGAN_model.name = "DCGAN"

    loss = ["mse", "categorical_crossentropy"]
    loss_weights = [args.l1_weight, args.gan_weight]
    DCGAN_model.compile(
        loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

    discriminator_model.trainable = True
    discriminator_model.layers[0].trainable = False
    print("Discriminator Status")

    discriminator_model.compile(
        loss="categorical_crossentropy", optimizer=opt_discriminator
    )

    gen_loss = 100
    disc_loss = 100

    # Losses to accumulate over bathces
    disc_loss_acc = []
    gen_loss_l1_acc = []
    gen_loss_log_acc = []

    # Losses to accumulate over epochs
    disc_loss_acc_epc = []
    gen_loss_l2_acc_epc = []
    gen_loss_log_acc_epc = []

    # Start training
    print("Start training")
    tf.global_variables_initializer()
    for e in range(args.nb_epoch):
        # Initiate progress bar and batch counter
        progbar = generic_utils.Progbar(epoch_size)
        batch_counter = 1
        start = time.time()
        for i in range(n_batch_per_epoch):
            index = i * args.batch_size
            x_batch = x_train[index: index + args.batch_size]
            y_batch = y_train[index: index + args.batch_size]

            #x_batch = utils.normalize(x_batch.astype(np.float32))
            #y_batch = utils.normalize(y_batch.astype(np.float32))

            # Create a batch to feed the discriminator model
            disc_input, labels = utils.get_disc_batch(
                x_batch, y_batch, generator_model, batch_counter, AE, state=True
            )

            # Update the discriminator
            disc_loss = discriminator_model.train_on_batch(disc_input, labels)

            # Create a batch to feed the generator model
            x_gen, y_gen_target = x_batch, y_batch
            labels = np.zeros((x_gen.shape[0], 2), dtype=np.uint8)
            labels[:, 1] = 1

            # Freeze the discriminator
            discriminator_model.trainable = False
            gen_loss = DCGAN_model.train_on_batch(
                [x_gen, np.zeros((x_gen.shape[0], 1, 1024))], [
                    y_gen_target, labels]
            )

            # Unfreeze the discriminator
            discriminator_model.trainable = True

            progbar.add(
                args.batch_size,
                values=[
                    ("D logloss", disc_loss),
                    ("G tot", gen_loss[0]),
                    ("G L2L1", gen_loss[1]),
                    ("G logloss", gen_loss[2]),
                ],
            )

            # Show prediction on current batch and plot losses
            if batch_counter % (n_batch_per_epoch / 8) == 0:

                # accumulate losses to plot
                disc_loss_acc.append(disc_loss)
                gen_loss_l1_acc.append(gen_loss[1])
                gen_loss_log_acc.append(gen_loss[2])

                # plt and save losses upto current batch
                plt.plot(disc_loss_acc)
                plt.plot(gen_loss_l1_acc)
                plt.plot(gen_loss_log_acc)

                plt.legend(
                    [
                        "dicriminator log loss",
                        "generator l1l2 loss",
                        "generator log loss",
                    ]
                )
                plt.savefig("results/loss.png")
                plt.clf()
                plt.close()

            if batch_counter >= n_batch_per_epoch:
                break
            batch_counter += 1

        # accumulate losses to plot over epoch
        disc_loss_acc_epc.append(disc_loss_acc[-1])
        gen_loss_l2_acc_epc.append(gen_loss_l1_acc[-1])
        gen_loss_log_acc_epc.append(gen_loss_log_acc[-1])

        # plt and save losses upto current epoch
        plt.plot(disc_loss_acc_epc)
        plt.plot(gen_loss_l2_acc_epc)
        plt.plot(gen_loss_log_acc_epc)

        plt.legend(["dicriminator log loss",
                   "generator l2 loss", "generator log loss"])
        plt.savefig("results/loss over epoch.png")
        plt.clf()
        plt.close()

        print("")
        print("Epoch %s/%s, Time: %s" %
              (e + 1, args.nb_epoch, time.time() - start))
    # Final weights

    print("Saving Weights")

    enc_gen.save_weights("GAN_weights/enc_gen.h5")
    enc_time.save_weights(
        "GAN_weights/enc_time.h5")
    dec_gen.save_weights("GAN_weights/dec_gen.h5")
    RNN_middle.save_weights(
        "GAN_weights/RNN_middle.h5")

    print("Weights saved")
    return


if __name__ == "__main__":

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Conditinal CAPSGAN.")

    parser.add_argument(
        "--opt", default="adam", choices=["adam", "sgd"], help="optimizer sgd or adam"
    )
    parser.add_argument(
        "-lw", "--load_weights", action="store_true", help="To load weights or not"
    )
    parser.add_argument(
        "-sigmoid",
        "--sigmoid",
        action="store_true",
        help="If true, gates will be added sigmoid instead softmax",
    )
    parser.add_argument(
        "-ng", "--no_gates", action="store_true", help="Have gates or not"
    )
    parser.add_argument(
        "--filters", default=8, type=int, help="Number of filters to start camnet"
    )
    parser.add_argument(
        "--weights", default="models/weights.h5", help="weight file location"
    )
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument("--nb_epoch", default=10,
                        type=int, help="Number of epochs")
    parser.add_argument(
        "--nb_split",
        default=10,
        type=int,
        help="Number of sub splits in individual dataset in combined approach",
    )
    parser.add_argument(
        "--lr", default=0.00001, type=float, help="Initial learning rate"
    )
    parser.add_argument(
        "--lr_decay",
        default=0.9,
        type=float,
        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs",
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="momentum term of adam"
    )
    parser.add_argument(
        "--l1_weight",
        type=float,
        default=100.0,
        help="weight on L1 term for generator gradient",
    )
    parser.add_argument(
        "--gan_weight",
        type=float,
        default=1.0,
        help="weight on GAN term for generator gradient",
    )
    parser.add_argument("--save_dir", default="./models")
    parser.add_argument("--log_dir", default="./log")
    parser.add_argument("--result_dir", default="./results")
    parser.add_argument(
        "--memfrac", type=float, default=0.9, help="Fraction of memory to use"
    )

    args = parser.parse_args(args=[])
    print(args)

    # Limit memory use
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.memfrac
    set_session(tf.Session(config=config))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    Dataset_X = []
    Dataset_Y = []
    paths = ['data/train/icab1/front_camera', 'data/train/icab1/back_camera',
             'data/train/icab1/side_camera', 'data/train/icab2']
    for path0 in paths:
        data_x, data_y = utils.Preprocessor(path0, slen=3, cap="/", flip=False)
        Dataset_X.extend(data_x)
        Dataset_Y.extend(data_y)

    train_GAN(data=(Dataset_X, Dataset_Y), AE=AE, args=args)
