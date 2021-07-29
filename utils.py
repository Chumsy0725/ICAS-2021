import os
import numpy as np
from keras.preprocessing.image import (
    ImageDataGenerator,
    array_to_img,
    img_to_array,
    load_img,
)


def normalize(image):
    """
    Normalize the given image.
    """
    image = image / 127.5 - 1
    return image


def lr_schedule(epoch):
    """
    Returns a custom learning rate that decreases as epochs progress.
    """
    learning_rate = 0.001
    if epoch > 30:
        learning_rate = 0.0001
    if epoch > 40:
        learning_rate *= 0.5

    return learning_rate


def Preprocessor(path0, slen=3, cap="\\", flip=False):
    folder = [path0]
    print('folder=', folder)
    train_files = [f for f in os.listdir(
        folder[0]) if os.path.isfile(os.path.join(folder[0], f))]
    print("Files in train_files: %d" % len(train_files))

    # Original Dimensions
    image_width = 128
    image_height = 128
    ratio = 1

    image_width = 128
    image_height = 128

    channels = 1
    nb_classes = 1

    dataset_x = np.ndarray(shape=(
        len(train_files), slen, image_height, image_width, channels), dtype=np.float32)
    dataset_y = np.ndarray(
        shape=(len(train_files), image_height, image_width, channels), dtype=np.float32)

    i = 0
    seq = []

    for num in range(1, len(train_files)):

        try:
            nums = str(num)
            img_path = folder[0] + cap + str(nums.zfill(5))+'.jpeg'
            #print('img_path:', img_path)
            img = load_img(img_path, grayscale=True,
                           target_size=(128, 128), interpolation="hamming")
            if flip:
                img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

            img.thumbnail((image_width, image_height))
            x = img_to_array(img)
            x = x.reshape((128, 128, 1))
            seq.append(x)

            if len(seq) != slen:
                continue
            else:
                dataset_x[i] = seq
                dataset_y[i] = x
                i += 1
                seq.pop(0)
        except OSError:
            continue
    dataset_x = dataset_x[:-1]
    dataset_y = dataset_y[1:]
    return dataset_x[:len(train_files)-slen-1], dataset_y[:len(train_files)-slen-1]


def get_disc_batch(x_batch, y_batch, generator_model, batch_counter, AE, state=False):

    # Create X_disc: alternatively only generated or real images
    # if batch_counter % 2 == 0:
    # Produce an output
    y_batch_fake = generator_model.predict(
        [x_batch, np.zeros((x_batch.shape[0], 1, 1024))])
    #y_batch_fake = np.array([y_batch_fake])

    if state:
        pass  # y_batch = np.array([y_batch])
    else:
        #x_batch = AE.predict(x_batch[0])
        y_batch = AE.predict(y_batch)
        #x_batch = x_batch[np.newaxis, ...]

    y_batch = y_batch[:, np.newaxis, ...]
    y_batch_fake = y_batch_fake[:, np.newaxis, ...]

    #y_batch = np.array([y_batch])

    #x_batch = x_batch[np.newaxis, ...]
    #print ('Prediction', y_batch.shape,y_batch_fake.shape)
    #print (x_batch.shape)

    labels_fake = np.zeros((x_batch.shape[0], 2), dtype=np.uint8)
    labels_fake[:, 0] = 1

    labels_real = np.zeros((x_batch.shape[0], 2), dtype=np.uint8)
    labels_real[:, 1] = 1

    print(x_batch.shape, y_batch.shape)
    batch = np.concatenate([x_batch, y_batch], axis=1)

    batch_fake = np.concatenate([x_batch, y_batch_fake], axis=1)
    #print (batch.shape,batch_fake.shape)

    disc_input = np.concatenate([batch, batch_fake], axis=0)
    labels = np.concatenate([labels_real, labels_fake], axis=0)

    #print( disc_input.shape, labels.shape)
    #print (disc_input.shape)
    #print (labels.shape)
    return disc_input, labels
