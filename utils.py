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
