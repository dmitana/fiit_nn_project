import tensorflow as tf

def resize_images(images, size):
    """
    Resize `images` to given `size`.

    :param images: np.array dim=(batch, height, width, channels),
        images to resize.
    :param size: tuple, contains new height and width of images.
    :return: Tensor dim=(batch, new_height, new_width, channels),
        resized images.
    """
    images_new = []
    for img in images:
        images_new.append(
            tf.image.resize(
                images=tf.convert_to_tensor(img, dtype=tf.uint8),
                size=size,
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
        )
    return tf.convert_to_tensor(images_new)