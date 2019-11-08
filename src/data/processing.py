import tensorflow as tf
import numpy as np
from src.utils import middle_point_from_bbox


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


def calculate_bboxes_middle_points(y):
    """
    Calculate middle points of bounding boxes in annotations.

    :param y: np.array dim=(n_images,) of list of annotations,
        annotations.
    :return: np.array dim=(n_images,) of list of annotations,
        annotations with middle points for each bounding box.
    """
    new_y = []
    for anns in y:
        new_anns = []
        for ann in anns:
            middle_point = middle_point_from_bbox(ann[0])
            new_ann = [ann[0], ann[1], middle_point]
            new_anns.append(new_ann)
        new_y.append(new_anns)
    return np.array(new_y)
