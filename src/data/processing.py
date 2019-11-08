import tensorflow as tf
import numpy as np
from src.utils import encode_category, is_point_in_bbox, middle_point_from_bbox


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


def _encode_yolo_grid_bbox(middle_point, bbox_resolution, img_size, grid):
    """
    Encode the bounding box and it's middle point to the YOLO grid
    bounding box.

    :param middle_point: tuple, (x, y) middle point of the bounding box.
    :param bbox_resolution: tuple, (bbox_width, bbox_height) of the
        bounding box.
    :param img_size: tuple, (img_height, img_width) of each image.
    :param grid: tuple, (grid_x, grid_y, grid_width, grid_height) of
        the grid cell.
    :return: tuple, (bx, by, bh, bw) of the YOLO grid bounding box.
    """
    (img_height, img_width) = img_size
    (grid_x, grid_y, grid_width, grid_height) = grid
    (mp_x, mp_y) = middle_point
    (bbox_width, bbox_height) = bbox_resolution

    return (
        (mp_x * img_width - grid_x) / grid_width,
        (mp_y * img_height - grid_y) / grid_height,
        bbox_height / grid_height,
        bbox_width / grid_width,
    )


def encode_anns_to_yolo(anns, img_size, grid_size, categories):
    """
    Encode annotations to the YOLO format.

    :param anns: np.array dim=(n_images,) of list of annotations,
        annotations.
    :param img_size: tuple, (img_height, img_width) of each image.
    :param grid_size: tuple, number of (grid_rows, grid_cols) of grid
        cell.
    :param categories: np.array dim=(n_categories), string vector of
        categories.
    :return: np.array dim=(grid_height, grid_width, 5 + n_categories),
        annotations in the YOLO format.
    """
    (img_height, img_width) = img_size
    (grid_rows, grid_cols) = grid_size
    grid_height = int(img_height / grid_rows)
    grid_width = int(img_width / grid_cols)

    grid_arr = []
    for grid_y in range(0, img_height, grid_height):
        yolo_arr = []

        for grid_x in range(0, img_width, grid_width):
            grid = (grid_x, grid_y, grid_width, grid_height)

            for ann in anns:
                middle_point = (ann[2][0] * img_width, ann[2][1] * img_height)

                if ann[1] in categories and \
                        is_point_in_bbox(grid, middle_point):
                    yolo_grid_bbox = _encode_yolo_grid_bbox(
                        ann[2],
                        ann[0][2:],
                        img_size,
                        grid
                    )
                    encoded_category = encode_category(categories, ann[1])
                    yolo = [1.0, *yolo_grid_bbox, *encoded_category]
                    break
            else:
                yolo = [0.0] * (5 + len(categories))
            yolo_arr.append(yolo)
        grid_arr.append(yolo_arr)
    return np.array(grid_arr)
