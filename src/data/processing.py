import tensorflow as tf
import numpy as np
from src.utils import encode_category, decode_category, is_point_in_bbox, \
    middle_point_from_bbox, bbox_from_middle_point


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
    :return: tuple, (bx, by, bw, bh) of the YOLO grid bounding box.
    """
    (img_height, img_width) = img_size
    (grid_x, grid_y, grid_width, grid_height) = grid
    (mp_x, mp_y) = middle_point
    (bbox_width, bbox_height) = bbox_resolution

    return (
        (mp_x * img_width - grid_x) / grid_width,
        (mp_y * img_height - grid_y) / grid_height,
        bbox_width * img_width / grid_width,
        bbox_height * img_height / grid_height,
    )


def _decode_yolo_grid_bbox(yolo_grid_bbox, img_size, grid):
    """
    Decode the YOLO grid bounding box to the bounding box.

    :param yolo_grid_bbox: tuple, (bx, by, bw, bh) of the YOLO grid
        bounding box.
    :param img_size: tuple, (img_height, img_width) of each image.
    :param grid: tuple, (grid_x, grid_y, grid_width, grid_height) of
        the grid cell.
    :return:
        bbox: tuple, (x, y, width, height) of the bounding box.
        middle_point: tuple, (x, y) middle point of the bounding box.
    """
    (img_height, img_width) = img_size
    (bx, by, bw, bh) = yolo_grid_bbox
    (grid_x, grid_y, grid_width, grid_height) = grid

    mp_x = bx * grid_width + grid_x
    mp_y = by * grid_height + grid_y
    middle_point = (mp_x / img_width, mp_y / img_height)
    bbox_width = bw * grid_width / img_width
    bbox_height = bh * grid_height / img_height

    return (
        bbox_from_middle_point(middle_point, bbox_width, bbox_height),
        middle_point
    )


def encode_anns_to_yolo(anns, img_size, grid_size, categories=None):
    """
    Encode annotations to the YOLO format.

    :param anns: list of annotations, annotations of one image.
    :param img_size: tuple, (img_height, img_width) of each image.
    :param grid_size: tuple, number of (grid_rows, grid_cols) of grid
        cell.
    :param categories: np.array dim=(n_categories) (default: None),
        string vector of categories. If `None` then there will be only
        bounding boxes, without category labels.
    :return: np.array dim=(grid_height, grid_width, 5 + n_categories),
        annotations in the YOLO format.
    """
    (img_height, img_width) = img_size
    (grid_rows, grid_cols) = grid_size
    grid_height = int(img_height / grid_rows)
    grid_width = int(img_width / grid_cols)

    categories_len = 0 if categories is None else len(categories)

    grid_arr = []
    for grid_y in range(0, img_height, grid_height):
        yolo_arr = []

        for grid_x in range(0, img_width, grid_width):
            grid = (grid_x, grid_y, grid_width, grid_height)

            for ann in anns:
                middle_point = (ann[2][0] * img_width, ann[2][1] * img_height)

                if (categories is None or ann[1] in categories) \
                        and is_point_in_bbox(grid, middle_point):

                    yolo_grid_bbox = _encode_yolo_grid_bbox(
                        ann[2],
                        ann[0][2:],
                        img_size,
                        grid
                    )

                    encoded_category = []
                    if categories is not None:
                        encoded_category = encode_category(categories, ann[1])

                    yolo = [1.0, *yolo_grid_bbox, *encoded_category]
                    break
            else:
                yolo = [0.0] * (5 + categories_len)
            yolo_arr.append(yolo)
        grid_arr.append(yolo_arr)
    return np.array(grid_arr, dtype=np.float32)


def decode_yolo_to_anns(yolo_anns, img_size, grid_size, categories=None,
                        confidence_threshold=0.0):
    """
    Decode annotations in the YOLO format to default annotation format.

    :param yolo_anns: np.array dim=(grid_height, grid_width,
        5 + n_categories), annotations in the YOLO format.
    :param img_size: tuple, (img_height, img_width) of each image.
    :param grid_size: tuple, number of (grid_rows, grid_cols) of grid
        cell.
    :param categories: np.array dim=(n_categories) (default: None),
        string vector of categories. If `None` then there will be only
        bounding boxes, without category labels.
    :param confidence_threshold: float (default: 0.0), only annotations
        where confidence is higher than threshold will be returned.
    :return: list of annotations, annotations of one image.
    """
    (img_height, img_width) = img_size
    (grid_rows, grid_cols) = grid_size
    grid_height = int(img_height / grid_rows)
    grid_width = int(img_width / grid_cols)

    anns = []
    for i, row in enumerate(yolo_anns):
        grid_y = i * grid_height

        for j, col in enumerate(row):
            grid_x = j * grid_width
            grid = (grid_x, grid_y, grid_width, grid_height)

            if col[0] > confidence_threshold:
                (bbox, middle_point) = _decode_yolo_grid_bbox(
                    col[1:5],
                    img_size,
                    grid
                )
                category = '' if categories is None else \
                    decode_category(categories, col[5:])
                anns.append([bbox, category, middle_point])
    return anns


def input_fn(imgs, anns, is_training=True, batch_size=16):
    """
    Data pipeline using TensorFlow Dataset class.

    :param imgs: np.array dim=(n_images, height, width, n_channels),
        images.
    :param anns: np.array dim=(grid_height, grid_width, 5 +
        n_categories), annotations in the YOLO format.
    :param is_training: bool (default: True), whether it's train
        dataset or not. If it's train then perform shuffle and repeat
        otherwise no.
    :param batch_size: int (default: 16), batch size of dataset.
    :return: Dataset dim=((None, 260, 260, 3), (None, 13, 13, 6))
    """

    # Normalize to (0.0, 1.0)
    imgs = imgs / 255

    dataset = tf.data.Dataset.from_tensor_slices((imgs, anns))

    # Shuffle and repeat
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(anns)).repeat()

    # Batch
    dataset = dataset.batch(
        batch_size,
        drop_remainder=False
    )

    # Prefetch
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def create_dataset(x, y, img_size, grid_size, **kwargs):
    """

    :param x:
    :param y:
    :param img_size:
    :param grid_size:
    :param kwargs:
    :return:
    """

    new_x = resize_images(x, img_size)
    new_y = calculate_bboxes_middle_points(y)

    yolo_anns = np.array([
        encode_anns_to_yolo(anns, img_size, grid_size)
        for anns in new_y
    ])

    dataset = input_fn(
        new_x,
        yolo_anns,
        **kwargs,
    )

    return dataset
