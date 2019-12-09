import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_img_with_bboxes(img, anns, true_anns=None, show_category=True, grid=None):
    """
    Plot image with bounding boxes for each object from annotations.

    :param img: np.array dim=(height, width), image to plot.
    :param anns: list of annotations, annotations for given image.
    :param true_anns: list of annotations, ground truth annotation for
        given image.
    :param show_category: bool (default: True), whether to plot
        category or not.
    :param grid: int|tuple (default: None), grid to plot. If int the
        grid is consider to be square.
    """
    def plot_anns(anns, color='y'):
        for ann in anns:
            # Bounding box
            x = ann[0][0] * img_width
            y = ann[0][1] * img_height
            width = ann[0][2] * img_width
            height = ann[0][3] * img_height

            rect = patches.Rectangle(
                xy=(x, y),
                width=width,
                height=height,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)

            if show_category:
                ax.text(x, y, ann[1], backgroundcolor='y')

            if len(ann) == 3:
                ax.plot(ann[2][0] * img_width, ann[2][1] * img_height, 'go')

    (img_height, img_width, _) = img.shape
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    plot_anns(anns)
    plot_anns(true_anns, color='g')

    if grid:
        (grid_height, grid_width) = grid if type(grid) in (tuple, list) else \
            (grid, grid)
        height_step = int(img_height / grid_height)
        width_step = int(img_width / grid_width)

        for point_y in range(0, img_height, height_step):
            ax.axhline(point_y, color='red')
        for point_x in range(0, img_width, width_step):
            ax.axvline(point_x, color='red')


def middle_point_from_bbox(bbox):
    """
    Get middle point of the given bounding box.

    :param bbox: tuple, (x, y, width, height) of the bounding box.
    :return: tuple, (x, y) middle point of the given bounding box.
    """
    (x, y, width, height) = bbox
    return (
        x + width / 2,
        y + height / 2
    )


def bbox_from_middle_point(middle_point, width, height):
    """
    Retrieve the bounding box from it's middle point.

    :param middle_point: tuple, (x, y) middle point of the bounding box.
    :param width: float, width of the bounding box.
    :param height: float, height of the bounding box.
    :return: tuple, (x, y, width, height) of the bounding box.
    """
    (x, y) = middle_point
    return (
        x - width / 2,
        y - height / 2,
        width,
        height
    )


def is_point_in_bbox(bbox, point):
    """
    Check if the point is in the bounding box.

    :param bbox: tuple, (x, y, width, height) of the bounding box.
    :param point: tuple, (x, y) point to be decided.
    :return: bool, `True` if the `point` is in the `bbox` otherwise
        `False`.
    """
    (x, y, width, height) = bbox
    (point_x, point_y) = point

    if x <= point_x <= x + width and y <= point_y <= y + height:
        return True


def encode_category(categories, category):
    """
    Encode `category` into one-hot vector.

    :param categories: np.array dim=(n_categories), string vector of
        categories.
    :param category: str, category to be encoded.
    :return: np.array dim=(n_categories), one-hot vector of `category`.
    """
    return np.array(categories == category, dtype=np.float32)


def decode_category(categories, encoded_category):
    """
    Decode one-hot vector `encoded_category` into string.

    :param categories: np.array dim=(n_categories), string vector of
        categories.
    :param encoded_category: np.array dim=(n_categories), one-hot
        vector of category.
    :return: str, decoded category.
    """
    return categories[np.argmax(encoded_category)]


def timestamp():
    """
    Return timestamp in format Y-m-d-H-M.

    :return: str, string timestamp.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
