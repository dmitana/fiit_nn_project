import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_img_with_bboxes(img, anns, show_category=True, grid=None):
    """
    Plot image with bounding boxes for each object from annotations.

    :param img: np.array dim=(height, width), image to plot.
    :param anns: list of annotations, annotations for given image.
    :param show_category: bool (default: True), whether to plot
        category or not.
    :param grid: int|tuple (default: None), grid to plot. If int the
        grid is consider to be square.
    """
    (img_height, img_width, _) = img.shape
    fig, ax = plt.subplots(1)
    ax.imshow(img)

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
            edgecolor='y',
            facecolor='none'
        )
        ax.add_patch(rect)

        if show_category:
            ax.text(x, y, ann[1], backgroundcolor='y')

        if len(ann) == 3:
            ax.plot(ann[2][0] * img_width, ann[2][1] * img_height, 'go')

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
