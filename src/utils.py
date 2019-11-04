import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_img_with_bboxes(img, anns):
    """
    Plot image with bounding boxes for each object from annotations.

    :param img: np.array dim=(height, width), image to plot.
    :param anns: list of annotations, annotations for given image.
    """
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    (height, width, _) = img.shape

    for ann in anns:
        rect = patches.Rectangle(
            xy=(ann[0][0] * width, ann[0][1] * height),
            width=ann[0][2] * width,
            height=ann[0][3] * height,
            linewidth=2,
            edgecolor='y',
            facecolor='none'
        )
        ax.add_patch(rect)