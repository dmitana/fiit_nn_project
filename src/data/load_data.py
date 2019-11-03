import numpy as np

def load_dataset(path):
    """
    Load dataset from pickled object arrays compressed in `.npz` format.

    Example of use: dev_x, dev_y = load_dataset('data/dev.npz')

    :param path: str, path to the dataset compressed `.npz` file.
    :return:
        x: np.array dim=(n_images, height, width, n_channels), images.
        y: np.array dim=(n_images,) of list of annotations, annotations.
    """
    with np.load(path, allow_pickle=True) as data:
        x, y = data['x'], data['y']
    return x, y