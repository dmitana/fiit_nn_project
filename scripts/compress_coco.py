#!/usr/bin/env python3

import argparse
import os
import numpy as np
from tqdm import tqdm
import matplotlib.image as mpimg
from pycocotools.coco import COCO
import tensorflow as tf

# Initialize argument parser
parser = argparse.ArgumentParser(
    description='Compress the COCO dataset to a format that is easier '
                'to load.'
)
parser.add_argument(
    'imgs_dir_path',
    type=str,
    help='Path to the images directory.'
)
parser.add_argument(
    'anns_file_path',
    type=str,
    help='Path to the corresponding annotations file.'
)
parser.add_argument(
    'target_dir',
    type=str,
    help='Path to the target directory where the dataset will be '
         'created.'
)
parser.add_argument(
    'name',
    type=str,
    help='Name of the dataset file.'
)
parser.add_argument(
    '-n', '--n-examples',
    type=int,
    default=None,
    help='Number of examples to be placed to the dataset. If not '
         'provided then all examples will be included.'
)
parser.add_argument(
    '-c', '--categories',
    nargs='*',
    default=None,
    help='List of categories of objects separated by space which '
         'image has to contain to be added to compressed dataset.'
         'If not provided then all categories will be included.'
)
parser.add_argument(
    '-s', '--img-size',
    type=int,
    nargs=2,
    default=None,
    help='Resolution to which images will be resized in the format '
         '`height width`.'
)
parser.add_argument(
    '-r', '--reverse',
    action='store_true',
    help='Flag which specifies whether images will be loaded in '
         'ascending or descending order. If not used, default is '
         'ascending order. '
)


def compress_coco(imgs_dir_path, anns_file_path, target_dir, name,
                  n_examples=None, categories=None, img_size=None,
                  reverse=False):
    """
    Compress given COCO dataset to more readable format.

    :param imgs_dir_path: str, path to the images directory.
    :param anns_file_path: str, path to the corresponding annotations
        file.
    :param target_dir: str, path to the target directory where the
        dataset will be created.
    :param name: str, name of the dataset file.
    :param n_examples: str (default: None), number of examples to be
        placed to the dataset. If `None` then all examples will be
        included.
    :param categories: list, only categories from this list will be
        added to compressed dataset.
    :param img_size: tuple, contains new height and width of images.
    :return: str, path to the compressed dataset.
    """
    imgs_filenames = sorted(os.listdir(imgs_dir_path), reverse=reverse)

    c = COCO(annotation_file=anns_file_path)
    x, y = [], []

    if n_examples:
        print(f'Loading {n_examples} images and annotations.')
    else:
        print(f'Loading {len(imgs_filenames)} images and annotations.')

    if categories is not None:
        print(f'Picking following categories: {categories}')

    tqdm_len = n_examples if n_examples else len(imgs_filenames)
    with tqdm(total=tqdm_len) as pbar:
        for i, img_filename in enumerate(imgs_filenames):
            # Load image
            img_file_path = os.path.join(imgs_dir_path, img_filename)
            img = mpimg.imread(img_file_path)
            if len(img.shape) == 2:
                img = np.stack((img,) * 3, axis=-1)

            # Resize image
            (original_height, original_width, _) = img.shape
            if img_size is not None:
                img = tf.image.resize(
                    images=img,
                    size=img_size,
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
                ).numpy()

            # Load annotations
            img_id = int(img_filename[:-4])
            ann_list = []
            ann_ids = c.getAnnIds(imgIds=[img_id])
            anns = c.loadAnns(ids=ann_ids)
            for ann in anns:
                label = c.loadCats(ids=[ann['category_id']])[0]['name']
                if categories is not None and label not in categories:
                    continue

                bbox = ann['bbox']

                # Scale bbox x, y, width, height
                bbox = [
                    bbox[0] / original_width,
                    bbox[1] / original_height,
                    bbox[2] / original_width,
                    bbox[3] / original_height
                ]

                ann_list.append([bbox, label])

            if len(ann_list) > 0:
                x.append(img)
                y.append(ann_list)
                pbar.update(1)

            if n_examples and len(x) == n_examples:
                break

    if categories is not None:
        print(f'Found {len(y)} images which contain some of following '
              f'objects: {categories}')

    # Save dataset to .npz
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    dataset_file_path = os.path.join(target_dir, name + '.npz')
    print(f'Save {name} dataset to {dataset_file_path}.')
    np.savez_compressed(dataset_file_path, x=x, y=y)

    return dataset_file_path


if __name__ == '__main__':
    args = parser.parse_args()
    opts = {
        'imgs_dir_path': args.imgs_dir_path,
        'anns_file_path': args.anns_file_path,
        'target_dir': args.target_dir,
        'name': args.name,
        'n_examples': args.n_examples,
        'categories': args.categories,
        'img_size': args.img_size,
        'reverse': args.reverse
    }
    compress_coco(**opts)
