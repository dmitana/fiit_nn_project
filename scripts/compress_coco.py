#!/usr/bin/env python3

import argparse
import os
import numpy as np
from tqdm import tqdm
import matplotlib.image as mpimg
from pycocotools.coco import COCO

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


def compress_coco(imgs_dir_path, anns_file_path, target_dir, name,
                  n_examples=None):
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
    :return: str, path to the compressed dataset.
    """
    imgs_filenames = sorted(os.listdir(imgs_dir_path))
    if n_examples:
        imgs_filenames = imgs_filenames[:n_examples]

    c = COCO(annotation_file=anns_file_path)
    x, y = [], []
    print(f'Loading {len(imgs_filenames)} images and annotations.')
    for i, img_filename in enumerate(tqdm(imgs_filenames)):
        # Load image
        img_file_path = os.path.join(imgs_dir_path, img_filename)
        img = mpimg.imread(img_file_path)
        if len(img.shape) == 2:
            img = np.stack((img,) * 3, axis=-1)
        (height, width, _) = img.shape
        x.append(img)

        # Load annotations
        img_id = int(img_filename[:-4])
        ann_list = []
        ann_ids = c.getAnnIds(imgIds=[img_id])
        anns = c.loadAnns(ids=ann_ids)
        for ann in anns:
            label = c.loadCats(ids=[ann['category_id']])[0]['name']
            bbox = ann['bbox']

            # Scale bbox x, y, width, height
            bbox = [
                bbox[0] / width,
                bbox[1] / height,
                bbox[2] / width,
                bbox[3] / height
            ]

            ann_list.append([bbox, label])
        y.append(ann_list)

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
        'n_examples': args.n_examples
    }
    compress_coco(**opts)
