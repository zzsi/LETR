"""
Split the York Urban dataset into train and val.

In the project root directory:

python ../helper/york_split.py ./YorkUrbanDB ./york_lines
"""


import os
import sys
import glob
import json
import os.path as osp
from itertools import combinations

import cv2
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from scipy.io import loadmat
from scipy.ndimage import zoom
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument('src_dir', type=str)
    parser.add_argument('dst_dir', type=str)
    args = parser.parse_args()

    src_dir = args.src_dir
    dataset = list(sorted(glob.glob(osp.join(src_dir, "*/*.jpg"))))
    print(f"dataset has {len(dataset)} images")
    splitted_ids = loadmat(os.path.join(src_dir, "ECCV_TrainingAndTestImageNumbers.mat"))
    train_image_ids = (splitted_ids["trainingSetIndex"] - 1).ravel().tolist()
    val_image_ids = (splitted_ids["testSetIndex"] - 1).ravel().tolist()
    train_image_paths = [dataset[i] for i in train_image_ids]
    val_image_paths = [dataset[i] for i in val_image_ids]

    image_id = 0
    anno_id = 0
        
    os.makedirs(os.path.join(args.dst_dir, "images"), exist_ok=True)

    anno = {}
    anno['images'] = []
    anno['annotations'] = []
    anno['categories'] = [{'supercategory': "line", "id": "0", "name": "line"}]

    def handle(iname, image_id, anno_id, batch):

        im = cv2.imread(iname)
        filename = iname.split("/")[-1]

        anno['images'].append({'file_name': filename,
                            'height': im.shape[0], 'width': im.shape[1], 'id': image_id})
        mat = loadmat(iname.replace(".jpg", "LinesAndVP.mat"))
        lines = np.array(mat["lines"]).reshape(-1, 2, 2)
        lines = lines.astype('float')
        image_path = os.path.join(args.dst_dir, "images", filename)
        line_set = save_and_process(f"{image_path}", filename, im[::, ::], lines)
        for line in line_set:
            info = {}
            info['id'] = anno_id
            anno_id += 1
            info['image_id'] = image_id
            info['category_id'] = 0
            info['line'] = line
            info['area'] = 1
            anno['annotations'].append(info)

        image_id += 1
        # print(f"Finishing {image_path}")
        return anno_id

    
    os.makedirs(os.path.join(args.dst_dir, "annotations"), exist_ok=True)
    for img, image_id in zip(train_image_paths, train_image_ids):
        anno_id = handle(img, image_id, anno_id, "train")    
    anno_path = os.path.join(args.dst_dir, "annotations", f"lines_train.json")
    with open(anno_path, 'w') as outfile:
        json.dump(anno, outfile, indent=2)
    
    anno['images'] = []
    anno['annotations'] = []
    for img, image_id in zip(val_image_paths, val_image_ids):
        anno_id = handle(img, image_id, anno_id, "val")
    anno_path = os.path.join(args.dst_dir, "annotations", f"lines_val.json")
    with open(anno_path, 'w') as outfile:
        json.dump(anno, outfile, indent=2)


def save_and_process(image_path, image_name, image, lines):
    # change the format from x,y,x,y to x,y,dx, dy
    # order: top point > bottom point
    #        if same y coordinate, right point > left point

    new_lines_pairs = []
    for line in lines:  # [ #lines, 2, 2 ]
        p1 = line[0]    # xy
        p2 = line[1]    # xy
        if p1[0] < p2[0]:
            new_lines_pairs.append([p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1]])
        elif p1[0] > p2[0]:
            new_lines_pairs.append([p2[0], p2[1], p1[0]-p2[0], p1[1]-p2[1]])
        else:
            if p1[1] < p2[1]:
                new_lines_pairs.append(
                    [p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1]])
            else:
                new_lines_pairs.append(
                    [p2[0], p2[1], p1[0]-p2[0], p1[1]-p2[1]])

    cv2.imwrite(f"{image_path}", image)
    return new_lines_pairs

if __name__ == "__main__":
    main()
