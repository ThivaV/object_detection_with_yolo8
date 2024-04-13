# https://github.com/EscVM/OIDv4_ToolKit

import os
import shutil

apple_id = '/m/014j1m'

class_ids = [apple_id]

DATA_MASTER_DIR = os.path.join('../../data/raw/open-images-v7', 'apple')
DATA_OUTPUT_DIR = os.path.join('../../data/', 'processed')

for set_ in ['train', 'validation', 'test']:
    for dir_ in [os.path.join(DATA_OUTPUT_DIR, set_), 
                 os.path.join(DATA_OUTPUT_DIR, set_, 'images'), os.path.join(DATA_OUTPUT_DIR, set_, 'labels')]:
        if os.path.exists(dir_):
            shutil.rmtree(dir_)
        os.mkdir(dir_)

train_bboxes_filename = os.path.join(DATA_MASTER_DIR, 'csv_folder', 'train-annotations-bbox.csv')
validation_bboxes_filename = os.path.join(DATA_MASTER_DIR, 'csv_folder', 'validation-annotations-bbox.csv')
test_bboxes_filename = os.path.join(DATA_MASTER_DIR, 'csv_folder', 'test-annotations-bbox.csv')

for j, filename in enumerate([train_bboxes_filename, validation_bboxes_filename, test_bboxes_filename]):
    print(filename)
    set_ = ['train', 'validation', 'test'][j]

    with open(filename, 'r') as f:
        line = f.readline()

        while len(line) != 0:
            id, _, class_name, _, x1, x2, y1, y2, _, _, _, _, _ = line.split(',')[:13]

            if class_name in class_ids:
                img_from = os.path.join(DATA_MASTER_DIR, set_, 'images', '{}.jpg'.format(id))
                img_to = os.path.join(DATA_OUTPUT_DIR, set_, 'images', '{}.jpg'.format(id))

                if os.path.exists(img_from):
                    if not os.path.exists(img_to):
                        shutil.copy(img_from, img_to)
                    
                    with open(os.path.join(DATA_OUTPUT_DIR, set_, 'labels', '{}.txt'.format(id)), 'a') as f_ann:
                        # class_id, xc, yx, w, h
                        x1, x2, y1, y2 = [float(j) for j in [x1, x2, y1, y2]]
                        xc = (x1 + x2) / 2
                        yc = (y1 + y2) / 2
                        w = x2 - x1
                        h = y2 - y1

                        f_ann.write('0 {} {} {} {}\n'.format(xc, yc, w, h))
                        f_ann.close()

            line = f.readline()