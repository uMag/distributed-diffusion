import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='Dataset Curator')
parser.add_argument('--dataset', type=str, default=None, help='Path to dataset')
args = parser.parse_args()

image_formats = ('.jpg','.jpeg','.png')

pathlist = Path(args.dataset).rglob('*')
for path in pathlist:
    extension = os.path.splitext(path)[1]
    if extension == ".txt":
        filename = os.path.splitext(path)[0]
        x = os.path.isfile(str(filename) + ".jpg") or os.path.isfile(str(filename) + ".png") or os.path.isfile(str(filename) + ".jpeg")
        if x is False:
            print("deleting " + str(path))
            os.remove(path)
    elif extension in image_formats:
        filename = os.path.splitext(path)[0]
        x = os.path.isfile(str(filename) + ".txt")
        if x is False:
            print("deleting " + str(path))
            os.remove(path)
    
index = 1
pathlist = Path(args.dataset).rglob('*.txt')
for path in pathlist:
    filename = os.path.splitext(path)[0]
    txt_new = str(index) + ".txt"
    os.rename(path, Path(args.dataset, txt_new))
    if os.path.isfile(str(filename) + ".jpg"):
        img_old = str(filename) + ".jpg"
        img_new = str(index) + ".jpg"
        os.rename(img_old, Path(args.dataset, img_new))
    elif os.path.isfile(str(filename) + ".png"):
        img_old = str(filename) + ".png"
        img_new = str(index) + ".png"
        os.rename(img_old, Path(args.dataset, img_new))
    elif os.path.isfile(str(filename) + ".jpeg"):
        img_old = str(filename) + ".jpeg"
        img_new = str(index) + ".jpeg"
        os.rename(img_old, Path(args.dataset, img_new))
    index = index + 1
