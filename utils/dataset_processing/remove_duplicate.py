import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='Remove duplicate images')
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