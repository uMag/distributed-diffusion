import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='Remove duplicate images')
parser.add_argument('--dataset', type=str, default=None, help='Path to dataset')
parser.add_argument('--option', type=str, default=None, help='option')
args = parser.parse_args()

image_formats = ('.jpg','.jpeg','.png')

def getPathlist(folder):
    pathlist = Path(folder).rglob('*')
    return pathlist

#Afects only images
def cleanUpFormats(folder):
    b = 0
    pathlist = getPathlist(folder)
    for path in pathlist:
        extension = os.path.splitext(path)[1]
        isImageFormat = extension in image_formats
        if isImageFormat is False:
            b = b + 1
            os.remove(path)
    print("Cleaned " + str(b) + " Files.")

#Afects images and texts if they exist
def renameNumeric(folder):
    b = 0
    i = 0
    pathlist = getPathlist(folder)
    for path in pathlist:
        i = i + 1
        extension = os.path.splitext(path)[1]
        filename = os.path.splitext(path)[0]
        if extension in image_formats:
            #image rename
            newImgName = str(folder) + "/" + str(i) + extension
            os.rename(path, newImgName)
            b = b + 1
            expectedTxtName = str(filename) + ".txt"
            hasPair = os.path.isfile(expectedTxtName)
            if hasPair:
                #text rename
                newTxtName = str(folder) + "/"+ str(i) + ".txt"
                os.rename(expectedTxtName, newTxtName)
                b = b + 1
    print("Renamed " + str(b) + " Files.")

def checkPair(folder):
    b = 0
    pathlist = getPathlist(folder)
    for path in pathlist:
        extension = os.path.splitext(path)[1]
        if extension == ".txt":
            filename = os.path.splitext(path)[0]
            x = os.path.isfile(str(filename) + ".jpg") or os.path.isfile(str(filename) + ".png") or os.path.isfile(str(filename) + ".jpeg")
            if x is False:
                print("Removing " + str(path))
                os.remove(path)
                b = b + 1
        elif extension in image_formats:
            filename = os.path.splitext(path)[0]
            x = os.path.isfile(str(filename) + ".txt")
            if x is False:
                print("Removing " + str(path))
                os.remove(path)
                b = b + 1
    print("Removed " + str(b) + " Files.")

option = args.option
dataset = args.dataset

if option == "paircheck":
    checkPair(dataset)
elif option == "rename":
    renameNumeric(dataset)
elif option == "cleanup":
    cleanUpFormats(dataset)
else:
    print("no option")
