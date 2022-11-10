from flask import Flask, jsonify, request, send_file
from pathlib import Path
from zipfile import ZipFile
import os
import argparse
import time
from io import BytesIO

parser = argparse.ArgumentParser(description='Dataset server')
parser.add_argument('--dataset', type=str, default=None, required=True, help='Path to dataset')
parser.add_argument('--new', type=bool, default=False, help='re-scan the dataset folder')
parser.add_argument('--load', type=str, default=None, help='path to the JSON DB snapshot')
args = parser.parse_args()

#gt/GetTime
def gt():
    return(str(time.time_ns()))

#this should not be used in the final version + might have directory traversal
def solvePath(filename):
    path = args.dataset + "/" + filename
    return(path)

def dictCreator(input):
    dataPath = Path(input)
    dataDict = os.listdir(dataPath)
    #sort everything
    sortedDict = {}
    entryId = 0
    for entry in dataDict:
        entryExt = os.path.splitext(entry)[1]
        entryFilename = os.path.splitext(entry)[0]
        #ignore txt files for now
        if entryExt == ".txt":
            continue
        expectedTxtName = str(entryFilename) + ".txt"
        #TODO: Change input to a proper Path object
        expectedTxtLocation = os.path.join(input + "/" + expectedTxtName)
        txtPairExists = os.path.isfile(expectedTxtLocation)
        #only add to dict the entries that have valid pairs (txt & img)
        if txtPairExists:
            entryId = entryId + 1
            tmpDict = {
                'imagefile': entry,
                'textfile': expectedTxtName,
                'assigned': False,
                'epochs': 0
            }
            sortedDict[entryId] = tmpDict
    print("Registered " + str(entryId) + " entries.")
    return sortedDict

#directory to the dataset
dataDir = args.dataset
filesDict = dictCreator(dataDir)

app = Flask(__name__)

filesDict[16]["epochs"] = 1

#getTasks: return entries(objects) in dataset that need training
#it should return a list, that contains entries with low train count.
@app.route("/v1/get/tasks/<string:wantedTasks>")
#reverse=True to get descending
def getTasks(wantedTasks):
    intWantedTasks = int(wantedTasks)
    listToReturn = []
    sortedDict = sorted(filesDict.items(), key=lambda x_y: x_y[1]['epochs'])
    for i in range(intWantedTasks):
        listToReturn.append(sortedDict[i])
    return jsonify(listToReturn)
    
@app.route('/v1/get/files', methods=['POST'])
def getFiles():
    print("Got request for files!")
    content = request.get_json(force=True)
    memory_file = BytesIO()
    with ZipFile(memory_file, 'w') as zf:
        for i in range(len(content)):
            imgFile = content[i][1]['imagefile']
            txtFile = content[i][1]['textfile']
            zf.write(solvePath(imgFile))
            zf.write(solvePath(txtFile))
        zf.close()
    memory_file.seek(0)
    print("About to be sent!")
    return send_file(memory_file, as_attachment=True, download_name="file.zip")

if __name__ == '__main__':
    app.run(debug=True, port=8080)