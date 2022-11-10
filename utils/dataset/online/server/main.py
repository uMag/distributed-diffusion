from flask import Flask, jsonify, request, send_file, Response
from pathlib import Path
from zipfile import ZipFile
import os
import argparse
import time
from io import BytesIO
from datetime import datetime

parser = argparse.ArgumentParser(description='Dataset server')
parser.add_argument('--dataset', type=str, default=None, required=True, help='Path to dataset')
parser.add_argument('--new', type=bool, default=False, help='re-scan the dataset folder')
parser.add_argument('--load', type=str, default=None, help='path to the JSON DB snapshot')
parser.add_argument('--name', type=str, default=None, required=True, help='Server name')
parser.add_argument('--description', type=str, default=None, required=True, help='Server description')
args = parser.parse_args()

#info
version = "0.1a"
execDate = str(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
print("Server Version: " + version)
print("Current Time: " + execDate)

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
            tmpDict = {
                'imagefile': entry,
                'textfile': expectedTxtName,
                'assigned': False,
                'assignedTimeLeft': 'none',
                'epochs': 0,
                'entryId': entryId
            }
            sortedDict[entryId] = tmpDict
            entryId = entryId + 1
    print("Registered " + str(entryId) + " entries.")
    return sortedDict, entryId

#directory to the dataset
dataDir = args.dataset
filesDict, numberFiles = dictCreator(dataDir)

app = Flask(__name__)

#current version and info
@app.route("/info")
def getInfo():
    info = {
        "ServerName": args.name,
        "ServerDescription": args.description,
        "ServerVersion": version,
        "FilesBeingServed": numberFiles,
        "ExecutedAt": execDate
    }
    return jsonify(info)

#getTasksFull
@app.route("/v1/get/tasks/full")
def getTasksFull():
    return jsonify(filesDict)

#getTasks: return entries(objects) in dataset that need training
#it should return a list, that contains entries with low train count.
@app.route("/v1/get/tasks/<string:wantedTasks>")
#reverse=True to get descending
def getTasks(wantedTasks):
    #minus one cuz computer number system != human
    intWantedTasks = int(wantedTasks) - 1
    listToReturn = []
    sortedDict = sorted(filesDict.items(), key=lambda x_y: x_y[1]['epochs'])
    obtainedTasks = 0
    x = 0
    while obtainedTasks < intWantedTasks:
        for i in sortedDict:
            if obtainedTasks > intWantedTasks:
                break
            if sortedDict[x][1]['assigned']:
                x = x + 1
                break
            listToReturn.append(sortedDict[x])
            entryId = sortedDict[x][0]
            filesDict[entryId]['assigned'] = True
            obtainedTasks = obtainedTasks + 1
            x = x + 1
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
            zf.write(solvePath(imgFile), imgFile)
            zf.write(solvePath(txtFile), txtFile)
        zf.close()
    memory_file.seek(0)
    print("About to be sent!")
    return send_file(memory_file, as_attachment=True, download_name="file.zip", mimetype="application/zip")

#for some reason the dict turned into a list out of nowhere idk what is going on here
@app.route("/v1/post/epochcount", methods=['POST'])
def epochCount():
    print("Someone is reporting an epoch completition.")
    try:
        content = request.get_json(force=True)
    except Exception:
        return(Response("Failed decoding JSON", status=400))
    for i in range(len(content)):
        entryId = content[i][1]['entryId']
        currentNumOfEpoch = filesDict[int(entryId)]['epochs']
        newNumOfEpoch = currentNumOfEpoch + 1
        filesDict[entryId]['epochs'] = newNumOfEpoch
    print("Saved Successfully")
    return(Response(status=200))
    

if __name__ == '__main__':
    app.run(debug=True, port=8080)