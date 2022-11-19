import hivemind
import requests
import zipfile
import shutil
import logging
import os
import time
import json

#logging functions
ld = logging.debug
li = logging.info
lw = logging.warning
le = logging.error

class DeposHandler():
    def __init__(self, config):
        super().__init__()
        dc = config.depos
        hostname = dc.server.hostname
        port = str(dc.server.port)
        li('hostname:' + hostname + " port: " + port)

        imgcap = dc.localcapacity
        workdir = dc.workingdirectory
        li('imgcap:' + str(imgcap) + " workdir: " + workdir)

        tmpdir = os.path.join(workdir, 'tmp')
        if os.path.exists(tmpdir):
            lw("Warning, " + str(tmpdir) + "will be cleared in 10 seconds.")
            time.sleep(10)
            shutil.rmtree(tmpdir)
        os.makedirs(tmpdir, exist_ok=True)
        
        httpserver = 'http://' + hostname + ":" + port + "/"
        r_serverinfo = requests.get(httpserver + '/info')
        if r_serverinfo.status_code == 200:
            print("Connection Success")
            data = json.loads(r_serverinfo.text)
            print("Server: " + data['ServerName'])
            li(data['ServerDescription'])
            li("Server Version: " + data['ServerVersion'])
            li("Currently serving " + str(data['FilesBeingServed']) + " Files")
            li("Age: " + data['ExecutedAt'])
        else:
            raise ConnectionError('Failed to get server info')

        self.httpserver = httpserver
        self.imgcap = imgcap
        self.tmpdir = tmpdir

    def gather(self):
        print("Gathering tasks...")
        urlGetTasks = self.httpserver + 'v1/get/tasks/' + str(self.imgcap)
        urlGetFiles = self.httpserver + 'v1/get/files'
        ld('urlGetTasks: ' + str(urlGetTasks))
        ld('urlGetFiles: ' + str(urlGetFiles))

        r_gettasks = requests.get(urlGetTasks)
        print("Downloading files...")
        p_downfiles = requests.post(urlGetFiles, json=r_gettasks.json())
        #TODO: don't forget about the memory file
        tmpZipName = os.path.join(self.tmpdir, 'tmp.zip')
        datasetPath = os.path.join(self.tmpdir, 'dataset')
        self.datasetdir = datasetPath
        open(tmpZipName, 'wb').write(p_downfiles.content)
        os.makedirs(self.datasetdir, exist_ok=True)
        print("Unzipping...")
        with zipfile.ZipFile(tmpZipName, 'r') as z:
            z.extractall(datasetPath)
        os.remove(tmpZipName)
        return(r_gettasks.json(), datasetPath)

    def report(self, recipt):
        print("Reporting tasks...")
        urlReport = self.httpserver + 'v1/post/epochcount'
        p_reportepoch = requests.post(urlReport, json=recipt)
        if p_reportepoch.status_code != 200:
            raise ConnectionError('Failed to report epoch, server failure')
        if os.path.exists(self.datasetdir):
            print("Clearing dataset...")
            shutil.rmtree(self.datasetdir)

