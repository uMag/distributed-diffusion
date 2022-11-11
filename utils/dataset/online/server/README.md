# Dataset Server
*note: this is not meant for final use as its very vague and directory traversal is an easy thing here

This dataset server will let you share chunks of the dataset over the internet to other peers, sorted by the amount of epochs they have been trained on.

## Flags
```
--dataset : Dataset Folder Path
--name : Name of the Server
--description : Description of the server
--tasktimeout : Time to wait for epoch report
```

## Requirements
```
flask
keyboard
zipfile
```

## Available URLs

### `/info`
returns information about the dataset server such as:
- Server Name
- Server Description
- Server Version
- Number of files being served
- Online Since date

### `/v1/get/tasks/full`
returns a JSON including info about all the entries in the dataset

### `/v1/get/tasks/<integer>`
Returns < integer > number of tasks for the node to do, and marks said tasks as "Assigned" along with a "AssignedExpirationDate". If the node does not answer before "AssignedExpirationDate", the entries will be set to "Unassigned" again.

### `/v1/get/files`
Requires JSON in POST

Returns a zip file containing all the files requested (both IMGs and TXTs)

### `/v1/post/epochcount`
Requires JSON in POST

Marks all the entries in the recieved JSON as one more epoch to the epoch count.