import hashlib
from flask import Flask, request
import sqlite3
import datetime
from ip2geotools.databases.noncommercial import DbIpCity

PATH_TO_USER_DB = "database.db"
PATH_TO_DATA_DB = "data.db"

app = Flask(__name__)

@app.route('/api/v1/telemetry', methods=['POST'])
def save_data():
    # Extract the username and password from the request
    jsonpop = request.get_json()
    username = jsonpop['username']
    password = jsonpop['password']

    # Hash the password using the SHA256 algorithm
    hashed_password = hashlib.sha256(password.encode('utf-8')).hexdigest()

    # Connect to the database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # Verify that the username and password match
    cursor.execute('''SELECT * FROM users WHERE username=? AND password=?''', (username, hashed_password))
    result = cursor.fetchone()
    if result is None:
        return 'Invalid username or password', 401

    tmp_data = jsonpop['data']
    client_ip = request.remote_addr
    geoinfo = DbIpCity.get(client_ip, api_key='free')

    tmp_data['location'] = {'latitude': geoinfo.latitude, 'longitude': geoinfo.longitude}

    data = str(tmp_data)

    # Connect to the data database
    data_conn = sqlite3.connect('data.db')
    data_cursor = data_conn.cursor()
    
    # Save the data to the database
    timestamp = datetime.datetime.now().isoformat()
    data_cursor.execute('''INSERT INTO data(username, timestamp, data) VALUES(?, ?, ?)''', (username, timestamp, data))
    data_conn.commit()

    return 'OK', 200



app.run()
