import time
from threading import Timer
import hivemind
from flask import Flask, request, jsonify
import geoip2.database
import requests
from flask_cors import CORS
from hivemind.optim.progress_tracker import ProgressTracker
from datetime import datetime, timedelta

DHT_ADRESS = ""
RUN_ID = "testrun"
PASSWORD = ""
DISABLE_DHT = False

if not DISABLE_DHT:
    dht = hivemind.DHT(
        initial_peers=[DHT_ADRESS],
        start=True,
        daemon=True,
        client_mode=True,
    )

    print(dht.num_workers)

    tracker = ProgressTracker(
        dht=dht,
        prefix=RUN_ID,
        target_batch_size=75000, #must be the same as peers
        start=True,
        daemon=True,
        min_refresh_period=0.5,
        max_refresh_period=1.0,
        default_refresh_period=1
    )

app = Flask(__name__)
CORS(app)

ip = requests.get("https://api.ipify.org/", timeout=5).text
print(ip)

# Create a list to store the client's geo location and timestamp
client_locations = []

# Create a timer to remove old entries from the list
def cleanup_client_locations():
    now = time.time()
    for client_location in client_locations:
        if now - client_location['timestamp'] > 7 * 60:
            client_locations.remove(client_location)
    # Schedule the next cleanup
    Timer(60, cleanup_client_locations).start()

# Start the cleanup timer
Timer(60, cleanup_client_locations).start()


lossdata = {}

def get_user_timeout(username):
    # Set the default timeout to 30 seconds
    timeout = 30
    
    # Look up the user's timeout, if it exists
    if username in lossdata:
        user_data = lossdata[username]
        if 'timeout' in user_data:
            timeout = user_data['timeout']
    
    return timeout

@app.route('/api/v1/private/postloss')
def postloss():
    # Get the JSON payload
    json = request.get_json()

    # Get the provided password
    username = request.authorization.username
    password = request.authorization.password

    # Verify the password
    if password != PASSWORD:
        return 'Invalid username or password', 401

    time = json['time']
    loss = json['loss']
    
    # Check if the user has made a request within the timeout period
    timeout = get_user_timeout(username)
    if username in lossdata:
        user_data = lossdata[username]
        last_request_time = user_data['time']
        if (datetime.now() - last_request_time) < timedelta(seconds=timeout):
            return 'Timeout', 200

    # Update the loss data for the user
    if username not in lossdata:
        lossdata[username] = {}
    lossdata[username][time] = loss
    lossdata[username]['time'] = datetime.now()
    
    return 'Success', 200


@app.route('/api/v1/get/announcelocation')
def index():
    # Determine the client's geo location using the geoip2 library
    client_ip = request.remote_addr

    maxmind_db = '/home/ubuntu/distributed-diffusion/utils/stats/db.mmdb'
    reader = geoip2.database.Reader(maxmind_db)
    response = reader.city(client_ip)
    client_location = {
        'latitude': response.location.latitude,
        'longitude': response.location.longitude
    }

    # Check if there is already an entry for the client's IP in the list
    for i, entry in enumerate(client_locations):
        if entry['ip'] == client_ip:
            # Update the entry with the new location and timestamp
            client_locations[i] = {
                'ip': client_ip,
                'location': client_location,
                'timestamp': time.time()
            }
            break
    else:
        # Add a new entry to the list
        client_locations.append({
            'ip': client_ip,
            'location': client_location,
            'timestamp': time.time()
        })

    return 'Client location stored in list!'

@app.route('/api/v1/get/peercoords')
def get_client_locations():
    # Format the client locations in the desired format
    full_list = []
    for entry in client_locations:
        latit = entry['location']['latitude']
        longi = entry['location']['longitude']
        to_send = {
            "lat": latit,
            "lng": longi,
            "size": 0.1,
            "color": "white"
        }
        full_list.append(to_send)

    return jsonify(full_list)

if not DISABLE_DHT:
    @app.route('/api/v1/get/dhtstats')
    def dhtstats():
        return jsonify(tracker.global_progress)

@app.route('/api/v1/get/lossreports')
def lossreports():
    return jsonify(lossdata)

if __name__ == '__main__':
    context = ('/etc/letsencrypt/live/stats.swarm.sail.pe/fullchain.pem', '/etc/letsencrypt/live/stats.swarm.sail.pe/privkey.pem')
    app.run(host="0.0.0.0", port=443, ssl_context=context)
