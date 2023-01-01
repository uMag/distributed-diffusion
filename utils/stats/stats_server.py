import time
from threading import Timer
import hivemind
from flask import Flask, request, jsonify
from ip2geotools.databases.noncommercial import DbIpCity
import requests
from flask_cors import CORS
from hivemind.optim.progress_tracker import ProgressTracker

DHT_ADRESS = "/ip4/204.15.42.199/tcp/3000/p2p/12D3KooWHAf7qweFdxHP7TidWGxT2LYrW5owFDBCrsFjwEzGBshm"
RUN_ID = "testrun"
PASSWORD = "WHHgVhHkYmK59jFzP4E4EUSR"

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

@app.route('/api/v1/private/postloss')
def postloss():
    # Get the JSON payload
    json = request.get_json()

    # Get the provided password
    username = request.authorization.username
    password = request.authorization.password

    # Verify thepassword
    if password != PASSWORD:
        return 'Invalid username or password', 401

    time = json['time']
    loss = json['loss']

    thisuserentry = lossdata[username]
    thisuserentry[time] = loss

    return 'Success', 200


@app.route('/api/v1/get/announcelocation')
def index():
    # Determine the client's geo location using the geoip2 library
    client_ip = request.remote_addr

    geoinfo = DbIpCity.get(client_ip, api_key='free')
    client_location= {'latitude': geoinfo.latitude, 'longitude': geoinfo.longitude}

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

@app.route('/api/v1/get/dhtstats')
def dhtstats():
    return jsonify(tracker.global_progress)

@app.route('/api/v1/get/lossreports')
def lossreports():
    return jsonify(lossdata)

if __name__ == '__main__':
    context = ('/etc/letsencrypt/live/stats.swarm.sail.pe/fullchain.pem', '/etc/letsencrypt/live/stats.swarm.sail.pe/privkey.pem')
    app.run(host="0.0.0.0", port=443, ssl_context=context)
