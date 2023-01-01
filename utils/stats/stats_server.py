import time
from threading import Timer

from flask import Flask, request, jsonify
from ip2geotools.databases.noncommercial import DbIpCity
import requests
from flask_cors import CORS

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

@app.route('/ping_geo')
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

@app.route('/client_locations')
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

if __name__ == '__main__':
    context = ('fullchain.pem', 'privkey.pem')
    app.run(host="0.0.0.0", port=8010, ssl_context=context)
