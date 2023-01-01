from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
from controlled_trainer import PyTorchTrainer
from threading import Thread
from queue import Queue
import omegaconf
import requests
import pickle
import os
import argparse

MOTHER = False

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--webui_port', help='Port to use for the WebUI', default=5080)
parser.add_argument('-m', '--mother', type=str, help='Run as Mother peer, include the conf file to use')
parser.add_argument('-t', '--tunnel', action='store_true', help='Enable Cloudflare Tunneling to WebUI')
parser.add_argument('-s', '--secret', type=str)
args = parser.parse_args()

if args.mother:
  MOTHER = True
  print("Warning, started as mother peer")


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
conf = omegaconf.OmegaConf.create()

if args.tunnel:
  import flask_cloudflared
  flask_cloudflared.run_with_cloudflared(app)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/config')
def config():
  return render_template('config.html')

@app.route('/saveconf', methods=['POST'])
def submit_conf():
  data = request.get_json()
  
  # Set the configuration values in the conf instance
  conf.username = data['username']
  conf.server = data['server'] #<--- this will now be a commoner peer
  conf.imageCount = data['imageCount']
  conf.batchSize = data['batchSize']
  conf.hftoken = data['hftoken']
  # conf.savesteps = data['savesteps']
  conf.gradckpt = data['gradckpt']
  conf.xformers = data['xformers']
  conf.eightbitadam = data['eightbitadam']
  conf.stats_ip = data['stats_ip']
  conf.enablestats = data['enablestats']
  conf.geoaprox = data['geoaprox']
  conf.bandwidth = data['bandwidth']
  conf.specs = data['specs']
  conf.trainermode = data['trainermode']
  conf.publicip = data['publicip']
  # conf.internal_udp = data['internal_udp']
  conf.internal_tcp = data['internal_tcp']
  # conf.external_udp = data['external_udp']
  conf.external_tcp = data['external_tcp']
  conf.internal_ie = data['internal_ie']
  conf.external_ie = data['external_ie']
  conf.enable_wandb = data['enable_wandb']
  conf.wandb_token = data['wandb_token']
  
  # hardcoded TODO: maybe one day we can add these to the webui
  conf.setdefault('intern', omegaconf.OmegaConf.create())
  conf.intern.workingdir = "workplace"
  conf.intern.tmpdataset = conf.intern.workingdir + "/dataset"
  conf.image_store_skip = False
  conf.image_store_extended = True
  conf.image_store_resize = True # <--- Slow as fuck
  conf.image_store_no_migration = True
  print(conf.server)
  # Get config from dataset server
  if args.secret:
    conf.sendloss = True
    conf.secretpass = args.secret
  if MOTHER:
    conf.everyone = omegaconf.OmegaConf.load(args.mother)
    conf.mother = True
  else:
    server_provided_config = requests.get('http://' + conf.server + '/globalconf')
    if server_provided_config.status_code == 200:
      server_provided_config = server_provided_config.json()
    else:
      return '', 502
    conf.setdefault('everyone', omegaconf.OmegaConf.create())
    conf.everyone.model = server_provided_config['model']
    conf.everyone.extended_chunks = int(server_provided_config['extended_chunks'])
    conf.everyone.clip_penultimate = bool(server_provided_config['clip_penultimate'])
    conf.everyone.fp16 = bool(server_provided_config['fp16'])
    conf.everyone.resolution = int(server_provided_config['resolution'])
    conf.everyone.seed = int(server_provided_config['seed'])
    conf.everyone.train_text_encoder = bool(server_provided_config['train_text_encoder'])
    conf.everyone.lr = float(server_provided_config['lr'])
    conf.everyone.ucg = float(server_provided_config['ucg'])
    conf.everyone.use_ema = bool(server_provided_config['use_ema'])
    conf.everyone.opt_betas_one = float(server_provided_config['opt_betas_one'])
    conf.everyone.opt_betas_two = float(server_provided_config['opt_betas_two'])
    conf.everyone.opt_epsilon = float(server_provided_config['opt_epsilon'])
    conf.everyone.opt_weight_decay = float(server_provided_config['opt_weight_decay'])
    conf.everyone.buckets_shuffle = bool(server_provided_config['buckets_shuffle'])
    conf.everyone.buckets_side_min = int(server_provided_config['buckets_side_min'])
    conf.everyone.buckets_side_max = int(server_provided_config['buckets_side_max'])

  print(conf)

  if os.path.isfile("DO_NOT_DELETE_config.pickle"):
    os.remove("DO_NOT_DELETE_config.pickle")
  
  with open("DO_NOT_DELETE_config.pickle", "wb") as pcklfile:
    pickle.dump(conf, pcklfile)

  return '', 200

@app.route('/getconf', methods=['GET'])
def get_conf():
  # Check if the conf instance has at least one key
  if len(conf.keys()) == 0:
    # Return an empty JSON object if the conf instance is empty
    return jsonify({})
  else:
    return jsonify(omegaconf.OmegaConf.to_container(conf))

@socketio.on('connect')
def handle_logs():
  def background_logs():
    while True:
      log = log_queue.get()
      print(log)
      socketio.emit("logs", log)
  socketio.start_background_task(target=background_logs)

@socketio.on('start')
def handle_start():
  print("started!")
  try:
    global trainer_process
    if trainer_process.is_alive():
      log_queue.put('TRAINER_MANAGER: Trainer is already running!.')
    else:
      log_queue.put('TRAINER_MANAGER: Initiating trainer thread. Start after previous termination.')
      trainer_process = Thread(target=PyTorchTrainer, args=(command_queue, log_queue))
      trainer_process.start()
      command_queue.put('start')
  except Exception as e:
    print("Got Exception", e)
    log_queue.put('TRAINER_MANAGER: Initiating trainer thread. First Start.')
    trainer_process = Thread(target=PyTorchTrainer, args=(command_queue, log_queue))
    trainer_process.start()
    # Send a 'start' command to the subprocess using the write end of the command pipe
    command_queue.put('start')

@socketio.on('stop')
def handle_stop():
  log_queue.put('TRAINER_MANAGER: Sending stop command to trainer.')
  command_queue.put('stop')

@socketio.on('save')
def handle_save():
  log_queue.put('TRAINER_MANAGER: Sending save command to trainer.')
  command_queue.put('save')

if __name__ == '__main__':
  log_queue = Queue()
  command_queue = Queue()
  socketio.run(app, host="0.0.0.0", port=args.webui_port)
  