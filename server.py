from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
from controlled_trainer import PyTorchTrainer
import omegaconf
import requests
from threading import Thread
from queue import Queue
import time
import pickle
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
conf = omegaconf.OmegaConf.create()

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
  conf.server = data['server']
  conf.imageCount = data['imageCount']
  conf.batchSize = data['batchSize']
  conf.hftoken = data['hftoken']
  conf.savesteps = data['savesteps']
  conf.gradckpt = data['gradckpt']
  conf.xformers = data['xformers']
  conf.eightbitadam = data['eightbitadam']
  conf.enablestats = data['enablestats']
  conf.geoaprox = data['geoaprox']
  conf.bandwidth = data['bandwidth']
  conf.specs = data['specs']
  conf.trainermode = data['trainermode']
  conf.publicip = data['publicip']
  conf.internal_udp = data['internal_udp']
  conf.internal_tcp = data['internal_tcp']
  conf.external_udp = data['external_udp']
  conf.external_tcp = data['external_tcp']
  conf.enable_wandb = data['enable_wandb']
  conf.wandb_token = data['wandb_token']
  conf.enable_inference = data['enable_inference']
  
  # hardcoded TODO: maybe one day we can add these to the webui
  conf.setdefault('intern', omegaconf.OmegaConf.create())
  conf.intern.workingdir = "workplace"
  conf.intern.tmpdataset = conf.intern.workingdir + "/dataset"
  conf.image_store_skip = True
  conf.image_store_extended = False
  conf.image_store_resize = False
  conf.image_store_no_migration = True
  conf.image_inference_scheduler = 'DDIMScheduler'
  print(conf.server)
  # Get config from dataset server
  server_provided_config = requests.get('http://' + conf.server + '/v1/get/config')
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
  conf.everyone.lr_scheduler = server_provided_config['lr_scheduler']
  conf.everyone.opt_betas_one = float(server_provided_config['opt_betas_one'])
  conf.everyone.opt_betas_two = float(server_provided_config['opt_betas_two'])
  conf.everyone.opt_epsilon = float(server_provided_config['opt_epsilon'])
  conf.everyone.opt_weight_decay = float(server_provided_config['opt_weight_decay'])
  conf.everyone.buckets_shuffle = bool(server_provided_config['buckets_shuffle'])
  conf.everyone.buckets_side_min = int(server_provided_config['buckets_side_min'])
  conf.everyone.buckets_side_max = int(server_provided_config['buckets_side_max'])
  conf.everyone.lr_scheduler_warmup = float(server_provided_config['lr_scheduler_warmup'])

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
  # Send a 'start' command to the subprocess using the write end of the command pipe
  command_queue.put('start')

@socketio.on('stop')
def handle_stop():
  command_queue.put('stop')

if __name__ == '__main__':
  log_queue = Queue()
  command_queue = Queue()
  trainer_process = Thread(target=PyTorchTrainer, args=(command_queue, log_queue))
  trainer_process.start()
  socketio.run(app)