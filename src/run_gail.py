#parsing command line arguments
import argparse
#decoding camera images
import base64
#for frametimestamp saving
from datetime import datetime
#reading and writing files
import os
#high level file operations
import shutil
#matrix math
import numpy as np
#real-time server
import socketio
#web server gateway interface
import eventlet.wsgi
#image manipulation
from PIL import Image
#web framework
from flask import Flask
#input output
from io import BytesIO
#helper class
import utils
#execute command
from subprocess import Popen
#mouse control
import pyautogui
#for sleep
from time import sleep

import numpy as np
import tensorflow as tf
from network_models.policy_net import Policy_net
from network_models.discriminator import Discriminator
from algo.ppo import PPOTrain

np.random.seed(1024)

#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)
# simulator
simul = None
# parameters
args = None
observation_space_count = 0
action_space_count = 0
feat_dim = []
img_dim = []
aux_dim = 0
encode_dim = 0
config = None
sess = None
#PPO
Policy = None
Old_Policy = None
PPO = None
D = None

#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    global simul
    #global variables
    # global args
    # global Policy
    # global Old_Policy
    # global PPO
    # global D
    # global saver

    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        #print(feat_dim.shape)

        # for test
        send_control(0, 20)  # max speed 30
        sleep(1)
        simul.terminate()

        #try:
            #image.save("img"+str(1)+".png","PNG")
            #image = np.asarray(image)  # from PIL image to numpy array  (66,200,3)
            #image = utils.preprocess(image)  # apply the preprocessing
            #image = np.array([image])  # the model expects 4D array




        #except Exception as e:
            #print(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)

#connect simulator
@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

#disconnect simulator
@sio.on('disconnect')
def disconnect(sid):
    print('disconnect ', sid)

#action
def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)
    print('send: {} {}'.format(steering_angle.__str__(),throttle.__str__()))

#initialize
def init_objects():
    env = ''
    observation_space_count = 0
    action_space_count = 0
    feat_dim = [7, 13, 1024]
    img_dim = [50, 50, 3]
    aux_dim = 10
    encode_dim = 2
    action_dim = 3

    # Policy = Policy_net('policy', observation_space_count=observation_space_count, action_space_count=action_space_count)
    # Old_Policy = Policy_net('old_policy', observation_space_count=observation_space_count, action_space_count=action_space_count)
    # PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma)
    # D = Discriminator(observation_space_count, action_space_count)
    #
    # saver = tf.train.Saver()
    #
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

#Run Simulator
def run_simulator():
    global simul
    #pyautogui.moveTo(820, 830)  # OK Position
    simul = Popen(["/home/deep/source/safegail/simul1/Default Linux desktop Universal.x86_64"])
    #subprocess.call('/home/deep/source/safegail/simul1/Default\ Linux\ desktop\ Universal.x86_64', shell=True)
    sleep(1)
    pyautogui.click(820, 830) #Click OK
    sleep(4)
    pyautogui.click(330, 410) #Click Autonomousmode

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/train/ppo')
    parser.add_argument('--savedir', help='save directory', default='trained_models/ppo')
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--iteration', default=int(1e4), type=int)
    return parser.parse_args()

if __name__ == '__main__':

    args = argparser()

    init_objects()

    # Run Simulator
    run_simulator()
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

