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

#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)
NNN = 0



#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    global NNN

    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image.save("img"+str(NNN)+".png","PNG")
            image = np.asarray(image)  # from PIL image to numpy array  (66,200,3)
            image = utils.preprocess(image)  # apply the preprocessing
            image = np.array([image])  # the model expects 4D array

            NNN += 1
            print('{} {}'.format(image.shape, NNN))
            send_control(0, 20) #max speed 30
        except Exception as e:
            print(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)

#connect simulator
@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    args = parser.parse_args()


    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)