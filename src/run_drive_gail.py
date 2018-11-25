import argparse
import base64
from datetime import datetime
import os
import shutil
import csv
import cv2

import numpy as np
import tensorflow as tf
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
from keras.models import Model
#execute command
from subprocess import Popen
#mouse control
import pyautogui
#display now time
from datetime import datetime
#for sleep
from time import sleep
#for kill process
import psutil

#for gail
from network_models.policy_net import Policy_net
from network_models.discriminator import Discriminator
from algo.ppo import PPOTrain
import algo.cocob_optimizer as cocob

model_dir = './drive/model.h5'
simul_path = "/home/deep/source/safegail/simul1/Default Linux desktop Universal.x86_64"
start_check_speed = False
MIN_SPEED_LIMIT = 30
MIN_SEPPD_CNT = 1
MAX_EPISODE_LENGTH = 1000

sio = socketio.Server()
app = Flask(__name__)
model = None
FEN = None
prev_image_array = None
controller = None
simul = None
simulator_pid = None

speed_counter = 0 #for termination
episode_length = 0
iteration = 0
max_iteration = 0

Policy = None
Old_Policy = None
PPO = None
Disc = None
observation_space = (100,)
action_space = 1

# trajectories
expert_observations = None
expert_actions = None
agent_observations = None
agent_actions = None
v_preds = None

# global session
sess = None
writer = None


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral

# extract agent's trajectory
@sio.on('telemetry')
def telemetry(sid, data):
    global writer
    global sess
    global Policy
    global agent_observations
    global agent_actions
    global speed_counter
    global episode_length
    global iteration
    global start_check_speed
    global MAX_EPISODE_LENGTH

    if data:
        # The current speed of the car
        speed = data["speed"]
        #add the moved distance by count
        episode_length += 1

        #check speed
        if start_check_speed==False and float(speed)>MIN_SPEED_LIMIT:
            start_check_speed = True
            print(">>>>> START TO CHECK SPEED LIMIT")
        elif start_check_speed == False and float(speed) < MIN_SPEED_LIMIT and episode_length>=MAX_EPISODE_LENGTH:
            try:
                sio.disconnect(sid)
            except:
                print('[{}] disconneting'.format(sid))
            kill_simulator()
            print('drive finished')
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=episode_length)]),
                               iteration)
            run_training()
        elif start_check_speed==True and float(speed)<MIN_SPEED_LIMIT:
            speed_counter += 1
            print('speed counter: {}'.format(speed_counter))
            if (speed_counter>=MIN_SEPPD_CNT):
                try:
                    sio.disconnect(sid)
                except:
                    print('[{}] disconneting'.format(sid))
                kill_simulator()
                print('drive finished')
                writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=episode_length)]), iteration)
                run_training()
        elif episode_length>=MAX_EPISODE_LENGTH: # reach to maximum episode length(max reward)
            try:
                MAX_EPISODE_LENGTH += 500  # increse 500 steps
                #kill_simulator()
                #print('###### MAX EPISODE ######')
                #writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=episode_length)]), iteration)
                # model save
                saver = tf.train.Saver()
                saver.save(sess, args.savedir + '/model.ckpt')
                #print('###### Model Saved ######')
                #finish_train()
                #print('###### DONE ######')

                #run_training()
            except:
                print('')
        else:
            speed_counter = 0


        # The current image from the center camera of the car
        imgString = data["image"]
        image_array = image_to_feature(BytesIO(base64.b64decode(imgString))) #capture image to feature

        # get trajectory
        obs = np.stack(image_array).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
        steering_angle, v_pred = Policy.act(sess, obs=obs, stochastic=True)
        steering_angle = float(steering_angle)
        v_preds.append(v_pred)
        agent_observations.append(obs)
        agent_actions.append(steering_angle)

        #throttle = controller.update(float(speed))
        throttle = 31

        print("[{}] [cnt:{}] steering_angle:{} , throttle:{}, speed:{}".format(str(datetime.now()), episode_length, steering_angle, throttle, speed))
        send_control(steering_angle, throttle)

    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    global simul_sid
    print("connect ", sid)
    send_control(0, 0)

@sio.on('disconnect')
def test_disconnect():
    print('disconnected')


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)

# Run Simulator
def run_simulator():
    global simulator_pid

    print(">>> run simulator")
    simul = Popen([simul_path])
    simulator_pid = simul.pid
    sleep(1)
    pyautogui.click(820, 830) #Click OK
    sleep(3)
    pyautogui.click(330, 410) #Click Autonomousmode

# Kill Simulator
def kill_simulator():
    print(">>> kill simulator")
    process = psutil.Process(simulator_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()


# Initialize objects for training
def init_train(args):
    global writer
    global sess
    global Policy
    global Old_Policy
    global PPO
    global Disc
    global max_iteration
    global iteration
    global observation_space
    global action_space
    global expert_observations
    global expert_actions

    print("###### INITIALIZING ######")
    max_iteration = args.iteration
    iteration = 0
    # PPO
    Policy = Policy_net('policy', observation_space)
    Old_Policy = Policy_net('old_policy', observation_space)
    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma)

    # GAIL
    Disc = Discriminator(observation_space)

    # read trajectories
    expert_observations = []
    expert_actions = []
    #for data balancing
    cnt_zero_trj = 0
    ZERO_LIMIT = 300 #limit zero trajectory size
    cnt_left_trj = 0
    LEFT_LIMIT = 776
    cnt_right_trj = 0
    #profiles = []  # center_img, left_img, right_img, wheel_angle, acc, break, speed
    for _dir in os.listdir(args.trjdir):
        raw_filename = os.path.join(os.getcwd(), args.trjdir, _dir, 'driving_log.csv')
        with open(raw_filename) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:  # each row is a list
                if float(row[3])==0.0 : #check zero(go straght)
                    if cnt_zero_trj<=ZERO_LIMIT:
                        cnt_zero_trj += 1
                        expert_observations.append(np.squeeze(image_to_feature(row[0])))
                        expert_actions.append(round(float(row[3]), 2))
                elif float(row[3])<0.0 : #check minus(left turn)
                    if cnt_left_trj<=LEFT_LIMIT:
                        cnt_left_trj += 1
                        expert_observations.append(np.squeeze(image_to_feature(row[0])))
                        expert_actions.append(round(float(row[3]), 2))
                else: #plus(right turn)
                    cnt_right_trj += 1
                    expert_observations.append(np.squeeze(image_to_feature(row[0])))
                    expert_actions.append(round(float(row[3]),2))

    print("###### READ TRAJECTORY: {} ######".format(len(expert_actions)))
    print("center:{}, left:{}, right:{}".format(cnt_zero_trj,cnt_left_trj,cnt_right_trj))

    # import matplotlib.pyplot as plt
    # plt.hist(expert_actions, bins=20)
    # plt.ylabel('Probability');
    # plt.xlabel('Weight')
    # plt.show()
    # return

    # initialize Tensorflow
    sess = tf.Session()
    writer = tf.summary.FileWriter(args.logdir, sess.graph)
    sess.run(tf.global_variables_initializer())

    if os.path.isfile(args.savedir + '/model.ckpt.meta')==True:
        print("###### LOAD SAVED MODEL !!!!! ######")
        saver = tf.train.Saver()
        saver.restore(sess, args.savedir + '/model.ckpt')

    extract_agent_trajectory()

def finish_train():
    print("###### FINISH TRAIN ######")
    writer.close()
    sess.close()

def extract_agent_trajectory():
    global episode_length
    global speed_counter
    global agent_observations
    global agent_actions
    global v_preds
    global start_check_speed

    print("###### Extract Agent's Trajectory ######")

    # initialize
    episode_length = 0
    speed_counter = 0
    agent_observations = []
    agent_actions = []
    v_preds = []
    start_check_speed = False

    run_simulator()


# run training
def run_training():
    global writer
    global sess
    global Policy
    global Old_Policy
    global PPO
    global Disc
    global max_iteration
    global iteration
    global observation_space
    global action_space
    global expert_observations
    global expert_actions
    global agent_observations
    global agent_actions
    global v_preds

    print("###### START TRAINING ######")
    # convert list to numpy array for feeding tf.placeholder
    #observations = agent_observations
    #actions = agent_actions
    exp_observations = np.reshape(expert_observations, newshape=[-1] + list(observation_space))
    exp_actions = np.reshape(expert_actions, newshape=[-1, 1])
    observations = np.reshape(agent_observations, newshape=[-1] + list(observation_space))
    actions = np.reshape(agent_actions, newshape=[-1, 1]) #np.array(agent_actions).astype(dtype=np.float32)

    ####FOR GAIL### to make d_rewards for PPO
    # train discriminator
    print('###### DISCRIMINATOR TRAIN ######')
    for i in range(2):
        Disc.train(sess,
            expert_s=exp_observations,
            expert_a=exp_actions,
            agent_s=observations,
            agent_a=actions)

    # output of this discriminator is reward
    d_rewards = Disc.get_rewards(sess, agent_s=observations, agent_a=actions)
    d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32)
    ###############

    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
    gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)
    gaes = np.array(gaes).astype(dtype=np.float32)
    v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

    # train policy
    inp = [observations, actions, gaes, d_rewards, v_preds_next]
    PPO.assign_policy_parameters(sess)
    print('###### PPO TRAIN ######')
    for epoch in range(6):
        sample_indices = np.random.randint(low=0, high=observations.shape[0],
                                           size=32)  # indices are in [low, high)
        sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
        PPO.train(sess,
                  obs=sampled_inp[0],
                  actions=sampled_inp[1],
                  gaes=sampled_inp[2],
                  rewards=sampled_inp[3],
                  v_preds_next=sampled_inp[4])
    summary = PPO.get_summary(sess,
                              obs=inp[0],
                              actions=inp[1],
                              gaes=inp[2],
                              rewards=inp[3],
                              v_preds_next=inp[4])

    writer.add_summary(summary, iteration)

    if (iteration >= max_iteration):
        finish_train()
        print('###### DONE ######')
    else:
        iteration += 1
        print("###### END TRAINING ######")
        extract_agent_trajectory()


#image file to array
def image_to_feature(image_file):
    try:
        image = Image.open(image_file)
        image_array = np.asarray(image)
        image_array = image_array[60:-25, :, :]
        feature = FEN.predict(image_array[None, ...], batch_size=1) #(1, 100)
        return feature
    except:
        print("NO IMAGE FILE!!!!!")
        return []

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trjdir', help='trajectory directory', default='trajectory')
    parser.add_argument('--logdir', help='log directory', default='log/train/gail')
    parser.add_argument('--savedir', help='save directory', default='trained_models/gail')
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('--iteration', default=int(1e4))
    return parser.parse_args()

if __name__ == '__main__':
    args = argparser()

    controller = SimplePIController(0.1, 0.002)
    set_speed = 30
    controller.set_desired(set_speed)

    model = load_model(model_dir)
    # print (model.summary())

    #extract pretraining feature
    FEN = Model(
        input=model.input,
        output=model.get_layer('dense_1').output  # dense 100의 output dense_1:100, 2:50, 3:10
    )

    # Initialize objects for training
    init_train(args)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    server = eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

    # start training
    # run_training()

