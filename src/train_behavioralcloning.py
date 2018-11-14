import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import csv
from network_models.resnet50 import RESNET50
import algo.cocob_optimizer as cocob

np.random.seed(1024)
BATCH_SIZE = 64
ITERATIONS = 10000

def main(args):

    # read trajectories
    profiles = [] #center_img, left_img, right_img, wheel_angle, acc, break, speed
    for _dir in os.listdir(args.trjdir):
        raw_filename = os.path.join(os.getcwd(), args.trjdir, _dir, 'driving_log.csv')
        with open(raw_filename) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:  # each row is a list
                profiles.append(row)
    train_profiles, test_profiles = train_test_split(profiles, shuffle=True)
    print("train data:{}, test data:{}".format(len(train_profiles), len(test_profiles)))
    train_image_arr = profile_to_centerimage_array(train_profiles)
    train_throttle_arr = profile_to_throttle(train_profiles)
    train_steering_arr = profile_to_steering(train_profiles)

    test_image_arr = profile_to_centerimage_array(test_profiles)
    test_throttle_arr = profile_to_throttle(test_profiles)
    test_steering_arr = profile_to_steering(test_profiles)

    # input image dimensions
    img_rows, img_cols, img_channel = 85, 320, 3
    input_shape = (img_rows, img_cols, img_channel)

    # define network
    trainable_weights_graph = tf.Graph()
    with trainable_weights_graph.as_default():

        input_data = tf.placeholder(tf.float32, [None, img_rows, img_cols, img_channel], name='input_data')
        y_throttle = tf.placeholder(tf.float32, [None, ], name='y_throttle')
        y_steering = tf.placeholder(tf.float32, [None, ], name='y_steering')
        y_throttle_ = tf.placeholder(tf.float32, [None, ], name='y_throttle_')
        y_steering_ = tf.placeholder(tf.float32, [None, ], name='y_steering_')

        resnet = RESNET50(image_shape=input_shape, input_tensor=input_data)
        add_layer = resnet['activation_46']

        add_layer = tf.layers.conv2d(inputs=add_layer, filters=256, kernel_size=[3,3], padding='SAME', activation=tf.nn.elu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        add_layer = tf.layers.batch_normalization(inputs=add_layer)
        add_layer = tf.layers.max_pooling2d(inputs=add_layer, pool_size=[2, 2], strides=1, padding="SAME")
        add_layer = tf.layers.conv2d(inputs=add_layer, filters=256, kernel_size=[3, 3], strides=2, padding='SAME', activation=tf.nn.elu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        add_layer = tf.layers.batch_normalization(inputs=add_layer)
        add_layer = tf.layers.max_pooling2d(inputs=add_layer, pool_size=[2, 2], strides=1, padding="SAME")
        add_layer = tf.reshape(add_layer, [-1, 2 * 5 * 256])
        add_layer = tf.layers.dense(inputs=add_layer, units=256, activation=tf.nn.elu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        add_layer = tf.layers.batch_normalization(inputs=add_layer)
        add_layer = tf.layers.dense(inputs=add_layer, units=128, activation=tf.nn.elu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        throttle_output = tf.layers.dense(inputs=add_layer, units=1, activation=tf.sigmoid)
        steering_output = tf.layers.dense(inputs=add_layer, units=1, activation=tf.tanh)

        #throttle_loss = tf.reduce_mean(tf.square(throttle_output - y_throttle))
        steering_loss = tf.reduce_mean(tf.square(steering_output - y_steering))
        #final_loss = throttle_loss + steering_loss
        #traing_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss=steering_loss)
        traing_op = cocob.COCOB().minimize(steering_loss)

        #throttle_mse = tf.reduce_mean(tf.square(throttle_output - y_throttle_))
        steering_mse = tf.reduce_mean(tf.square(steering_output - y_steering_))
        #accuracy = throttle_mse + steering_mse

        tf.summary.scalar('loss', steering_loss)
        #tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()

    with tf.Session(graph=trainable_weights_graph) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        resnet.load_weights()

        writer = tf.summary.FileWriter(args.logdir, sess.graph)

        iter_cnt = 1
        print("### START TRAINING ###")
        for iter_ in range(1, ITERATIONS+1):
            num_steps = len(train_profiles) // BATCH_SIZE
            print("iteration: {}, steps: {}".format(iter_, num_steps))
            for step_ in range(num_steps):
                #with tf.device('/gpu:0'):
                with trainable_weights_graph.device('/device:GPU:0'):
                    # if step_ % 10 == 0:
                    #     acc = sess.run([steering_mse],
                    #             feed_dict={input_data: test_image_arr[:BATCH_SIZE],
                    #                           # y_throttle_: test_throttle_arr[:BATCH_SIZE],
                    #                           y_steering_: test_steering_arr[:BATCH_SIZE]})
                    #     print("accuracy: {}".format(acc))

                    summary, loss, _ = sess.run([merged, steering_loss, traing_op],
                                feed_dict={input_data: train_image_arr[
                                                       step_ * BATCH_SIZE: step_ * BATCH_SIZE + BATCH_SIZE],
                                           # y_throttle: train_throttle_arr[
                                           #             step_ * BATCH_SIZE: step_ * BATCH_SIZE + BATCH_SIZE],
                                           y_steering: train_steering_arr[
                                                       step_ * BATCH_SIZE: step_ * BATCH_SIZE + BATCH_SIZE]})
                    print("[{}-{}] loss: {}".format(iter_, step_, loss))
                    writer.add_summary(summary, iter_cnt)
                    iter_cnt += 1
            # Create a checkpoint in every iteration
            if iter_ % 100 == 0:
                saver.save(sess, os.path.join(args.savedir, 'model_iter'), global_step=iter_)

        tf.train.SummaryWriter(sess.graph)
        writer.close()
        # Save the final model
        saver.save(sess, os.path.join(args.savedir, 'model_final'))
        print("### END TRAINING ###")

# Get Image Array
def profile_to_centerimage_array(profile):
    img_array = []
    len_ = len(profile)
    for i in range(len_):
        img = cv2.imread(profile[i][0])
        img = preprocess_image(img)
        img_array.append(img)

    return img_array

# Get Answer Array of Throttle
def profile_to_throttle(profile):
    item_array = []
    len_ = len(profile)
    for i in range(len_):
        item_array.append(profile[i][4])
    return item_array


# Get Answer Array of Steering Angle
def profile_to_steering(profile):
    item_array = []
    len_ = len(profile)
    for i in range(len_):
        item_array.append(profile[i][3])
    return item_array


def preprocess_image(img):
    '''
    Method for preprocessing images: this method is the same used in drive.py, except this version uses
    BGR to YUV and drive.py uses RGB to YUV (due to using cv2 to read the image here, where drive.py images are
    received in RGB)
    '''
    # original shape: 160x320x3
    # crop to 85x320x3
    roi = img[50:135, 0:320,:]
    # apply subtle blur
    #roi = cv2.GaussianBlur(roi, (3, 3), 0)
    # convert to YUV color space (as nVidia paper suggests)
    ####### REMEMBER: IMAGES FROM SIMULATOR COME IN RGB!!!!!! #######
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2YUV)

    return roi


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trjdir', help='trajectory directory', default='trajectory')
    parser.add_argument('--logdir', help='log directory', default='log/train/behavioral')
    parser.add_argument('--savedir', help='save directory', default='trained_models/behavioral')
    parser.add_argument('--gamma', default=0.95, type=float)
    parser.add_argument('--iteration', default=int(1e4), type=int)
    return parser.parse_args()\

if __name__ == '__main__':
    args = argparser()
    main(args)