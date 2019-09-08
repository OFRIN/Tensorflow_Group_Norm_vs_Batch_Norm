import os
import sys
import cv2
import glob
import argparse
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from CNN import *

def one_hot(value, C = 5):
    vector = np.zeros((C), dtype = np.float32)
    vector[value] = 1
    return vector

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help = "batch_size=1/4/8/16/32")
    parser.add_argument('--norm_type', help = 'norm=batch/group')
    return parser.parse_args()

args = parser_args()
BATCH_SIZE = int(args.batch_size)

root_dir = './flower_dataset/'
data_dic = {'train' : [],
            'test' : []}

for domain in ['train', 'test']:
    sub_dir = root_dir + domain + '/'
    class_names = os.listdir(sub_dir)

    for label, class_name in enumerate(class_names):
        image_paths = glob.glob(sub_dir + class_name + '/*')
        
        for image_path in image_paths:
            data_dic[domain].append([image_path, one_hot(label)])

print('[i] train data : {}'.format(len(data_dic['train'])))
print('[i] test data : {}'.format(len(data_dic['test'])))

input_var = tf.placeholder(tf.float32, [None, 112, 112, 3])
label_var = tf.placeholder(tf.float32, [None, 5])
is_training = tf.placeholder(tf.bool)

logits, prediction_op = Model(input_var, is_training, args.norm_type)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_var))

correct_op = tf.equal(tf.argmax(prediction_op, 1), tf.argmax(label_var, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_op, tf.float32)) * 100

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(loss_op)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_data_list = data_dic['train']
test_data_list = data_dic['test']

train_iteration = len(data_dic['train']) // BATCH_SIZE
test_iteration = len(data_dic['test']) // BATCH_SIZE

f = open('./log/{}_{}.txt'.format(args.norm_type, BATCH_SIZE), 'w')

for epoch in range(1, 100 + 1):
    
    loss_list = []
    accuracy_list = []
    np.random.shuffle(train_data_list)

    for iter in range(train_iteration):
        sys.stdout.write('\r# [train] epoch = {}, [{}/{}]'.format(epoch, iter, train_iteration))
        sys.stdout.flush()

        batch_image_data = np.zeros((BATCH_SIZE, 112, 112, 3), dtype = np.float32)
        batch_label_data = np.zeros((BATCH_SIZE, 5), dtype = np.float32)

        batch_data_list = train_data_list[iter * BATCH_SIZE : (iter + 1) * BATCH_SIZE]
        for i, data in enumerate(batch_data_list):
            image_path, label = data

            image = cv2.imread(image_path)
            image = cv2.resize(image, (112, 112))

            batch_image_data[i] = image.copy()
            batch_label_data[i] = label.copy()

        _, loss = sess.run([train_op, loss_op], feed_dict = {input_var : batch_image_data, label_var : batch_label_data, is_training : True})
        loss_list.append(loss)
    print()

    for iter in range(test_iteration):
        sys.stdout.write('\r# [test] epoch = {}, [{}/{}]'.format(epoch, iter, test_iteration))
        sys.stdout.flush()

        batch_image_data = np.zeros((BATCH_SIZE, 112, 112, 3), dtype = np.float32)
        batch_label_data = np.zeros((BATCH_SIZE, 5), dtype = np.float32)

        batch_data_list = test_data_list[iter * BATCH_SIZE : (iter + 1) * BATCH_SIZE]
        for i, data in enumerate(batch_data_list):
            image_path, label = data

            image = cv2.imread(image_path)
            image = cv2.resize(image, (112, 112))

            batch_image_data[i] = image.copy()
            batch_label_data[i] = label.copy()

        accuracy = sess.run(accuracy_op, feed_dict = {input_var : batch_image_data, label_var : batch_label_data, is_training : False})
        accuracy_list.append(accuracy)
    print()

    loss = np.mean(loss_list)
    accuracy = np.mean(accuracy_list)
    print('# epoch = {}, loss = {:.4f}, accuracy = {:.0f}%'.format(epoch, loss, accuracy))

    f.write('{},{},{}\n'.format(epoch, loss, accuracy))
f.close()
