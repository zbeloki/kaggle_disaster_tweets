import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np

import argparse
import csv
import random
import sys

import pdb

ITERATIONS = 10
LEARNING_RATE = 0.1
NUM_EPOCHS = 15
LAMBD = 0.3

# TODOs:
# - tf.one_hot() erabili


def train(fpath):

    ops.reset_default_graph()
    
    entries = parse_input_data(fpath)

    w2i = create_word_dict(entries)
    n_x = len(w2i)
    
    X_dev, Y_dev = create_dataset(entries[:1000], w2i)
    X_test, Y_test = create_dataset(entries[1000:2000], w2i)
    X_train, Y_train = create_dataset(entries[2000:], w2i)
    
    # m = X.shape[1]
    # mu = (1/m) * np.sum(X, axis=1, keepdims=True)
    # std = np.sqrt((1/m) * np.sum(X * X, axis=1, keepdims=True))
    # X = (X - mu) / std

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    X = tf.placeholder(tf.float32, shape=[n_x, None], name='X')
    Y = tf.placeholder(tf.float32, shape=[1, None], name='Y')

    layer_lens = [n_x, 25, 12, 1]
    #layer_lens = [n_x, 1]
    
    W1 = tf.get_variable("W1", [layer_lens[1],layer_lens[0]], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [layer_lens[1],1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [layer_lens[2],layer_lens[1]], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [layer_lens[2],1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [layer_lens[3],layer_lens[2]], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [layer_lens[3],1], initializer=tf.zeros_initializer())

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.dropout(tf.nn.relu(Z1), keep_prob)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.dropout(tf.nn.relu(Z2), keep_prob)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
    
    init = tf.global_variables_initializer()

    np.set_printoptions(threshold=sys.maxsize)
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(NUM_EPOCHS):
            _, cost_ = sess.run([optimizer, cost], feed_dict={X:X_train, Y:Y_train, keep_prob: 0.7})
            print("Epoch: {} | Cost: {}".format(epoch, cost_))

        predictions = tf.round(tf.sigmoid(Z3))
        correct_prediction = tf.equal(predictions, Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train, keep_prob: 1.0}))
        print ("Test Accuracy:", accuracy.eval({X: X_dev, Y: Y_dev, keep_prob: 1.0}))
            

def parse_input_data(fpath):

    entries = []
    with open(fpath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_ = int(row['id'])
            text = ' '.join([row['keyword'], row['location'], row['text']]).lower()
            cls = int(row['target'])
            entries.append( (id_, text, cls) )

    random.shuffle(entries)

    return entries


def create_word_dict(entries):

    words = set()
    for _, text, _ in entries:
        words.update(text.split())

    word2index = {}
    for i, word in enumerate(words):
        word2index[word] = i

    return word2index
        

def create_dataset(entries, w2i):

    m = len(entries)
    n_x = len(w2i)

    X = np.zeros((n_x, m))
    Y = np.zeros((1, m))
    for i in range(m):
        text_words = set(entries[i][1].split())
        for word in text_words:
            word_index = w2i[word]
            X[word_index][i] = 1
        Y[0][i] = entries[i][2]

    return X, Y


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("train", help='Train CSV file')
    #parser.add_argument("model", help='Output model')
    args = parser.parse_args()

    train(args.train)
