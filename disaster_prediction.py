import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from nltk.tokenize import word_tokenize

import argparse
import csv
import random
import pickle
import sys

import pdb

LEARNING_RATE = 0.002
NUM_EPOCHS = 400
DEV_SIZE = 2200

def train(fpath):

    ops.reset_default_graph()

    wvecs = load_embeddings('embeddings/en_vocabulary.pickle', 'embeddings/en_embedding_matrix.npy')
    m, n_x = wvecs[1].shape
    
    entries = parse_input_data(fpath)

    X_dev, Y_dev = create_emb_dataset(entries[:DEV_SIZE], wvecs)
    X_train, Y_train = create_emb_dataset(entries[DEV_SIZE:], wvecs)
    
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    X = tf.placeholder(tf.float32, shape=[n_x, None], name='X')
    Y = tf.placeholder(tf.float32, shape=[1, None], name='Y')

    layer_lens = [n_x, 8, 4, 1]
    
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
            _, cost_ = sess.run([optimizer, cost], feed_dict={X:X_train, Y:Y_train, keep_prob: 0.75})
            print("Epoch: {} | Cost: {}".format(epoch+1, cost_))

        predictions = tf.round(tf.sigmoid(Z3))
        correct_prediction = tf.equal(predictions, Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train, keep_prob: 1.0}))
        print ("Test Accuracy:", accuracy.eval({X: X_dev, Y: Y_dev, keep_prob: 1.0}))


def load_embeddings(vocab_f, matrix_f):

    with open(vocab_f, 'rb') as f:
        vocab = pickle.load(f)

    E = np.load(matrix_f)

    return (vocab, E)
    

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


def create_emb_dataset(entries, wvecs):

    vocab, E = wvecs

    m = len(entries)
    n_x = E.shape[1]

    X = np.zeros((n_x, m))
    Y = np.zeros((1, m))
    for i in range(m):
        text_words = set(word_tokenize(entries[i][1]))
        for word in text_words:
            if word in vocab:
                widx = vocab[word][0]
                X[:,i] += E[widx]
            # else: if OOV, then add zero-vector (= do nothing)
        X[:,i] /= len(text_words)
        Y[0][i] = entries[i][2]

    return X, Y


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
