import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.contrib.layers import xavier_initializer
import numpy as np
from nltk.tokenize import word_tokenize

import argparse
import csv
import random
import pickle
import sys
import re
import math

import pdb

LEARNING_RATE = 0.002
NUM_EPOCHS = 20
DO_KEEP_RATE = 0.5
DEV_SIZE = 2500
BATCH_SIZE = 500
Tx = 22  # max sentence size


def main(train_fpath, test_fpath, out_fpath):

    wvecs = load_default_embeddings('embeddings/en_vocabulary.pickle', 'embeddings/en_embedding_matrix.npy')
    #wvecs = load_glove_embeddings('embeddings/glove.twitter.27B.50d.txt')
    n_v, n_x = wvecs[1].shape
    
    train_entries, test_entries = parse_input_data(train_fpath, test_fpath)

    X_dev, Y_dev = create_emb_dataset(train_entries[:DEV_SIZE], wvecs)
    X_train, Y_train = create_emb_dataset(train_entries[DEV_SIZE:], wvecs)
    X_test, _ = create_emb_dataset(test_entries, wvecs)

    tfp = {}  # Tensorflow parameters
    create_full_model(n_x, BATCH_SIZE, tfp)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        train(X_train, Y_train, sess, tfp)
        
        add_accuracy_to_model(tfp)
        
        accuracy_train = get_accuracy(X_train, Y_train, tfp)
        print ("Train Accuracy: {}".format(accuracy_train))

        accuracy_dev = get_accuracy(X_dev, Y_dev, tfp)
        print ("Dev Accuracy: {}".format(accuracy_dev))

        keys = ['Wc', 'Wu', 'Wf', 'Wo', 'Wy', 'b_c', 'b_u', 'b_f', 'b_o', 'b_y']
        params = { key: tfp[key] for key in keys }
        learnt_params = sess.run(params)

    # make predictions on test data

    if out_fpath is not None:
        m = X_test.shape[1]
        batch_size = m

        tfp = {}
        create_prediction_model(n_x, batch_size, tfp)
        add_accuracy_to_model(tfp)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            assign_trained_values(tfp, learnt_params, sess)

            test_pred = predict_test_data(X_test, batch_size, tfp)
            with open(out_fpath, 'w') as f:
                print("id,target", file=f)
                for i in range(len(test_entries)):
                    eid = test_entries[i][0]
                    pred = int(test_pred[i])
                    print("{},{}".format(eid, pred), file=f)


def create_full_model(n_x, batch_size, tfp):

    create_prediction_model(n_x, batch_size, tfp)

    tfp['logits'] = tf.transpose(tfp['out_z'])
    tfp['labels'] = tf.transpose(tfp['Y'])

    tfp['cost'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tfp['logits'], labels=tfp['labels']))
    tfp['optimizer'] = tf.train.AdamOptimizer(LEARNING_RATE).minimize(tfp['cost'])
    

def create_prediction_model(n_x, batch_size, tfp):

    ops.reset_default_graph()

    tfp['X'] = tf.placeholder(tf.float32, shape=[n_x, None, Tx])
    tfp['Y'] = tf.placeholder(tf.float32, shape=[1, None])
    tfp['keep_prob'] = tf.placeholder(tf.float32)
    
    h = 100
    tfp['Wc'] = tf.get_variable("Wc", [h, h + n_x], initializer=xavier_initializer())
    tfp['Wu'] = tf.get_variable("Wu", [h, h + n_x], initializer=xavier_initializer())
    tfp['Wf'] = tf.get_variable("Wf", [h, h + n_x], initializer=xavier_initializer())
    tfp['Wo'] = tf.get_variable("Wo", [h, h + n_x], initializer=xavier_initializer())
    tfp['Wy'] = tf.get_variable("Wy", [1, h], initializer=xavier_initializer())
    tfp['b_c'] = tf.get_variable("b_c", [h, 1], initializer=tf.zeros_initializer())
    tfp['b_u'] = tf.get_variable("b_u", [h, 1], initializer=tf.zeros_initializer())
    tfp['b_f'] = tf.get_variable("b_f", [h, 1], initializer=tf.zeros_initializer())
    tfp['b_o'] = tf.get_variable("b_o", [h, 1], initializer=tf.zeros_initializer())
    tfp['b_y'] = tf.get_variable("b_y", [1, 1], initializer=tf.zeros_initializer())

    tfp['C_prev'] = tf.get_variable("C_prev", [h, batch_size], initializer=tf.zeros_initializer())
    tfp['A_prev'] = tf.get_variable("A_prev", [h, batch_size], initializer=tf.zeros_initializer())

    tfp['A_prev'] = tf.zeros(tfp['A_prev'].shape)
    tfp['C_prev'] = tf.zeros(tfp['C_prev'].shape)
    for t in range(Tx):
        tfp['Xt'] = tfp['X'][:,:,t]
        lstm_cell(tfp)
        tfp['A_prev'] = tfp['A']
        tfp['C_prev'] = tfp['C']
    tfp['out_z'] = tf.add(tf.matmul(tfp['Wy'], tfp['A_prev']), tfp['b_y'])


def add_accuracy_to_model(tfp):

    tfp['predictions'] = tf.round(tf.sigmoid(tfp['out_z']))
    tfp['correct_prediction'] = tf.equal(tfp['predictions'], tfp['Y'])
    tfp['accuracy'] = tf.reduce_mean(tf.cast(tfp['correct_prediction'], "float"))
    

def lstm_cell(tfp):

    tfp['i_v'] = tf.concat([tfp['A_prev'], tfp['Xt']], 0)
    tfp['C_new'] = tf.math.tanh(tf.add(tf.matmul(tfp['Wc'], tfp['i_v']), tfp['b_c']))
    tfp['Gu'] = tf.math.sigmoid(tf.add(tf.matmul(tfp['Wu'], tfp['i_v']), tfp['b_u']))
    tfp['Gf'] = tf.math.sigmoid(tf.add(tf.matmul(tfp['Wf'], tfp['i_v']), tfp['b_f']))
    tfp['Go'] = tf.math.sigmoid(tf.add(tf.matmul(tfp['Wo'], tfp['i_v']), tfp['b_o']))
    tfp['C'] = tfp['Gu'] * tfp['C_new'] + tfp['Gf'] * tfp['C_prev']
    tfp['A'] = tf.nn.dropout(tfp['Go'] * tf.math.tanh(tfp['C']), DO_KEEP_RATE)


def train(X_data, Y_data, session, tfp):
    
    num_batches = int(X_data.shape[1] / BATCH_SIZE)
    for epoch in range(NUM_EPOCHS):
        print("Epoch: {}".format(epoch+1))
        for i_batch in range(num_batches):
            X_batch = X_data[: , i_batch*BATCH_SIZE:(i_batch+1)*BATCH_SIZE]
            Y_batch = Y_data[: , i_batch*BATCH_SIZE:(i_batch+1)*BATCH_SIZE]
            _, batch_cost = session.run([tfp['optimizer'], tfp['cost']],
                                        feed_dict={tfp['X']:X_batch,
                                                   tfp['Y']:Y_batch,
                                                   tfp['keep_prob']: DO_KEEP_RATE})
            print("Batch: {} | Cost: {}".format(i_batch, batch_cost))


def get_accuracy(X_data, Y_data, tfp):

    num_batches = int(X_data.shape[1] / BATCH_SIZE)
    accuracies = []
    for i_batch in range(num_batches):
        X_batch = X_data[: , i_batch*BATCH_SIZE:(i_batch+1)*BATCH_SIZE]
        Y_batch = Y_data[: , i_batch*BATCH_SIZE:(i_batch+1)*BATCH_SIZE]
        accuracies.append(tfp['accuracy'].eval({
            tfp['X']: X_batch,
            tfp['Y']: Y_batch,
            tfp['keep_prob']: 1.0
        }))

    return sum(accuracies) / len(accuracies)


def assign_trained_values(tfp, learnt_vals, sess):

    for pid in learnt_vals.keys():
        op = tf.assign(tfp[pid], learnt_vals[pid])
        sess.run(op)
    

def predict_test_data(X_test, batch_size, tfp):

    m = X_test.shape[1]
    test_pred = []
    num_batches = int(m / batch_size)
    for i_batch in range(num_batches):
        X_batch = X_test[: , i_batch*batch_size:(i_batch+1)*batch_size]
        ys = tfp['predictions'].eval({tfp['X']: X_batch, tfp['keep_prob']: 1.0})
        for y in ys[0]:
            test_pred.append(y)

    return test_pred
    
            
def load_default_embeddings(vocab_f, matrix_f):

    with open(vocab_f, 'rb') as f:
        vocab = pickle.load(f)
        vocab = { w: vocab[w][0] for w in vocab.keys() }

    E = np.load(matrix_f)

    return (vocab, E)


def load_glove_embeddings(fpath):

    vocab = {}
    vectors = []
    d = None
    with open(fpath, 'r') as f:
        i = 0
        for ln in f.readlines():
            ln_items = ln.split()
            if i == 0:
                d = len(ln_items)
            if len(ln_items) == d:
                vocab[ln_items[0]] = i
                vectors.append([ float(v) for v in ln_items[1:] ])
                i += 1

    E = np.array(vectors)

    return (vocab, E)


def parse_input_data(train_fpath, test_fpath):

    train_entries = []
    with open(train_fpath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_ = int(row['id'])
            text = ' '.join([row['keyword'], row['location'], row['text']]).lower()
            text = parse_text(text)
            cls = int(row['target'])
            train_entries.append( (id_, text, cls) )

    test_entries = []
    with open(test_fpath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_ = int(row['id'])
            text = ' '.join([row['keyword'], row['location'], row['text']]).lower()
            text = parse_text(text)
            test_entries.append( (id_, text, None) )

    random.shuffle(train_entries)

    return train_entries, test_entries


def parse_text(text):

    # remove hashtag #
    text = text.replace('#', '')
    # replace URLs with <url>
    text = re.sub(r'https?:\/\/[^ ]*', '<url>', text)
    # replace usernames with <user>
    #text = re.sub(r'@[^ ,.]*', '<user>', text)
    # replace numbers with <number>
    text = re.sub(r'\b[\d+]\.?[\d]*\b', '<number>', text)
    # replace emojis
    text = text.replace(':)', '<smile>')
    text = text.replace(':(', '<sadface>')
    text = text.replace('xD', '<lolface>')
    text = text.replace(':|', '<neutralface>')

    return text


def create_emb_dataset(entries, wvecs):

    vocab, E = wvecs

    m = len(entries)
    n_x = E.shape[1]
    unk_vec = np.mean(E, axis=0)
    
    X = np.zeros((n_x, m, Tx))
    Y = np.zeros((1, m))
    for i in range(m):
        text_words = word_tokenize(entries[i][1])
        for iw in range(len(text_words[:Tx])):
            word = text_words[iw]
            if word in vocab:
                widx = vocab[word]
                X[:,i,iw] = E[widx]
            else:
                X[:,i,iw] = unk_vec
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
    parser.add_argument("test", help='Test CSV file')
    parser.add_argument("--output", required=False, help='Output CSV file containing testset with labels')
    args = parser.parse_args()

    main(args.train, args.test, args.output)
