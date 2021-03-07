import tensorflow as tf
from tensorflow.python.framework import ops
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

    ops.reset_default_graph()

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    X = tf.placeholder(tf.float32, shape=[n_x, None, Tx], name='X')
    Y = tf.placeholder(tf.float32, shape=[1, None], name='Y')

    h = 100
    params = {}
    params['Wc'] = tf.get_variable("Wc", [h, h + n_x], initializer=tf.contrib.layers.xavier_initializer())
    params['Wu'] = tf.get_variable("Wu", [h, h + n_x], initializer=tf.contrib.layers.xavier_initializer())
    params['Wf'] = tf.get_variable("Wf", [h, h + n_x], initializer=tf.contrib.layers.xavier_initializer())
    params['Wo'] = tf.get_variable("Wo", [h, h + n_x], initializer=tf.contrib.layers.xavier_initializer())
    params['Wy'] = tf.get_variable("Wy", [1, h], initializer=tf.contrib.layers.xavier_initializer())
    params['b_c'] = tf.get_variable("b_c", [h, 1], initializer=tf.zeros_initializer())
    params['b_u'] = tf.get_variable("b_u", [h, 1], initializer=tf.zeros_initializer())
    params['b_f'] = tf.get_variable("b_f", [h, 1], initializer=tf.zeros_initializer())
    params['b_o'] = tf.get_variable("b_o", [h, 1], initializer=tf.zeros_initializer())
    params['b_y'] = tf.get_variable("b_y", [1, 1], initializer=tf.zeros_initializer())

    C_prev = tf.get_variable("C_prev", [h, BATCH_SIZE], initializer=tf.zeros_initializer())
    A_prev = tf.get_variable("A_prev", [h, BATCH_SIZE], initializer=tf.zeros_initializer())

    for t in range(Tx):
        Xt = X[:,:,t]
        A_prev, C_prev = lstm_cell(Xt, A_prev, C_prev, params)
    out_z = tf.add(tf.matmul(params['Wy'], A_prev), params['b_y'])

    logits = tf.transpose(out_z)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)

        train(X_train, Y_train, sess, optimizer, cost)
        
        predictions = tf.round(tf.sigmoid(out_z))
        correct_prediction = tf.equal(predictions, Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        accuracy_train = get_accuracy(X_train, Y_train, accuracy)
        print ("Train Accuracy: {}".format(accuracy_train))

        accuracy_dev = get_accuracy(X_dev, Y_dev, accuracy)
        print ("Dev Accuracy: {}".format(accuracy_dev))

        if out_fpath is not None:
            test_pred = []
            num_batches = math.ceil(X_test.shape[1] / BATCH_SIZE)
            for i_batch in range(num_batches):
                X_batch = np.zeros([n_x, BATCH_SIZE, Tx])
                i_first = i_batch * BATCH_SIZE
                i_last = min((i_batch + 1) * BATCH_SIZE, X_test.shape[1])
                X_batch[: , :i_last-i_first, :] = X_test[: , i_first:i_last, :]
                ys = predictions.eval({X: X_batch, keep_prob: 1.0})
                for y in ys[0]:
                    test_pred.append(y)
            with open(out_fpath, 'w') as f:
                print("id,target", file=f)
                for i in range(len(test_entries)):
                    eid = test_entries[i][0]
                    pred = int(test_pred[i])
                    print("{},{}".format(eid, pred), file=f)


def lstm_cell(Xt, A_prev, C_prev, params):

    i_v = tf.concat([A_prev, Xt], 0)
    C_new = tf.math.tanh(tf.add(tf.matmul(params['Wc'], i_v), params['b_c']))
    Gu = tf.math.sigmoid(tf.add(tf.matmul(params['Wu'], i_v), params['b_u']))
    Gf = tf.math.sigmoid(tf.add(tf.matmul(params['Wf'], i_v), params['b_f']))
    Go = tf.math.sigmoid(tf.add(tf.matmul(params['Wo'], i_v), params['b_o']))
    C = Gu * C_new + Gf * C_prev
    A = tf.nn.dropout(Go * tf.math.tanh(C), DO_KEEP_RATE)

    return A, C


def train(X_data, Y_data, session, optimizer, cost):

    X = tf.get_default_graph().get_tensor_by_name("X:0")
    Y = tf.get_default_graph().get_tensor_by_name("Y:0")
    keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")
    
    num_batches = int(X_data.shape[1] / BATCH_SIZE)
    for epoch in range(NUM_EPOCHS):
        print("Epoch: {}".format(epoch+1))
        for i_batch in range(num_batches):
            X_batch = X_data[: , i_batch*BATCH_SIZE:(i_batch+1)*BATCH_SIZE]
            Y_batch = Y_data[: , i_batch*BATCH_SIZE:(i_batch+1)*BATCH_SIZE]
            _, batch_cost = session.run([optimizer, cost], feed_dict={X:X_batch, Y:Y_batch, keep_prob: DO_KEEP_RATE})
            print("Batch: {} | Cost: {}".format(i_batch, batch_cost))


def get_accuracy(X_data, Y_data, accuracy):

    X = tf.get_default_graph().get_tensor_by_name("X:0")
    Y = tf.get_default_graph().get_tensor_by_name("Y:0")
    keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")

    num_batches = int(X_data.shape[1] / BATCH_SIZE)
    accuracies = []
    for i_batch in range(num_batches):
        X_batch = X_data[: , i_batch*BATCH_SIZE:(i_batch+1)*BATCH_SIZE]
        Y_batch = Y_data[: , i_batch*BATCH_SIZE:(i_batch+1)*BATCH_SIZE]
        accuracies.append(accuracy.eval({X: X_batch, Y: Y_batch, keep_prob: 1.0}))

    return sum(accuracies) / len(accuracies)
    
            
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
