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

import pdb

LEARNING_RATE = 0.002
NUM_EPOCHS = 400
DO_KEEP_RATE = 0.75
DEV_SIZE = 3000


def main(train_fpath, test_fpath, out_fpath):

    #wvecs = load_default_embeddings('embeddings/en_vocabulary.pickle', 'embeddings/en_embedding_matrix.npy')
    wvecs = load_glove_embeddings('embeddings/glove.twitter.27B.50d.txt')
    m, n_x = wvecs[1].shape
    
    train_entries, test_entries = parse_input_data(train_fpath, test_fpath)

    X_dev, Y_dev = create_emb_dataset(train_entries[:DEV_SIZE], wvecs)
    X_train, Y_train = create_emb_dataset(train_entries[DEV_SIZE:], wvecs)
    X_test, _ = create_emb_dataset(test_entries, wvecs)

    ops.reset_default_graph()

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
    with tf.Session() as sess:

        sess.run(init)
        
        for epoch in range(NUM_EPOCHS):
            _, cost_ = sess.run([optimizer, cost], feed_dict={X:X_train, Y:Y_train, keep_prob: DO_KEEP_RATE})
            print("Epoch: {} | Cost: {}".format(epoch+1, cost_))

        predictions = tf.round(tf.sigmoid(Z3))
        correct_prediction = tf.equal(predictions, Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train, keep_prob: 1.0}))
        print ("Test Accuracy:", accuracy.eval({X: X_dev, Y: Y_dev, keep_prob: 1.0}))

        if out_fpath is not None:

            test_pred = predictions.eval({X: X_test, keep_prob: 1.0})
            
            with open(out_fpath, 'w') as f:
                print("id,target", file=f)
                for i in range(len(test_entries)):
                    eid = test_entries[i][0]
                    pred = int(test_pred[0][i])
                    print("{},{}".format(eid, pred), file=f)


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
    text = re.sub(r'@[^ ,.]*', '<user>', text)
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

    X = np.zeros((n_x, m))
    Y = np.zeros((1, m))
    for i in range(m):
        text_words = set(word_tokenize(entries[i][1]))
        for word in text_words:
            if word in vocab:
                widx = vocab[word]
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
    parser.add_argument("test", help='Test CSV file')
    parser.add_argument("--output", required=False, help='Output CSV file containing testset with labels')
    args = parser.parse_args()

    main(args.train, args.test, args.output)
