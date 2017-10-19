"""
autoencoder.py
~~~~~~~~~~~~~~~

This module implements a word embedder using tensorflow NCE.

"""

import functools
import tensorflow as tf
import numpy as np


class Autoencoder:
    def __init__(self,
                 corpus: str,
                 embedding_size: int,
                 learning_rate: float,
                 num_epochs: int,
                 input_size: int) -> None:
        """ """
        self.corpus = corpus
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = 150 # dix
        self.input_size = input_size

    def preprocess(self):
        """ """
        pass

    @functools.lru_cache(maxsize=1)
    def embed(self, batcher_func):
        """
        Implementation of a Word2Vec form of autoencoder.
        :returns A tensorflow/numpy array containing embeddings

        """
        with tf.Graph().as_default():
            with tf.name_scope('placeholders'):
                inputs = tf.placeholder(tf.int32, [self.batch_size], name='inputs')
                labels = tf.placeholder(tf.int32, [self.batch_size, 1], name='labels')

            with tf.device('/cpu:0'):  # multicore cpu if available
                with tf.name_scope('embeddings'):
                    embeddings = tf.get_variable('embeddings',
                                                 [self.input_size, self.embedding_size],
                                                 initializer=tf.truncated_normal_initializer(0., 1.))
                    embeddings_ = tf.nn.embedding_lookup(embeddings, inputs)

                with tf.name_scope('nce_weights'):
                    nce_w = tf.get_variable('nce_w',
                                            [self.input_size, self.embedding_size],
                                            tf.float32,
                                            tf.random_normal_initializer(0., 1./np.sqrt(self.input_size)))
                    nce_b = tf.get_variable('nce_b',
                                            [self.input_size],
                                            tf.float32,
                                            tf.constant_initializer(0.))

                with tf.name_scope('nce_loss'):
                    nce_loss = tf.nn.nce_loss(nce_w, nce_b, labels, embeddings_,
                                              num_sampled=5,
                                              num_classes=self.input_size)
                    loss = tf.reduce_mean(nce_loss)

                with tf.name_scope('optimizer'):
                    optimizer = (tf.train.AdamOptimizer(self.learning_rate)
                                 .minimize(loss))

                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    average_loss = 0.0
                    num_steps = 10000
                    for step in range(num_steps):
                        batch_inputs, batch_outputs = batcher_func(self.corpus, self.batch_size)
                        _, loss_value = sess.run([optimizer, loss],
                                                 feed_dict={inputs: batch_inputs,
                                                            labels: batch_outputs})
                        average_loss += loss_value
                        if step % 100 == 0:
                            print('The average loss for step {} was {}.'
                                  .format(step, loss_value))
                    return embeddings.eval()
