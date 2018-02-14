"""
autoencoder.py
~~~~~~~~~~~~~~~

This module implements a word embedder using tensorflow NCE.

"""

import tensorflow as tf
import numpy as np
import collections
import random


class Autoencoder:
    def __init__(self, corpus, embedding_size, batch_size, learning_rate=0.01):
        """ Initialize an Autoencoder object.

        Parameters
        ----------
        corpus: list of int placeholders
            Corpus converted into ints.
        embedding_size: int
            The dimensionality of the embeddings (hidden state) required.
        batch_size: int
            Batch size.
        learning_rate: float, optional
            Learning rate, decaying.

        Returns
        -------
        None

        """
        self.corpus = corpus
        self.embedding_size = int(embedding_size)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.input_size = len(corpus)
        self.logdir = '.'
        self._data_index = 0

    @staticmethod
    def batcher(data, batch_size, num_skips=2, skip_window=2):
        """ Batcher function. Ripped from tensorflow's Word2Vec for now. """
        global data_index
        assert batch_size % num_skips == 0
        assert num_skips / 2 <= skip_window
        batch = np.ndarray(shape=[batch_size], dtype=np.int32)
        labels = np.ndarray(shape=[batch_size, 1], dtype=np.int32)
        span = 2 * skip_window + 1
        buf = collections.deque(maxlen=span)
        for _ in range(span):
            buf.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        for i in range(batch_size // num_skips):
            target = skip_window
            targets_to_avoid = [skip_window]
            for j in range(num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * num_skips + j] = buf[skip_window]
                labels[i * num_skips + j, 0] = buf[target]
            buf.append(data[data_index])
            data_index = (data_index + 1) % len(data)
        data_index = (data_index + len(data) - span) % len(data)
        return batch, labels

    def embed(self, batcher_func, *args, **kwargs):
        """
        Implementation of a Word2Vec form of autoencoder.

        Parameters
        ----------
        batcher_func: callable (unstable)
            A function that generates batches from the corpus
            i.e. from a list of ints.

        """
        with tf.Graph().as_default():
            with tf.name_scope('placeholders'):
                inputs = tf.placeholder(
                    tf.int32,
                    (self.batch_size),
                    name='inputs')
                labels = tf.placeholder(
                    tf.int32,
                    (self.batch_size, 1),
                    name='labels')

            with tf.device('/cpu:0'):

                # utilities for decaying the learning rate
                with tf.name_scope('util'):
                    _global_step = tf.train.create_global_step()
                    learning_rate = tf.train.exponential_decay(
                        self.learning_rate,
                        _global_step,
                        1000,
                        0.95)

                # embeddings
                with tf.name_scope('embeddings'):
                    embeddings = tf.get_variable(
                        'embeddings',
                        (self.input_size, self.embedding_size),
                        initializer=tf.truncated_normal_initializer(0., 1.))
                    embeddings_ = tf.nn.embedding_lookup(embeddings, inputs)

                # weights and bias for NCE sampler
                with tf.name_scope('nce_weights'):
                    nce_w = tf.get_variable(
                        'nce_w',
                        (self.input_size, self.embedding_size),
                        tf.float32,
                        tf.random_normal_initializer(
                            0.,
                            1./np.sqrt(self.input_size)))
                    nce_b = tf.get_variable(
                        'nce_b',
                        [self.input_size],
                        tf.float32,
                        tf.constant_initializer(0.))

                with tf.name_scope('nce_loss'):
                    loss = tf.reduce_mean(
                        tf.nn.nce_loss(
                            nce_w,  # nce weights
                            nce_b,  # nce biases
                            labels,
                            embeddings_,
                            num_sampled=10,
                            num_classes=self.input_size))

                with tf.name_scope('optimizer'):
                    optimizer = tf.train.AdamOptimizer(learning_rate) \
                                        .minimize(
                                            loss,
                                            global_step=_global_step)

                with tf.Session() as sess:
                    print('Initialized session...')
                    sess.run(tf.global_variables_initializer())

                    # tensorboard summaries
                    writer = tf.summary.FileWriter(
                        self.logdir,
                        graph=tf.get_default_graph())
                    tf.summary.scalar('loss', loss)
                    summary_op = tf.summary.merge_all()

                    average_loss = 0.0
                    num_steps = 10000
                    for step in range(num_steps):
                        batch_inputs, batch_outputs = Autoencoder.batcher(
                            self.corpus,
                            self.batch_size)
                        _, loss_value, summary = sess.run(
                            [optimizer, loss, summary_op],
                            feed_dict={inputs: batch_inputs,
                                       labels: batch_outputs})
                        average_loss += loss_value
                        writer.add_summary(summary, step)

                        if step % 100 == 0:
                            print(f'The average loss for step {step} '
                                  f'was {loss_value}.')

                    return embeddings.eval()


if __name__ == '__main__':
    pass
