"""
autoencoder.py
~~~~~~~~~~~~~~~

This module implements a word embedder using tensorflow NCE.

"""

import functools
import tensorflow as tf
import numpy as np


def tfboard_summaries(args):
    with tf.name_scope('summary'):
        pass

    
class Autoencoder:
    def __init__(
            self,
            corpus: str,
            embedding_size: int,
            learning_rate: float,
            num_epochs: int,
            input_size: int):

        """ 
        Initializer for class `Autoencoder`
        
        :param corpus: the corpus for embedding
        :param embedding_size: size of the hidden state
        :param learning_rate: learning rate
        :param num_epochs: number of epochs to train
        :param input_size: (decremented)

        """
        
        self.corpus = corpus
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = 150
        self.input_size = input_size
        self.logdir = '.'

    def preprocess(self):
        """ """
        pass

    @functools.lru_cache(maxsize=1)
    def embed(self, batcher_func):
        """
        Implementation of a Word2Vec form of autoencoder.

        :param batcher_func: a function that produces batches
        :returns A tensorflow/numpy array containing embeddings

        """
        with tf.Graph().as_default():
            with tf.name_scope('placeholders'):
                inputs = tf.placeholder(
                    tf.int32,
                    [self.batch_size],
                    name='inputs')
                labels = tf.placeholder(
                    tf.int32,
                    [self.batch_size, 1],
                    name='labels')

            with tf.device('/cpu:0'):
                with tf.name_scope('embeddings'):
                    embeddings = tf.get_variable(
                        'embeddings',
                        [self.input_size, self.embedding_size],
                        initializer=tf.truncated_normal_initializer(0., 1.))
                    embeddings_ = tf.nn.embedding_lookup(embeddings, inputs)

                with tf.name_scope('nce_weights'):
                    nce_w = tf.get_variable(
                        'nce_w',
                        [self.input_size, self.embedding_size],
                        tf.float32,
                        tf.random_normal_initializer(0., 1./np.sqrt(self.input_size)))
                    nce_b = tf.get_variable(
                        'nce_b',
                        [self.input_size],
                        tf.float32,
                        tf.constant_initializer(0.))

                with tf.name_scope('nce_loss'):
                    loss = tf.reduce_mean(
                        tf.nn.nce_loss(
                            nce_w,
                            nce_b,
                            labels,
                            embeddings_,
                            num_sampled=5,
                            num_classes=self.input_size))

                with tf.name_scope('optimizer'):
                    optimizer = tf.train.AdamOptimizer(
                        self.learning_rate).minimize(loss)

                with tf.Session() as sess:
s                   sess.run(tf.global_variables_initializer())

                    # tensorboard summaries
                    writer = tf.summary.FileWriter(self.logdir,
                                                   graph=tf.get_default_graph())
                    tf.summary.scalar('loss', loss)
                    summary_op = tf.summary.merge_all()
                    
                    average_loss = 0.0
                    num_steps = 1e5
                    for step in range(num_steps):
                        batch_inputs, batch_outputs = batcher_func(
                            self.corpus,
                            self.batch_size)
                        _, loss_value, summary = sess.run(
                            [optimizer, loss, summary_op],
                            feed_dict={inputs: batch_inputs,
                                       labels: batch_outputs})
                        average_loss += loss_value
                        writer.add_summary(summary, step)
                        if step % 100 == 0:
                            print(f'The average loss for step {step} was {loss_value}.')
                    return embeddings.eval()     

        def nearest(self, token, n, dist='euclidean') -> List[str]:
            if dist not in ('euclidean', 'mahalanobis', 'kldiv'):
                raise ValueError(f'Unknown distance metric: {dist}')

            if dist == 'euclidean':
                return 
