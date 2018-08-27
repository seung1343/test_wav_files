# -*- coding: utf-8 -*-
# /usr/bin/python3

from __future__ import print_function

import tensorflow as tf
from tqdm import tqdm

from hparams import Hyperparams as hp
from networks import encoder, vq
from Wavenet_Decoder import decoder
from Wavenet_Decoder import generate
from Wavenet_Decoder import transposed_conv
from utils import mu_law_decode, get_wav
from scipy.io.wavfile import write
import numpy as np
from get_ivector import get_ivec
import os

class data:
    def __init__(self, mode="train"):
        '''
        Args:
          mode: Either "train" or "eval".
        '''
        # Set flag
        training = True if mode=="train" else False

        # Graph
        # Data Feeding
        ## x: Quantized wav. (B, T, 1) int32
        ## wavs: Raw wav. (B, length) float32
        ## speakers: Speaker ids. (B,). [0, 108]. int32.
        if mode == "train":
            # load training data
            self.data_paths = np.load('/home/shyou/Desktop/train_data.npy')
            self.data_num = len(self.data_paths)
            
            
            self.num_batch = self.data_num//hp.batch_size
            # self.ivectors = get_ivec()

    def get_batch(self):
        data = self.data_paths[:hp.batch_size, :]
        qts = []
        speakers = []
        for i in range(0,hp.batch_size):
            qt = np.load(data[i][0])
            qt = np.pad(qt, ([0, hp.T], [0, 0]), mode="constant", constant_values=0)[:hp.T, :]
            speaker = int(data[i][1])
            qts.append(qt)
            speakers.append(speaker)
        self.data_paths = np.concatenate((self.data_paths[hp.batch_size:, :], data))
        return qts, speakers


class Graph:
    def __init__(self, mode="train"):
        training = True if mode == "train" else False

        self.x = tf.placeholder(tf.int32, shape=[hp.batch_size, hp.T, 1])
        self.y = self.x
        self.encoder_inputs = tf.one_hot(tf.squeeze(self.x,axis=-1), hp.Q, dtype=tf.float32)
        self.speaker_id = tf.placeholder(tf.int32, shape=[hp.batch_size, ])
        self.speakers = tf.one_hot(self.speaker_id,len(hp.speakers), dtype=tf.float32)

        # encoder
        self.z_e = encoder(self.encoder_inputs)  # (B, T', D)

        # vq
        self.z_q = vq(self.z_e)  # (B, T', D)

        # decoder: y -> reconstructed logits.
        self.y_logits = decoder(self.encoder_inputs, self.speakers, self.z_q)  # (B, T-receptivefield+1, Q)
        # monitor
        # self.sample0 = tf.py_func(mu_law_decode, [self.y_hat[0]], tf.float32)
        # self.sample1 = tf.py_func(mu_law_decode, [self.y_hat[1]], tf.float32)

        # speech samples
        # tf.summary.audio('{}/original1'.format(mode), self.wavs[:1], hp.sr, 1)
        # tf.summary.audio('{}/original2'.format(mode), self.wavs[1:], hp.sr, 1)
        # tf.summary.audio('{}/sample0'.format(mode), tf.expand_dims(self.sample0, 0), hp.sr, 1)
        # tf.summary.audio('{}/sample1'.format(mode), tf.expand_dims(self.sample1, 0), hp.sr, 1)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if training:
            self.y = tf.slice(self.y, [0, hp.dilations[-1] * hp.size - 1, 0], [-1, -1, -1])
            self.y = tf.squeeze(self.y,axis=2)
            self.dec_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y_logits, labels=self.y))
            self.vq_loss = tf.reduce_mean(tf.squared_difference(tf.stop_gradient(self.z_e), self.z_q))
            self.enc_loss = hp.beta * tf.reduce_mean(tf.squared_difference(self.z_e, tf.stop_gradient(self.z_q)))
            # decoder grads
            decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder")
            decoder_grads = tf.gradients(self.dec_loss, decoder_vars)
            decoder_grads_vars = list(zip(decoder_grads, decoder_vars))

            # embedding variables grads
            embed_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "vq")
            embed_grads = tf.gradients(self.vq_loss, embed_vars)
            embed_grads_vars = list(zip(embed_grads, embed_vars))

            # encoder grads
            encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoder")
            transferred_grads = tf.gradients(self.dec_loss, self.z_q)
            encoder_grads = [tf.gradients(self.z_e, var, transferred_grads)[0] + tf.gradients(self.enc_loss, var)[0]
                             for var in encoder_vars]
            encoder_grads_vars = list(zip(encoder_grads, encoder_vars))

            # total grads
            grads_vars = decoder_grads_vars + embed_grads_vars + encoder_grads_vars

            # Training Scheme
            optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)

            # Summary
            tf.summary.scalar('train/dec_loss', self.dec_loss)
            tf.summary.scalar('train/vq_loss', self.vq_loss)
            tf.summary.scalar('train/enc_loss', self.enc_loss)

            # tf.summary.scalar("lr", self.lr)

            # gradient clipping
            for grad, var in grads_vars:
                if grad is not None:
                    self.clipped = [(tf.clip_by_value(grad, -1., 1.), var)]

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = optimizer.apply_gradients(self.clipped, global_step=self.global_step)

        # Summary
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(hp.logdir+'/train')
        if training == False:
            with tf.variable_scope('decoder'):
                self.z_q = transposed_conv(self.z_q)
        # audio = mu_law_decode(audio)
        # write(os.path.join(hp.sampledir, '{}.wav'.format('testfile')), hp.sr, audio)


if __name__ == '__main__':
    training_data = data()
    g = Graph(); print("Training Graph loaded")

    config = tf.ConfigProto()
    saver = tf.train.Saver(max_to_keep=30)
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is not None:
        saver.restore(sess, ckpt)
        print("checkpoint restored")
    else:
        sess.run(init)
        print("no checkpoint detected. Variables are initialized")
    step = 0
    for k in range(0,30):
        for _ in tqdm(range(training_data.num_batch), total=training_data.num_batch, ncols=70, leave=False, unit='b'):
            qts, speaker_embedding = training_data.get_batch()
            summary, gs, _ = sess.run([g.merged, g.global_step, g.train_op],feed_dict={g.x: qts, g.speaker_id: speaker_embedding})
            step += 1
            if step % training_data.num_batch == 0:
                saver.save(sess=sess, save_path=hp.logdir, global_step=gs)
                g.train_writer.add_summary(summary,step)
    print("Done")
