import tensorflow as tf
from layers.func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net
import time
import os
import logging
import json
import numpy as np
from utils import compute_bleu_rouge, normalize


class Model(object):
    def __init__(self, config , opt=True):

        self.logger = logging.getLogger("brc")
        self.config = config



        #The start time of build model.
        self.start_t = time.time()
        self.c = tf.placeholder(tf.int32, shape=[None, None])   #context
        self.q = tf.placeholder(tf.int32, shape=[None, None])
        self.c_len = tf.placeholder(tf.int32, [None])
        self.q_len = tf.placeholder(tf.int32, [None])
        self.start_label = tf.placeholder(tf.int32, [None])
        self.end_label = tf.placeholder(tf.int32, [None])

        self.y1 = tf.one_hot(self.start_label, self.c_len)
        self.y2 = tf.one_hot(self.end_label, self.c_len)

        #是否是训练过程，用在dropout
        with tf.device("/cpu:0"):
            self.is_train = tf.get_variable(
                "is_train", dtype=tf.bool, trainable=False, initializer=tf.constant(True, dtype=tf.bool))
            #todo: where is the shape?
            self.word_mat = tf.get_variable("word_mat", shape=[self.config.vocab_size, self.config.embed_size], initializer=tf.random_uniform_initializer, dtype=tf.float32)

        #Todo:修改字典， span_id必须是0
        self.c_mask = tf.cast(self.c, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)

        # if opt:
        #     #Todo:batch_size 变量名对不对
        #     N= config.batch_size
        #     self.c_maxlen = tf.reduce_max(self.c_len)
        #     self.q_maxlen = tf.reduce_max(self.q_len)
        #     self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
        #     self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
        #     self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
        #     self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
        #     self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
        #     self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])
        # else:
        #     self.c_maxlen, self.q_maxlen = config.max_p_len, config.max_q_len

        self.c_maxlen, self.q_maxlen = config.max_p_len, config.max_q_len

        self.ready()
        self.end_t = time.time()
        self.logger.info("Time to build the model-{}: {} s".format(self, self.end_t-self.start_t))

        # if trainable:
        #     self.lr = tf.get_variable(
        #         "lr", shape=[], dtype=tf.float32, trainable=False)
        #     self.opt = tf.train.AdadeltaOptimizer(
        #         learning_rate=self.lr, epsilon=1e-6)
        #     grads = self.opt.compute_gradients(self.loss)
        #     gradients, variables = zip(*grads)
        #     capped_grads, _ = tf.clip_by_global_norm(
        #         gradients, config.grad_clip)
        #     self.train_op = self.opt.apply_gradients(
        #         zip(capped_grads, variables), global_step=self.global_step)

    def ready(self):
        config = self.config
        N, PL, QL, CL, d= config.batch_size, self.c_maxlen, self.q_maxlen, config.char_limit, config.hidden
        gru = cudnn_gru if config.use_cudnn else native_gru

        with tf.variable_scope("emb"):
            # maybe add char emb
            with tf.name_scope("word"):
                c_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
                q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)

            # c_emb = tf.concat([c_emb, ch_emb], axis=2)
            # q_emb = tf.concat([q_emb, qh_emb], axis=2)

        with tf.variable_scope("encoding"):
            #Todo: add num_layers para
            with tf.device("/cpu:0"):
                rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=c_emb.get_shape(
                ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            c = rnn(c_emb, seq_len=self.c_len)
            q = rnn(q_emb, seq_len=self.q_len)

        with tf.variable_scope("attention"):
            qc_att = dot_attention(c, q, mask=self.q_mask, hidden=d,
                                   keep_prob=config.keep_prob, is_train=self.is_train)
            with tf.device("/cpu:0"):
                rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=qc_att.get_shape(
                ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            att = rnn(qc_att, seq_len=self.c_len)

        with tf.variable_scope("match"):
            self_att = dot_attention(
                att, att, mask=self.c_mask, hidden=d, keep_prob=config.keep_prob, is_train=self.is_train)
            with tf.device("/cpu:0"):
                rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=self_att.get_shape(
                ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            match = rnn(self_att, seq_len=self.c_len)

        with tf.variable_scope("pointer"):
            init = summ(q[:, :, -2 * d:], d, mask=self.q_mask,
                        keep_prob=config.ptr_keep_prob, is_train=self.is_train)
            with tf.device("/cpu:0"):
                pointer = ptr_net(batch=N, hidden=init.get_shape().as_list(
                )[-1], keep_prob=config.ptr_keep_prob, is_train=self.is_train)
            logits1, logits2 = pointer(init, match, d, self.c_mask)

        with tf.variable_scope("predict"):
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, 15)
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits1, labels=self.y1)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits2, labels=self.y2)

            #maybe add 12 loss like baseline model.
            self.loss = tf.reduce_mean(losses + losses2)

    def get_loss(self):
        return self.loss


def tower_loss(batch, config):
    model = Model()

def train(data, vocab, epochs, batch_size, save_dir, save_prefix, dropout_keep_prob=1.0, evaluate=True):
    pad_id = vocab.get_id(vocab.pad_token)
    max_bleu_4=0


