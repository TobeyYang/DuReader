import tensorflow as tf
from layers.func import cudnn_gru, native_gru, dot_attention, summ, dropout, ptr_net
import time
import os
import logging
import json
import numpy as np
from utils import compute_bleu_rouge, normalize, read_and_decode_single_example


class Model(object):
    def __init__(self, config , contexts, questions, start_labels, end_labels, opt=True):

        self.logger = logging.getLogger("brc")
        self.config = config

        self.batch_size = tf.shape(start_labels)[0]


        self.start_t = time.time()
        self.c = contexts
        self.q = questions

        self.c_mask = tf.cast(self.c, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)

        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=-1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=-1)
        self.y1 = tf.one_hot(start_labels, config.max_p_len*config.max_p_num)
        self.y2 = tf.one_hot(end_labels, config.max_p_len*config.max_p_num)
        self.c_maxlen, self.q_maxlen = config.max_p_len, config.max_q_len

        #是否是训练过程，用在dropout
        with tf.device("/cpu:0"):
            self.is_train = tf.get_variable(
                "is_train", dtype=tf.bool, trainable=False, initializer=tf.constant(True, dtype=tf.bool))
            #todo: where is the shape?
            self.word_mat = tf.get_variable("word_mat", shape=[self.config.vocab_size, self.config.embed_size], initializer=tf.random_uniform_initializer, dtype=tf.float32)



        self.ready()
        self.end_t = time.time()
        self.logger.info("Time to build the model-{}: {} s".format(self, self.end_t-self.start_t))


    def ready(self):
        config = self.config
        N, d= config.batch_size, config.hidden_size
        gru = cudnn_gru if config.use_cudnn else native_gru
        concat_layers = True

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
            c = rnn(c_emb, seq_len=self.c_len, concat_layers=True)
            q = rnn(q_emb, seq_len=self.q_len, concat_layers=True)

        with tf.variable_scope("attention"):
            qc_att = dot_attention(c, q, mask=self.q_mask, hidden=d,
                                   keep_prob=config.keep_prob, is_train=self.is_train)
            with tf.device("/cpu:0"):
                rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=qc_att.get_shape(
                ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            att = rnn(qc_att, seq_len=self.c_len, concat_layers=True)

        with tf.variable_scope("match"):
            self_att = dot_attention(
                att, att, mask=self.c_mask, hidden=d, keep_prob=config.keep_prob, is_train=self.is_train)
            with tf.device("/cpu:0"):
                rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=self_att.get_shape(
                ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            match = rnn(self_att, seq_len=self.c_len, concat_layers=True)

            #concat the passages in the same document.
            match = tf.reshape(
                match,
                [self.batch_size, self.c_maxlen*self.config.max_p_num, d*2]
            )

        with tf.variable_scope("pointer"):
            init = summ(q[:, :, -2 * d:], d, mask=self.q_mask,
                        keep_prob=config.ptr_keep_prob, is_train=self.is_train)
            with tf.device("/cpu:0"):
                pointer = ptr_net(batch=N, hidden=init.get_shape().as_list(
                )[-1], keep_prob=config.ptr_keep_prob, is_train=self.is_train)

            #reshape the
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


def tower_loss(config, contexts, questions, start_labels, end_labels):
    logger = logging.getLogger("brc")
    model = Model(config, contexts, questions, start_labels, end_labels)
    tf.add_to_collection('models', model)
    return model.get_loss()

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train(config):

    with tf.Graph.as_default(), tf.device('/cpu:0')
        logger = logging.getLogger("brc")

        train_data_file = os.path.join(config.records_dir, "train.tfrecords")
        dev_data_file = os.path.join(config.records_dir, "dev.tfrecords")
        test_data_file = os.path.join(config.records_dir, "test.tfrecords")
        statistics_file = os.path.join(config.records_dir, "statistics.json")

        with open(statistics_file, 'r')as p:
            statistics = json.load(p)


        # opt = tf.train.AdamOptimizer(config.learning_rate)

        # logger.info("Training the model for epoch {}".format(epoch))
        context_ids, question_ids, y1, y2 = read_and_decode_single_example(train_data_file)
        context_batch, question_batch, start_batch, end_batch = tf.train.shuffle_batch(
            [context_ids, question_ids, y1, y2],
            batch_size=config.batch_size,
            capacity=500,
            min_after_dequeue=100,
            num_threads=4
        )
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([context_batch,
                                                                     question_batch,
                                                                     start_batch,
                                                                     end_batch])
        #
        # split_context_ids = tf.split(0, config.gpu_num, context_batch)
        # split_question_ids = tf.split(0, config.gpu_num, question_batch)
        # start_batch = tf.split(0, config.gpu_num, start_batch)
        # end_batch = tf.split(0, config.gpu_num, end_batch)

        opt = tf.train.AdamOptimizer(config.learning_rate)

        gpus = config.gpu.split(',')
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in config.gpu_num:
                with tf.device('/gpu:{}'.format(gpus[i])):
                    with tf.name_scope('tower {}'.format(i)) as scope:
                        context_batch, question_batch, start_batch, end_batch = batch_queue.dequeue()
                        loss = tower_loss(config, context_batch, question_batch, start_batch, end_batch)
                        tf.get_variable_scope().reuse_variables()
                        grads = opt.compute_gradients(loss)
                        tower_grads.append(grads)

        grads = average_gradients(tower_grads)

        train_op = opt.apply_gradients(grads)

        saver = tf.train.Saver(tf.global_variables())

        init = tf.global_variables_initializer()

        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True
        ))
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)

        for step in config.max_steps:
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time()-start_time

            assert not np.isnan(loss_value)

            if step %10 == 0:
                num_examples_per_step = config.batch_size*config.gpu_num
                examples_per_sec = num_examples_per_step/duration
                sec_per_batch = duration/config.gpu_num
                format_str = ('step {:d}, loss = {:.2f} ({:.1f} examples/sec; {:.3f} sec/batch)')
                logger.info(format_str.format(step, loss_value, examples_per_sec, sec_per_batch))

            if step % 1000 == 0 or (step+1) == config.max_steps:
                checkpoint_path = os.path.join(config.model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def evaluate(config, sess):
    logger = logging.getLogger("brc")
    logger.info('start to evaluate..')
    dev_data_file = os.path.join(config.records_dir, "dev.tfrecords")

    context_ids, question_ids, y1, y2 = read_and_decode_single_example(dev_data_file)
    context_batch, question_batch, start_batch, end_batch = tf.train.shuffle_batch(
        [context_ids, question_ids, y1, y2],
        batch_size=1,
        capacity=500,
        min_after_dequeue=100,
        num_threads=4
    )
    batch_queue = tf.contrib.slim.
    gpus = config.gpu.split(',')
    with tf.device('/gpu:{}'.format(gpus[0])):
        tf.get_variable_scope().reuse_variables()
        model = Model(config, context_batch, question_batch, start_batch, end_batch)








