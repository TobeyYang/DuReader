#!python3


import tensorflow as tf

def read_and_decode_single_example(config, filename, is_test=False):


    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'passage_token_ids': tf.FixedLenFeature([], tf.string),
            'question_token_ids': tf.FixedLenFeature([], tf.string),
            'start_id': tf.FixedLenFeature([], tf.int64),
            'end_id': tf.FixedLenFeature([], tf.int64)
        }
    )
    context_ids = tf.reshape(tf.decode_raw(features['passage_token_ids'], tf.int32), [config.max_p_num, config.max_p_len])
    question_ids = tf.reshape(tf.decode_raw(features['question_token_ids'], tf.int32), [config.max_q_len])
    y1 = features['start_id']
    y2 = features['end_id']

    return context_ids, question_ids, y1, y2
