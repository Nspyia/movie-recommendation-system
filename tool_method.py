import pickle
import tensorflow as tf


def save_params(params):
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    return pickle.load(open('params.p', mode='rb'))


def compute_loss(labels, logits):
    return tf.reduce_mean(tf.keras.losses.mse(labels, logits))


def compute_metrics(labels, logits):
    return tf.keras.metrics.mae(labels, logits)  #