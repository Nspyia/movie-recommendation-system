import matplotlib
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Flatten, Dense, concatenate
from fmlayer import FMLayer
import numpy as np
import tensorflow as tf
import time
import platform
import os
import pickle

title_count, title_set, genres2int, features, targets_values, \
ratings, users, movies, data, movies_orig, users_orig = pickle.load(open('preprocess.p', mode='rb'))

n_uid = max(features.take(0, 1)) + 1  # 用户ID个数 # 6040
n_gender = max(features.take(2, 1)) + 1  # 性别个数 # 1 + 1 = 2
n_age = max(features.take(3, 1)) + 1  # 年龄类别个数 # 6 + 1 = 7
n_job = max(features.take(4, 1)) + 1  # 职业个数 # 20 + 1 = 21
n_mid = max(features.take(1, 1)) + 1  # 电影ID个数 # 3952
n_m_genres = max(genres2int.values()) + 1  # 电影类型个数 # 18 + 1 = 19
n_movie_title = len(title_set)  # 电影名单词个数 # 5216
sentences_size = title_count  # 电影名长度 # = 15
mid2idx = {val[0]: i for i, val in enumerate(movies.values)}  # 电影ID转连续索引值
idx2mid = dict(zip(mid2idx.values(), mid2idx.keys()))
# 超参数
embed_dim = 16  # 嵌入矩阵的维度
num_epochs = 20
batch_size = 256
dropout_keep = 0.5
learning_rate = 0.0001
show_every_n_batches = 20  # Show stats for every n number of batches
save_dir = './save'

'True' 'False'
use_user_dnn = True
use_movie_dnn = True
use_user_movie_dnn = True
use_fm = True
dnn_dims = [256, 64]
assert not dnn_dims[0] == dnn_dims[1]
n_fm_factor = 256

str_record = ''


# 定义输入的占位符
def get_inputs():
    uid = tf.keras.layers.Input(shape=(1,), dtype='int32', name='uid')
    user_gender = tf.keras.layers.Input(shape=(1,), dtype='int32', name='user_gender')
    user_age = tf.keras.layers.Input(shape=(1,), dtype='int32', name='user_age')
    user_job = tf.keras.layers.Input(shape=(1,), dtype='int32', name='user_job')

    movie_id = tf.keras.layers.Input(shape=(1,), dtype='int32', name='movie_id')
    movie_categories = tf.keras.layers.Input(shape=(18,), dtype='int32', name='movie_categories')
    return uid, user_gender, user_age, user_job, movie_id, movie_categories


# 定义User的嵌入矩阵
def get_user_embedding(uid, user_gender, user_age, user_job):
    uid_embed_layer = tf.keras.layers.Embedding(n_uid, embed_dim, input_length=1, name='uid_embed_layer')(uid)
    gender_embed_layer = tf.keras.layers.Dense(1, activation='relu', name='gender_embed_layer')(user_gender)
    age_embed_layer = tf.keras.layers.Embedding(n_age, embed_dim // 2, input_length=1, name='age_embed_layer')(
        user_age)
    job_embed_layer = tf.keras.layers.Embedding(n_job, embed_dim // 2, input_length=1, name='job_embed_layer')(
        user_job)
    return uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer


# 定义Movie ID的嵌入矩阵
def get_movie_id_embed_layer(movie_id):
    movie_id_embed_layer = tf.keras.layers.Embedding(n_mid, embed_dim, input_length=1,
                                                     name='movie_id_embed_layer')(movie_id)
    return movie_id_embed_layer


# 合并电影类型的多个嵌入向量
def get_m_genres_embed_layers(movie_categories):
    movie_categories_embed_layer = tf.keras.layers.Embedding(n_m_genres, embed_dim // 2, input_length=18,
                                                             name='movie_categories_embed_layer')(movie_categories)
    movie_categories_embed_layer = tf.keras.layers.Lambda(lambda layer: tf.reduce_sum(layer, axis=1, keepdims=True))(
        movie_categories_embed_layer)
    return movie_categories_embed_layer


MODEL_DIR = "models_for_web"


def get_Flatten(layers):
    result = []
    for layer in layers:
        result.append(tf.keras.layers.Flatten()(layer))
    return result


def creat_dnn(input_layer, dims, name=''):
    for k in range(len(dims)):
        input_layer = Dense(dims[k], activation='relu', name=name + 'DnnLayerDeep' + str(k + 1))(input_layer)
    return Dense(1, activation='relu', name=name + 'DnnResult')(input_layer)


# 构建计算图
class MyNetwork(object):

    def __init__(self):
        self.batch_size = batch_size
        self.best_loss = 9999
        self.losses = {'train': [], 'test': []}

        # 获取输入占位符
        uid, user_gender, user_age, user_job, movie_id, movie_categories = get_inputs()
        # 获取User的4个嵌入向量
        uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = get_user_embedding(uid, user_gender,
                                                                                                   user_age, user_job)
        # 获取电影ID的嵌入向量
        mid_embed_layer = get_movie_id_embed_layer(movie_id)
        # 获取电影类型的嵌入向量
        m_genres_embed_layer = get_m_genres_embed_layers(movie_categories)

        # 所有嵌入层合并
        x_user_embed_combine = concatenate(get_Flatten([uid_embed_layer, gender_embed_layer, age_embed_layer,
                                                        job_embed_layer]), 1)
        x_movie_embed_combine = concatenate(get_Flatten([mid_embed_layer, m_genres_embed_layer]), 1)
        x_all_embed_combine = concatenate(get_Flatten([uid_embed_layer, gender_embed_layer, age_embed_layer,
                                                       job_embed_layer, mid_embed_layer, m_genres_embed_layer]), 1)

        merge_layers = []
        # user 的 DNN 部分
        if use_user_dnn:
            merge_layers.append(creat_dnn(x_user_embed_combine, dnn_dims, name='user'))
        # movie 的 DNN 部分
        if use_movie_dnn:
            merge_layers.append(creat_dnn(x_movie_embed_combine, dnn_dims, name='movie'))
        # user&movie 的 DNN 部分
        if use_user_movie_dnn:
            merge_layers.append(creat_dnn(x_all_embed_combine, dnn_dims, name='user_and_movie'))
        # DeepFM 的 FM 部分
        merge_layers.append(FMLayer(1, n_fm_factor)(x_all_embed_combine))
        # 合并 FM 和 DNN 并输出一个值
        dnn_fm_merge_layer = merge_layers[0]
        if len(merge_layers) > 1:
            dnn_fm_merge_layer = concatenate(merge_layers, 1)

        inference = Dense(1, activation='relu', name="output")(dnn_fm_merge_layer)

        self.model = tf.keras.Model(inputs=[uid, user_gender, user_age, user_job, movie_id, movie_categories
                                            ], outputs=[inference])
        self.model.summary()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.ComputeLoss = tf.keras.losses.MeanSquaredError()
        self.ComputeMetrics = tf.keras.metrics.MeanAbsoluteError()
        checkpoint_dir = os.path.join(MODEL_DIR, 'checkpoints')
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

        # Restore variables on creation if a checkpoint exists.
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_hat = self.model(x[:6], training=True)
            loss = self.ComputeLoss(y, y_hat)
            self.ComputeMetrics(y, y_hat)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, y_hat

    def training(self, features, targets_values, epochs=5, log_freq=50):

        for epoch_i in range(epochs):
            # 将数据集分成训练集和测试集，随机种子不固定
            train_X, test_X, train_y, test_y = train_test_split(features,
                                                                targets_values,
                                                                test_size=0.2,
                                                                random_state=0)

            train_batches = get_batches(train_X, train_y, self.batch_size)
            batch_num = (len(train_X) // self.batch_size)

            train_start = time.time()
            #             with self.train_summary_writer.as_default():
            if True:
                start = time.time()
                avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
                for batch_i in range(batch_num):
                    x, y = next(train_batches)
                    categories = np.zeros([self.batch_size, 18])
                    for i in range(self.batch_size):
                        categories[i] = x.take(6, 1)[i]

                    loss, y_hat = self.train_step([np.reshape(x.take(0, 1), [self.batch_size, 1]).astype(np.float32),
                                                   np.reshape(x.take(2, 1), [self.batch_size, 1]).astype(np.float32),
                                                   np.reshape(x.take(3, 1), [self.batch_size, 1]).astype(np.float32),
                                                   np.reshape(x.take(4, 1), [self.batch_size, 1]).astype(np.float32),
                                                   np.reshape(x.take(1, 1), [self.batch_size, 1]).astype(np.float32),
                                                   categories.astype(np.float32)],
                                                  np.reshape(y, [self.batch_size, 1]).astype(np.float32))
                    avg_loss(loss)
                    #                     avg_mae(metrics)
                    self.losses['train'].append(loss)

                    if tf.equal(self.optimizer.iterations % log_freq, 0):
                        rate = log_freq / (time.time() - start)
                        print('Step #{}\tEpoch {:>3} Batch {:>4}/{}   Loss: {:0.6f} mae: {:0.6f} ({} steps/sec)'.format(
                            self.optimizer.iterations.numpy(),
                            epoch_i,
                            batch_i,
                            batch_num,
                            loss, (self.ComputeMetrics.result()), rate))
                        avg_loss.reset_states()
                        self.ComputeMetrics.reset_states()
                        # avg_mae.reset_states()
                        start = time.time()

            train_end = time.time()
            print(
                '\nTrain time for epoch #{} ({} total steps): {}'.format(epoch_i + 1, self.optimizer.iterations.numpy(),
                                                                         train_end - train_start))
            global str_record
            if str_record.count('Epoch   ' + str(epoch_i) + ': '):
                str_record += (
                    '\nTrain time for epoch #{} ({} total steps): {}\n'.format(epoch_i + 1,
                                                                               self.optimizer.iterations.numpy(),
                                                                               train_end - train_start))
            #             with self.test_summary_writer.as_default():
            self.testing((test_X, test_y), self.optimizer.iterations)
            # self.checkpoint.save(self.checkpoint_prefix)
        self.export_path = os.path.join(MODEL_DIR, 'export')
        tf.keras.models.save_model(self.model, self.export_path)

    def testing(self, test_dataset, step_num):

        test_X, test_y = test_dataset
        test_batches = get_batches(test_X, test_y, self.batch_size)

        """Perform an evaluation of `model` on the examples from `dataset`."""
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        #         avg_mae = tf.keras.metrics.Mean('mae', dtype=tf.float32)

        batch_num = (len(test_X) // self.batch_size)
        for batch_i in range(batch_num):
            x, y = next(test_batches)
            categories = np.zeros([self.batch_size, 18])
            for i in range(self.batch_size):
                categories[i] = x.take(6, 1)[i]

            titles = np.zeros([self.batch_size, sentences_size])
            for i in range(self.batch_size):
                titles[i] = x.take(5, 1)[i]

            y_hat = self.model([np.reshape(x.take(0, 1), [self.batch_size, 1]).astype(np.float32),
                                np.reshape(x.take(2, 1), [self.batch_size, 1]).astype(np.float32),
                                np.reshape(x.take(3, 1), [self.batch_size, 1]).astype(np.float32),
                                np.reshape(x.take(4, 1), [self.batch_size, 1]).astype(np.float32),
                                np.reshape(x.take(1, 1), [self.batch_size, 1]).astype(np.float32),
                                categories.astype(np.float32),
                                titles.astype(np.float32)], training=False)
            test_loss = self.ComputeLoss(np.reshape(y, [self.batch_size, 1]).astype(np.float32), y_hat)
            avg_loss(test_loss)
            # 保存测试损失
            self.losses['test'].append(test_loss)
            self.ComputeMetrics(np.reshape(y, [self.batch_size, 1]).astype(np.float32), y_hat)
            # avg_loss(self.compute_loss(labels, y_hat))
            # avg_mae(self.compute_metrics(labels, y_hat))
        global str_record
        str_record += (
            'Model test set loss: {:0.6f} mae: {:0.6f}\n'.format(avg_loss.result(), self.ComputeMetrics.result()))
        print('Model test set loss: {:0.6f} mae: {:0.6f}'.format(avg_loss.result(), self.ComputeMetrics.result()))
        # print('Model test set loss: {:0.6f} mae: {:0.6f}'.format(avg_loss.result(), avg_mae.result()))
        #         summary_ops_v2.scalar('loss', avg_loss.result(), step=step_num)
        #         summary_ops_v2.scalar('mae', self.ComputeMetrics.result(), step=step_num)
        # summary_ops_v2.scalar('mae', avg_mae.result(), step=step_num)

        if avg_loss.result() < self.best_loss:
            self.best_loss = avg_loss.result()
            str_record += ("best loss = {}\n".format(self.best_loss))
            print("best loss = {}".format(self.best_loss))
            self.checkpoint.save(self.checkpoint_prefix)

    def forward(self, xs):
        predictions = self.model(xs)
        # logits = tf.nn.softmax(predictions)

        return predictions


# get batch
def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]

# test code
# mv_net.training(features, targets_values,epochs=1)
