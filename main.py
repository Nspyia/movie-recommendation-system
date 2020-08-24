import pickle
import numpy as np
import tensorflow as tf
from test_fm import MyNetwork, movies, sentences_size, users, dnn_dims, mid2idx, idx2mid, movies_orig, users_orig


def get_user_movie_matrix():
    if not tf.io.gfile.exists('movie_matrix.p'):
        movie_layer_model = tf.keras.models.Model(inputs=[net.model.input[4], net.model.input[5]],
                                                  outputs=net.model.get_layer("movieDnnLayerDeep2").output)
        movie_matrix_list = []
        for item in movies.values:
            categories = np.zeros([1, 18])
            categories[0] = item.take(2)
            titles = np.zeros([1, sentences_size])
            titles[0] = item.take(1)
            movie_combine_layer_flat_val = movie_layer_model([np.reshape(item.take(0), [1, 1]), categories, titles])
            movie_matrix_list.append(movie_combine_layer_flat_val)
        pickle.dump((np.array(movie_matrix_list).reshape(-1, dnn_dims[1])), open('movie_matrix.p', 'wb'))
    movie_matrix_list = pickle.load(open('movie_matrix.p', mode='rb'))

    if not tf.io.gfile.exists('users_matrix.p'):
        user_layer_model = tf.keras.models.Model(
            inputs=[net.model.input[0], net.model.input[1], net.model.input[2], net.model.input[3]],
            outputs=net.model.get_layer("userDnnLayerDeep2").output)
        users_matrix_list = []
        for item in users.values:
            user_combine_layer_flat_val = user_layer_model([np.reshape(item.take(0), [1, 1]),
                                                            np.reshape(item.take(1), [1, 1]),
                                                            np.reshape(item.take(2), [1, 1]),
                                                            np.reshape(item.take(3), [1, 1])])
            users_matrix_list.append(user_combine_layer_flat_val)
        pickle.dump((np.array(users_matrix_list).reshape(-1, dnn_dims[1])), open('users_matrix.p', 'wb'))
    users_matrix_list = pickle.load(open('users_matrix.p', mode='rb'))
    return users_matrix_list, movie_matrix_list


# 预测用户-电影评分
def get_rating_for_user(my_model, user_id_val, movie_id_val):
    categories = np.zeros([1, 18])
    categories[0] = movies.values[mid2idx[movie_id_val]][2]
    titles = np.zeros([1, sentences_size])
    titles[0] = movies.values[mid2idx[movie_id_val]][1]
    inference_val = my_model([np.reshape(users.values[user_id_val - 1][0], [1, 1]),
                              np.reshape(users.values[user_id_val - 1][1], [1, 1]),
                              np.reshape(users.values[user_id_val - 1][2], [1, 1]),
                              np.reshape(users.values[user_id_val - 1][3], [1, 1]),
                              np.reshape(movies.values[mid2idx[movie_id_val]][0], [1, 1]),
                              categories,
                              titles])

    return inference_val.numpy()


# 预测用户-所有电影评分 返回list
def get_rating_list_for_user(my_model, user_id_val):
    n = movies.values.shape[0]
    categories = np.zeros([n, 18])
    for i in range(n):
        categories[i] = np.array(movies.values[i, 2]).astype(np.float32)

    titles = np.zeros([n, sentences_size])
    for i in range(n):
        titles[i] = np.array(movies.values[i, 1]).astype(np.float32)
    np_zeros = np.zeros([n, 1])
    result = my_model([(np.reshape(users.values[user_id_val - 1][0], [1, 1]) + np_zeros).astype(np.float32),
                       (np.reshape(users.values[user_id_val - 1][1], [1, 1]) + np_zeros).astype(np.float32),
                       (np.reshape(users.values[user_id_val - 1][2], [1, 1]) + np_zeros).astype(np.float32),
                       (np.reshape(users.values[user_id_val - 1][3], [1, 1]) + np_zeros).astype(np.float32),
                       np.reshape(movies.values[:, 0], [n, 1]).astype(np.float32),
                       np.reshape(categories, [n, 18]),
                       np.reshape(titles, [n, sentences_size])
                       ])
    result = result.numpy()[:, 0].argsort()
    return result


# 推荐同类型的电影
def recommend_same_type_movie(movie_id_val):
    norm_movie_matrix = tf.sqrt(tf.reduce_sum(tf.square(movie_matrix), 1, keepdims=True))
    normalized_movie_matrix = movie_matrix / norm_movie_matrix
    probs_embeddings = (movie_matrix[mid2idx[movie_id_val]]).reshape([1, dnn_dims[1]])
    probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_movie_matrix))
    sim = (probs_similarity.numpy())
    results = (-sim[0]).argsort()
    return results


# 推荐您喜欢的电影
def recommend_your_favorite_movie(user_id_val):
    probs_embeddings = (users_matrix[user_id_val - 1]).reshape([1, dnn_dims[1]])
    probs_similarity = tf.matmul(probs_embeddings, tf.transpose(movie_matrix))
    sim = (probs_similarity.numpy())
    results = (-sim[0]).argsort()
    return results


net = MyNetwork()
users_matrix, movie_matrix = get_user_movie_matrix()


print(get_rating_list_for_user(net.model, 1))
