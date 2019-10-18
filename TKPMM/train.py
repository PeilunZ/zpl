import numpy as np
import json
import os
import cv2
import tensorflow as tf
from tensorflow import pywrap_tensorflow
import re
import random
import tensorflow.contrib.slim as slim
import jieba

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

midu = 15

len_min = 3

len_max = 25

class Config():
    act_margin = 0.05
    _BATCH_NORM_DECAY = 0.0
    _BATCH_NORM_EPSILON = 1e-5
    data_format = 'channels_last'
    training = True
    pre_activation = True
    block_sizes = [3, 4, 6, 3]
    block_strides = [1, 2, 2, 2]
    num_filters = 64
    kernel_size = 3
    run_size = 256
    run_size_z = 300
    conv_stride = 2
    first_pool_size = 3
    first_pool_stride = 2
    hot = 1.0
    batch_size = 50
    lr = 0.001
    image_height = 256
    image_width = 128
    weight_decay = 0.0005
    init_scale = 0.04
    max_grad_norm = 15
    momentum = 0.9  #
    epoches = 5000
    data_dir = '/home/zpl/comic_person/train_ban'
    mode = 'train'
    load_dir = '/home/zpl/comic_person/output_new/'
    test_dir = '/home/zpl/comic_person/test_ban'
    output_new = '/home/zpl/comic_person/output_mixed/'
    period_before = -10
    period_after = 15
    period_num = 5
    period_dis = (period_after - period_before) / period_num
    period_before_z = -45
    period_after_z = 45
    period_num_z = 10
    period_dis_z = (period_after_z - period_before_z) / period_num_z


config = Config()

from Model import pixel_attention_model

import re

def get_train_pair(path, num_id, positive, tongji):
    pair = []
    if positive:
        value = int(random.random() * num_id)
        id = [value, value]
    else:
        while True:
            id = [int(random.random() * num_id), int(random.random() * num_id)]
            if id[0] != id[1]:
                break
    for i in range(2):
        filepath = ''
        while True:
            index = os.listdir(path + '/' + list(tongji.keys())[id[i]])[
                int(random.random() * tongji[list(tongji.keys())[id[i]]])]
            filepath = '%s/%s/%s' % (path, list(tongji.keys())[id[i]], index)
            if not os.path.exists(filepath):
                continue
            if i == 0:
                barrage_x, barrage_x_num = take_barrage(index.split('.')[0], pattern, embedding)
                barrage_x_z = take_zimu(index.split('.')[0], sgns)
            else:
                barrage_y, barrage_y_num = take_barrage(index.split('.')[0], pattern, embedding)
                barrage_y_z = take_zimu(index.split('.')[0], sgns)
            break
        pair.append(filepath)
    if barrage_x_num > midu and barrage_y_num > midu:
        choose = True
    else:
        choose = False
    return pair, barrage_x, barrage_y, barrage_x_z, barrage_y_z, choose

def get_easy_pair(path, num_id, positive, tongji):
    pair = []
    if positive:
        value = int(random.random() * num_id)
        id = [value, value]
    else:
        while True:
            id = [int(random.random() * num_id), int(random.random() * num_id)]
            if id[0] != id[1]:
                break
    for i in range(2):
        filepath = ''
        while True:
            index = os.listdir(path + '/' + list(tongji.keys())[id[i]])[int(random.random() * tongji[list(tongji.keys())[id[i]]])]
            filepath = '%s/%s/%s' % (path, list(tongji.keys())[id[i]], index)
            if not os.path.exists(filepath):
                continue
            if i == 0:
                barrage_x = np.zeros([config.period_num, 256]).tolist()
                barrage_x_z = take_zimu(index.split('.')[0], sgns)
            else:
                barrage_y = np.zeros([config.period_num, 256]).tolist()
                barrage_y_z = take_zimu(index.split('.')[0], sgns)
            break
        pair.append(filepath)
    return pair, barrage_x, barrage_y, barrage_x_z, barrage_y_z

def get_num_id(tongji):  # total number
    return len(tongji) - 1

def tong_ji(path):
    file_list = os.listdir(path)
    tongji = {}
    for i in file_list:
        tongji[i] = len(os.listdir(path + '/' + i))
    return tongji

def read_train_data(path, num_id, tongji, image_width, image_height, batch_size):
    batch_images = []
    labels = []
    document_x = []
    document_y = []
    document_x_z = []
    document_y_z = []
    for i in range(batch_size):
        images = []
        sam = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        if sam <= 2:
            pp = 0
            while True:
                pairs, text_x, text_y, text_x_z, text_y_z, choose = get_train_pair(path, num_id, True, tongji)
                pp = pp + 1
                if choose == True:
                    labels.append([1., 0.])
                    document_x.append(text_x)
                    document_y.append(text_y)
                    document_x_z.append(text_x_z)
                    document_y_z.append(text_y_z)
                    break
                elif pp > 5:
                    pairs, text_x, text_y, text_x_z, text_y_z = get_easy_pair(path, num_id, True, tongji)
                    labels.append([1., 0.])
                    document_x.append(text_x)
                    document_y.append(text_y)
                    document_x_z.append(text_x_z)
                    document_y_z.append(text_y_z)
                    break
        else:
            pp = 0
            while True:
                pairs, text_x, text_y, text_x_z, text_y_z, choose = get_pair(path, num_id, False, tongji)
                pp = pp + 1
                if choose == True:
                    labels.append([0., 1.])
                    document_x.append(text_x)
                    document_y.append(text_y)
                    document_x_z.append(text_x_z)
                    document_y_z.append(text_y_z)
                    break
                elif pp > 5:
                    pairs, text_x, text_y, text_x_z, text_y_z = get_pair_easy(path, num_id, False, tongji)
                    labels.append([0., 1.])
                    document_x.append(text_x)
                    document_y.append(text_y)
                    document_x_z.append(text_x_z)
                    document_y_z.append(text_y_z)
                    break
        for p in pairs:
            image = cv_imread(p)
            image = cv2.resize(image, (image_width, image_height))
            images.append(image)
        batch_images.append(images)
    print(np.array(document_x).shape)
    return np.transpose(np.array(batch_images), (1, 0, 2, 3, 4)), np.array(labels), np.array(document_x), np.array(
        document_y), np.array(document_x_z), np.array(document_y_z)


######################################################################
def read_test_data(path, num_id, tongji, image_width, image_height, batch_size):
    batch_images = []
    labels = []
    document_x = []
    document_y = []
    document_x_z = []
    document_y_z = []
    for i in range(batch_size):
        while True:
            target = int(random.random() * num_id)
            while True:
                target_index = os.listdir(path + '/' + list(tongji.keys())[target])[int(random.random() * tongji[list(tongji.keys())[target]])]
                target_path = '%s/%s/%s' % (path, list(tongji.keys())[target], target_index)
                if not os.path.exists(target_path):
                    continue
                break
            barrage_x, x_num = take_barrage(target_index.split('.')[0], pattern, embedding)
            if x_num > midu:
                break
        a = random.randint(1, 100)
        if a <= 20:
            positive = True
            labels.append([1., 0.])
        else:
            positive = False
            labels.append([0., 1.])
        while True:
            pairs, text_x, text_y, text_x_z, text_y_z, y_num = get_test_pair(target, target_index, path, num_id, positive, tongji)
            if y_num > midu:
                document_x.append(text_x)
                document_y.append(text_y)
                document_x_z.append(text_x_z)
                document_y_z.append(text_y_z)
                break
        images = []
        for p in pairs:
            image = cv_imread(p)
            image = cv2.resize(image, (image_width, image_height))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        batch_images.append(images)
    return np.transpose(np.array(batch_images), (1, 0, 2, 3, 4)), np.array(labels), np.array(document_x), np.array(
        document_y), np.array(document_x_z), np.array(document_y_z)

def get_test_pair(target, target_index, path, num_id, positive, tongji):
    pair = []
    barrage_x, x_num = take_barrage(target_index.split('.')[0], pattern, embedding)
    barrage_x_z = take_zimu(target_index.split('.')[0], sgns)
    target_path = path + '/' + list(tongji.keys())[target] + '/' + target_index
    pair.append(target_path)
    if positive:
        value = target
        id = [value, value]
    else:
        while True:
            id = int(random.random() * num_id)
            if id != target:
                id = [target, id]
                break
    filepath = ''
    while True:
        index = os.listdir(path + '/' + list(tongji.keys())[id[1]])[
            int(random.random() * tongji[list(tongji.keys())[id[1]]])]
        filepath = '%s/%s/%s' % (path, list(tongji.keys())[id[1]], index)
        if not os.path.exists(filepath):
            continue
        barrage_y, y_num = take_barrage(index.split('.')[0], pattern, embedding)
        barrage_y_z = take_zimu(index.split('.')[0], sgns)
        break
    pair.append(filepath)
    return pair, barrage_x, barrage_y, barrage_x_z, barrage_y_z, y_num


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    return cv_img

######################################################################

def get_embedding():
    model_dir = '/home/zpl/Model_rnn/whole_256_3/model_longtime' + '-%d' % 16
    reader = pywrap_tensorflow.NewCheckpointReader(model_dir)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map.keys():
        # print(var_to_shape_map[key])
        if key == 'model/embedding':
            # print("tensor_name: ", key)
            embedding = reader.get_tensor(key)
            return embedding

def pattern_pick(pattern, text):
    for l in pattern:
        try:
            m = re.search(l, text)
            if m is not None:
                return False
        except:
            pass
    return True

def vocab_take():
    vocab = {}
    file = open('/home/zpl/whole_danmu_voc_100.txt', 'r', encoding='utf-8')
    vocab_all = file.readlines()
    for x in vocab_all:
        vocab[x[0]] = int(x[2:-1])
    file.close()
    return vocab

def index_take():
    index_list = {}
    for k in vocab.keys():
        index_list[vocab[k]] = k
    return index_list

def take_barrage(index, pattern, embedding):
    document = []
    period = {}
    for i in range(1, config.period_num + 1):
        period[i] = ''
    time = int(index.split('_')[1]) * 0.5
    path = '/home/zpl/comic_person/danmu_moviee/'
    file = open(path + str(index.split('_')[0]) + '.json', 'r')
    barrage_all = file.readlines()
    file.close()
    count = 0
    for barrage in barrage_all:
        barrage = json.loads(barrage)
        text = barrage['text']
        for i in range(1, config.period_num + 1):
            if (barrage['time'] - len(barrage['text']) * 0.5 - time) < (
                config.period_before + config.period_dis * i) and (
                    barrage['time'] - len(barrage['text']) * 0.5 - time) > (
                config.period_before + config.period_dis * (i - 1)) and len(text) > len_min and len(text) < len_max:
                period[i] = period[i] + text
                count = count + 1
    for i in range(1, config.period_num + 1):
        document.append(period[i])
    document = convert_to_vector(document, index, embedding)
    return document, count

def convert_to_vector(document, index, embedding):
    vector = []
    for i in document:
        if i == '':
            feature = [0] * config.run_size
            vector.append(feature)
        else:
            feature = np.array([0] * config.run_size)
            for char in i:
                if char in vocab.keys():
                    temp = vocab[char]
                    feature = feature + embedding[temp]
            vector.append(feature/len(i).tolist())
    return vector

embedding = get_embedding()
vocab = vocab_take()
print(len(vocab))
index_list = index_take()

######################################################################

file = open('/home/zpl/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5.txt', 'r', encoding='utf-8')
file.readline()
word_char_all = file.readlines()
file.close()
sgns = {}

for i in range(len(word_char_all)):
    index = word_char_all[i].find(' ')
    lis = word_char_all[i][(index + 1):].split(' ')[:-1]
    char_feature = np.array([float(a) for a in lis])
    sgns[word_char_all[i][:index]] = char_feature

print('sgns load down!')

def take_zimu(index, sgns):
    document_z = []
    period_z = {}
    for i in range(1, config.period_num_z + 1):
        period_z[i] = ''
    time = int(index.split('_')[1]) * 0.50
    path = '/home/zpl/comic_person/zimu_movie/'
    file = open(path + str(index.split('_')[0]) + '.ass', 'r')
    zimu_all = file.readlines()
    file.close()
    for zimu in zimu_all:
        zimu = json.loads(zimu)
        text = zimu['text']
        for i in range(1, config.period_num_z + 1):
            if (zimu['start'] - time) < (config.period_before_z + config.period_dis_z * i) and (zimu['start'] - time) > (config.period_before_z + config.period_dis_z * (i - 1)):
                period_z[i] = period_z[i] + text
    for i in range(1, config.period_num_z + 1):
        document_z.append(period_z[i])
    document_z = convert_to_vector_zimu(document_z, index, sgns)
    return document_z

def convert_to_vector_zimu(document, index, sgns):
    vector = []
    for i in document:
        if i == '':
            feature = [0] * config.run_size_z
            vector.append(feature)
        else:
            i = ' '.join(jieba.cut(i)).split(' ')
            feature = np.array([0] * config.run_size_z)
            for char in i:
                if char in sgns.keys():
                    temp = sgns[char]
                    feature = feature + temp
            vector.append(feature/len(i).tolist())
    return vector

######################################################################
path_test = config.test_dir

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
    if config.mode == 'train':
        initializer = tf.random_uniform_initializer(-Config.init_scale,
                                                    Config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mod = pixel_attention_model(config=config)
        tf.global_variables_initializer().run()
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars = bn_moving_vars + [g for g in g_list if 'moving_variance' in g.name]
        var_list = var_list + bn_moving_vars
        model_saver_1 = tf.train.Saver(var_list=var_list, max_to_keep=100)
        var_list = [k for k in var_list if 'fc_to_text' not in k.name]
        var_list = [k for k in var_list if 'fc_to_text_z' not in k.name]
        var_list = [k for k in var_list if 'fc_embedding' not in k.name]
        var_list = [k for k in var_list if 'semantic_bn' not in k.name]
        var_list = [k for k in var_list if 'fc1' not in k.name]
        var_list = [k for k in var_list if 'fc2' not in k.name]
        model_saver_2 = tf.train.Saver(var_list=var_list, max_to_keep=100)
        print('load model')
        model_saver_2.restore(session, config.load_dir + 'model_shoulian.ckpt' + '-26000')
        print('done')
        tongji = tong_ji(path)
        train_num_id = get_num_id(tongji)
        tongji_test = tong_ji(path_test)
        train_num_id_test = get_num_id(tongji_test)
        for i in range(config.epoches):
            print('load data')
            batch_images, batch_labels, batch_document_x, batch_document_y, batch_document_x_z, batch_document_y_z = read_train_data(path=config.data_dir,
                                                                                       num_id=train_num_id,
                                                                                       tongji=tongji,
                                                                                       image_width=mod.image_width,
                                                                                       image_height=mod.image_height,
                                                                                       batch_size=mod.batch_size)
            print(batch_images.shape)
            batch_data = np.zeros(
                [batch_images.shape[1] * 2, batch_images.shape[2], batch_images.shape[3], batch_images.shape[4]])
            batch_data[::2, :, :, :] = batch_images[0]
            batch_data[1::2, :, :, :] = batch_images[1]
            print(batch_data.shape)
            print(batch_document_x.shape)
            print(batch_document_x_z.shape)
            print("Training Epoch: %d ..." % (i + 1))
            lr = config.lr * ((0.0001 * i + 1) ** -0.75)
            feed_dict = {mod.lr: lr, mod.images: batch_data, mod.text_x: batch_document_x,
                         mod.text_y: batch_document_y, mod.text_x_z: batch_document_x_z, mod.text_y_z: batch_document_y_z,
                         mod.labels: batch_labels, mod.training: True}
            session.run(mod.train, feed_dict=feed_dict)
            train_loss = session.run([mod.cross_entropy, mod.l2_loss], feed_dict=feed_dict)
            print('epoch:', i)
            print('training loss:', train_loss)
            # test~
            if i % 50 == 0:
                batch_images_test, batch_labels_test, batch_test_x, batch_test_y, batch_test_x_z, batch_test_y_z = read_test_data(
                    path=config.test_dir, num_id=train_num_id_test, tongji=tongji_test,
                    image_width=mod.image_width, image_height=mod.image_height,
                    batch_size=mod.batch_size)
                batch_data_test = np.zeros([batch_images_test.shape[1] * 2, batch_images_test.shape[2],
                                            batch_images_test.shape[3], batch_images_test.shape[4]])
                batch_data_test[::2, :, :, :] = batch_images_test[0]
                batch_data_test[1::2, :, :, :] = batch_images_test[1]
                feed_dict_test = {mod.images: batch_data_test, mod.text_x: batch_test_x, mod.text_y: batch_test_y,
                                  mod.text_x_z: batch_test_x_z, mod.text_y_z: batch_test_y_z,
                                  mod.labels: batch_labels_test, mod.training: False}
                test_loss = session.run([mod.cross_entropy, mod.l2_loss, mod.accuracy, mod.recall, mod.precision],
                                        feed_dict=feed_dict_test)
                print("Testing Epoch: %d ..." % (i + 1))
                print('Step: %d, Learning rate: %f, Test loss:' % (i, lr), test_loss)
            if (i) % 1000 == 0:
                model_saver_1.save(session, config.output_new + 'model.ckpt', i)






























