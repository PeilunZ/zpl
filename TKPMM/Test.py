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
import heapq
from Model import pixel_attention_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ai = '-7000'

midu = 15

len_min = 5

len_max = 25

class Config():
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
    lr = 0.003
    image_height = 256
    image_width = 128
    weight_decay = 0.0005
    init_scale = 0.04
    max_grad_norm = 15
    momentum = 0.9  #
    epoches = 5000
    data_dir = '/home/zpl/qikan/train_ban/train_ban'
    test_dir = '/home/zpl/qikan/test_ban/test_ban'
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

import re

######################################################################
def read_data(path, tongji, batch_size, order):
    batch_images = []
    labels = []
    document_x = []
    document_y = []
    document_x_z = []
    document_y_z = []
    for i in range(batch_size):
        target = order
        p = 0
        while True:
            p = p + 1
            while True:
                target_index = os.listdir(path + '/' + list(tongji.keys())[target])[int(random.random() * tongji[list(tongji.keys())[target]])]
                target_path = '%s/%s/%s' % (path, list(tongji.keys())[target], target_index)
                if not os.path.exists(target_path):
                    continue
                break
            barrage_x, x_num = take_barrage(target_index.split('.')[0], pattern, embedding)
            if x_num > midu:
                break
            if p > 50:
                return 'insufficient'
        a = random.randint(1, 100)
        if a <= 20:
            positive = True
            labels.append([1., 0.])
        else:
            positive = False
            labels.append([0., 1.])
        p = 0
        while True:
            p = p + 1
            tmp, text_x, text_y, text_x_z, text_y_z, y_num = get_pair_image(target,
                                                target_index, positive)
            if y_num > midu:
                document_x.append(text_x)
                document_y.append(text_y)
                document_x_z.append(text_x_z)
                document_y_z.append(text_y_z)
                break
            if p > 200:
                return 'insufficient'
        batch_images.append(tmp)
    return batch_images, np.array(labels), np.array(document_x), np.array(
        document_y), np.array(document_x_z), np.array(document_y_z)


def get_pair_image(target, target_index, positive):
    pair = []
    barrage_x, x_num = take_barrage(target_index.split('.')[0], pattern, embedding)
    barrage_x_z = take_zimu(target_index.split('.')[0], sgns)
    target_path = path_test + '/' + list(tongji_test.keys())[target] + '/' + target_index
    pair.append(target_path)
    if positive:
        while True:
            pick = random.choice(zheng)
            if pick == target_index:
                continue
            else:
                id = target
                break
    else:
        naixin = 0
        while True:
            naixin = naixin + 1
            id = int(random.random() * train_num_id_test)
            result_fu = '/home/zpl/Cascade-RCNN_Tensorflow/data/results_for_search_ban/' + list(tongji_test.keys())[id]
            file = open(result_fu, 'r')
            fu = file.readlines()
            file.close()
            fu = [x.split('|')[0] for x in fu]
            fu = [json.loads(x) for x in fu]
            pick = random.choice(fu)
            if pick in zheng:
                continue
            else:
                barrage_y, y_num = take_barrage(pick.split('.')[0], pattern, embedding)
                if y_num > midu or naixin > 10:
                    break
    barrage_y, y_num = take_barrage(pick.split('.')[0], pattern, embedding)
    filepath = '/home/zpl/Cascade-RCNN_Tensorflow/data/results_for_search_ban/' + list(tongji_test.keys())[id] + '|' + pick
    barrage_y_z = take_zimu(pick.split('.')[0], sgns)
    pair.append(filepath)
    return pair, barrage_x, barrage_y, barrage_x_z, barrage_y_z, y_num

def gallery_get(img):
    features = []
    dirpath = img.split('|')[0]
    index = img.split('|')[1]
    base_path = '/home/zpl/qikan/test_ban/test_ban/' + dirpath.split('/')[-1] + '/' + index
    file = open(dirpath, 'r')
    all = file.readlines()
    file.close()
    for line in all:
        if index in line:
            box = json.loads(line.split('|')[1])
            ROI = cv_imread(base_path)
            a, b, c = ROI.shape
            ROI = cv2.resize(ROI, (int(b*600/a), 600))
            ROI = ROI[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            ROI = cv2.resize(ROI, (config.image_width, config.image_height))
            features.append(ROI)
    return features

def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    return cv_img

def extract_from_anno(p, img):
    anno_path = '/home/zpl/qikan/bangumi/annotations/'
    xml_path = anno_path + p.split('/')[-2] + '/' + p.split('/')[-1].split('.')[0] + '.xml'
    file = open(xml_path, 'r')
    lines = file.readlines()
    file.close()
    for line in lines:
        if 'ymin' in line:
            ymin = max(int(float(line.split('>')[1].split('<')[0])), 0)
        elif 'ymax' in line:
            ymax = max(int(float(line.split('>')[1].split('<')[0])), 0)
        elif 'xmin' in line:
            xmin = max(int(float(line.split('>')[1].split('<')[0])), 0)
        elif 'xmax' in line:
            xmax = max(int(float(line.split('>')[1].split('<')[0])), 0)
        else:
            continue
    cropped = img[ymin:ymax, xmin:xmax]
    return cropped

######################################################################

def get_embedding():
    model_dir = '/home/zpl/Model_rnn/whole_256_3/model_longtime' + '-%d' % 16
    reader = pywrap_tensorflow.NewCheckpointReader(model_dir)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map.keys():
        if key == 'model/embedding':
            # print("tensor_name: ", key)
            embedding = reader.get_tensor(key)  # Remove this is you want to print only variable names
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
    # print('index')
    period = {}
    for i in range(1, config.period_num + 1):
        period[i] = ''
    time = int(index.split('_')[1]) * 0.50
    path = '/home/zpl/comic_person/barrage_all/'
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
    return document, count  # [4, 256]


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
            vector.append((feature / len(i)).tolist())
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
    path = '/home/zpl/comic_person/zimu_all/'
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
            vector.append((feature/len(i)).tolist())
    return vector

######################################################################
path = config.data_dir
path_test = config.test_dir

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
    if config.mode == 'train':
        config.batch_size = 1
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
        model_saver_1 = tf.train.Saver(var_list=var_list, max_to_keep=50)
        print('load model')
        model_saver_1.restore(session, config.output_new + 'model.ckpt' + ai)
        print('done')
        tongji_test = tong_ji(path_test)
        print(tongji_test.keys())
        train_num_id_test = get_num_id(tongji_test)
        print(train_num_id_test)
        tongji_gallery = tong_ji_gallery(config.gallery_dir)
        print(tongji_gallery.keys())
        train_num_id_gallery = get_num_id_gallery(tongji_gallery)
        print(train_num_id_gallery)
        acc = {}
        rec = {}
        pre = {}
        t_acc = {}
        t_acc[1] = 0
        t_acc[5] = 0
        t_acc[10] = 0
        thres = 0.3
        for l in range(5):
            acc[thres] = 0
            rec[thres] = 0
            pre[thres] = 0
            thres = round(thres + 0.1, 1)
        y = 0
        for k in range(train_num_id_test + 1):
            result_path = '/home/zpl/Cascade-RCNN_Tensorflow/data/results_for_search_ban/' + list(tongji_test.keys())[k]
            file = open(result_path, 'r')
            zheng = file.readlines()
            file.close()
            zheng = [x.split('|')[0] for x in zheng]
            zheng = [json.loads(x) for x in zheng]
            print('test char:')
            print(list(tongji_test.keys())[k])
            mean_acc = {}
            mean_rec = {}
            mean_pre = {}
            top_acc = {}
            top_acc[1] = 0
            top_acc[5] = 0
            top_acc[10] = 0
            thres = 0.3
            for l in range(5):
                mean_acc[thres] = 0
                mean_rec[thres] = 0
                mean_pre[thres] = 0
                thres = round(thres + 0.1, 1)
            jishu = 0
            for i in range(30):
                alll = read_data(path=config.test_dir, tongji=tongji_test, batch_size=50, order=k)
                if alll != 'insufficient':
                    jishu = jishu + 1
                    batch_images, batch_labels, batch_dx, batch_dy, batch_dx_z, batch_dy_z = alll
                    top_mat = np.zeros([50, 2])
                    count = 0
                    for images in batch_images:
                        image_1 = extract_from_anno(images[0], cv_imread(images[0]))
                        image_1 = cv2.resize(image_1, (config.image_width, config.image_height))
                        images_2 = gallery_get(images[1])
                        text_x = [batch_dx[count]]
                        text_x_z = [batch_dx_z[count]]
                        text_y = [batch_dy[count]]
                        text_y_z = [batch_dy_z[count]]
                        top_max = 0
                        for img_2 in images_2:
                            batch_data = np.array([image_1, img_2])
                            feed_dict = {mod.images: batch_data, mod.text_x: text_x,
                                         mod.text_y: text_y, mod.text_x_z: text_x_z, mod.text_y_z: text_y_z, mod.training: False}
                            train_loss = session.run([mod.soft_logit], feed_dict=feed_dict)
                            top_tmp = train_loss[0][0]
                            if top_tmp[0] > top_max:
                                top_mat[count] = top_tmp
                                top_max = top_tmp[0]
                        count = count + 1
                    prec = {}
                    recc = {}
                    accc = {}
                    top_mat_ban = top_mat[:, 0]
                    labels_ban = batch_labels[:, 0]
                    labels_ban2 = batch_labels[:, 1]
                    thres = 0.3
                    for l in range(5):
                        if np.sum(top_mat_ban > thres) == 0:
                            prec[thres] = 0
                        else:
                            prec[thres] = np.sum(labels_ban[top_mat_ban > thres]) / np.sum(top_mat_ban > thres)
                        recc[thres] = np.sum(labels_ban[top_mat_ban > thres]) / np.sum(labels_ban)
                        accc[thres] = (np.sum(labels_ban[top_mat_ban > thres]) + np.sum(labels_ban2[top_mat_ban < thres])) / 50
                        thres = round(thres + 0.1, 1)
                    zheng_score_index = list(top_mat[:, 0]).index(max(top_mat[:, 0]))
                    if batch_labels[:, 0][zheng_score_index] == 1:
                        top_acc[1] = top_acc[1] + 1
                    zheng_score_max_5 = heapq.nlargest(5, top_mat[:, 0])
                    for he in range(5):
                        if batch_labels[:, 0][list(top_mat[:, 0]).index(zheng_score_max_5[he])] == 1:
                            top_acc[5] = top_acc[5] + 1
                            break
                    zheng_score_max_10 = heapq.nlargest(10, top_mat[:, 0])
                    for he in range(10):
                        if batch_labels[:, 0][list(top_mat[:, 0]).index(zheng_score_max_10[he])] == 1:
                            top_acc[10] = top_acc[10] + 1
                            break
                # done!
                    thres = 0.30
                    for l in range(5):
                        mean_acc[thres] = mean_acc[thres] + accc[thres]
                        if recc[thres] == recc[thres]:
                            mean_rec[thres] = mean_rec[thres] + recc[thres]
                        if prec[thres] == prec[thres]:
                            mean_pre[thres] = mean_pre[thres] + prec[thres]
                        thres = round(thres + 0.1, 1)
            if jishu != 0:
                thres = 0.30
                for l in range(5):
                    mean_acc[thres] = mean_acc[thres]/(jishu)
                    mean_rec[thres] = mean_rec[thres]/(jishu)
                    mean_pre[thres] = mean_pre[thres]/(jishu)
                    thres = round(thres + 0.1, 1)
                y = y + 1
                print(mean_acc, mean_rec, mean_pre)
                thres = 0.30
                t_acc[1] = t_acc[1] + top_acc[1]/(jishu)
                t_acc[5] = t_acc[5] + top_acc[5]/(jishu)
                t_acc[10] = t_acc[10] + top_acc[10]/(jishu)
                for l in range(5):
                    acc[thres] = acc[thres] + mean_acc[thres]
                    rec[thres] = rec[thres] + mean_rec[thres]
                    pre[thres] = pre[thres] + mean_pre[thres]
                    thres = round(thres + 0.1, 1)
        thres = 0.30
        t_acc[1] = t_acc[1]/(y)
        t_acc[5] = t_acc[5]/(y)
        t_acc[10] = t_acc[10]/(y)
        f1 = {}
        for l in range(5):
            acc[thres] = acc[thres]/(y)
            rec[thres] = rec[thres]/(y)
            pre[thres] = pre[thres]/(y)
            f1[thres] = (rec[thres] * pre[thres] * 2)/(pre[thres] + rec[thres])
            thres = round(thres + 0.1, 1)

        print('final:')
        print('acc:', acc)
        print('rec:', rec)
        print('pre:', pre)
        print('top_n:', t_acc)
        print(f1)





























